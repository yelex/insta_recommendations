"""text-aggregation: GLM (текстовая модель) собирает JSON локации из
caption/transcript/vision-данных (SKILLS.md, скилл 5).

Тот же ключ GLM Coding Plan, что и в vision.py (см. AGENTS.md, раздел
"какой API-ключ использовать"), но текстовая модель glm-5, а не vision
glm-4.6v.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Literal

import requests

logger = logging.getLogger(__name__)

GLM_TEXT_MODEL = "glm-5"

# Порог уверенности для автосохранения без переспроса пользователя —
# единственное место, где задаётся это число (AGENTS.md, "не размазывать
# магические числа по коду"). Используется здесь и в будущем
# user-clarification/orchestrator.
CONFIDENCE_THRESHOLD = 0.6

REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 1.0

PlaceType = Literal["природа", "еда", "жильё", "достопримечательность", "другое"]
_ALLOWED_PLACE_TYPES: set[str] = {"природа", "еда", "жильё", "достопримечательность", "другое"}

AGGREGATION_SYSTEM_PROMPT = (
    "Ты помогаешь агрегировать данные о месте из путешествия по Дагестану. "
    "На входе — подпись к видео, транскрипт речи, текст с оверлеев на кадрах и "
    "распознанное на кадрах название места (любое из полей может отсутствовать). "
    "Собери из них единую структурированную запись о месте.\n"
    "Верни СТРОГО JSON без пояснений и без markdown-обрамления, в формате:\n"
    '{"name": "строка или null", "region": "строка или null", '
    '"place_type": "природа|еда|жильё|достопримечательность|другое", '
    '"confidence": 0.0}\n'
    "confidence — твоя уверенность в правильности name и region (0.0-1.0). "
    "Если ничего не удаётся определить — не выдумывай, ставь null и низкий confidence.\n"
    "\n"
    "ВАЖНО:\n"
    "- name — это название КОНКРЕТНОГО места (заведение, гора, каньон, музей). "
    "НЕ используй название города или региона как name. "
    "Если из источников понятно только название города (Дербент, Махачкала) — "
    "поставь name: null, а город уйди в region.\n"
    "- Если text-overlay содержит только название города — это не место, а регион.\n"
    "- region — город, район или республика (Дагестан, Махачкала, Дербент, Гунибский район)."
)


class AggregationError(RuntimeError):
    """GLM text API недоступен или вернул нераспознаваемый ответ."""


@dataclass
class AggregationInput:
    caption: str | None = None
    transcript: str | None = None
    overlay_text: str | None = None
    recognized_place: str | None = None


@dataclass
class AggregatedLocation:
    name: str | None
    region: str | None
    place_type: PlaceType
    confidence: float
    raw_sources: list[str] = field(default_factory=list)


def _build_user_message(data: AggregationInput) -> str:
    lines: list[str] = []
    if data.caption:
        lines.append(f"Подпись: {data.caption}")
    if data.transcript:
        lines.append(f"Транскрипт: {data.transcript}")
    if data.overlay_text:
        lines.append(f"Текст на кадрах: {data.overlay_text}")
    if data.recognized_place:
        lines.append(f"Распознанное место: {data.recognized_place}")
    if not lines:
        lines.append("Источников нет — не выдумывай, верни null/другое/низкий confidence.")
    return "\n".join(lines)


def _raw_sources(data: AggregationInput) -> list[str]:
    sources: list[str] = []
    if data.caption:
        sources.append("caption")
    if data.transcript:
        sources.append("transcript")
    if data.overlay_text or data.recognized_place:
        sources.append("vision")
    return sources


def _build_payload(data: AggregationInput) -> dict:
    return {
        "model": GLM_TEXT_MODEL,
        "messages": [
            {"role": "system", "content": AGGREGATION_SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(data)},
        ],
        # glm-5 — reasoning-модель: помимо видимого content есть скрытые
        # reasoning-токены (см. `reasoning_content` в ответе), которые тоже
        # считаются в max_tokens. При низком лимите content обрезается до
        # пустой строки — проверено вживую. Берём запас.
        "max_tokens": 2048,
        "temperature": 0.2,
    }


def _parse_response(raw_content: str, raw_sources: list[str]) -> AggregatedLocation:
    text = raw_content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[len("json"):]
        text = text.strip()

    try:
        payload = json.loads(text)
        place_type = payload.get("place_type") or "другое"
        if place_type not in _ALLOWED_PLACE_TYPES:
            logger.warning("text-aggregation: неизвестный place_type=%r, беру 'другое'", place_type)
            place_type = "другое"
        return AggregatedLocation(
            name=payload.get("name") or None,
            region=payload.get("region") or None,
            place_type=place_type,
            confidence=float(payload.get("confidence") or 0.0),
            raw_sources=raw_sources,
        )
    except (json.JSONDecodeError, ValueError, AttributeError) as exc:
        raise AggregationError(f"Не удалось разобрать ответ GLM aggregation: {raw_content!r}") from exc


def _call_glm_sync(data: AggregationInput, api_key: str, base_url: str) -> AggregatedLocation:
    """Синхронный HTTP-вызов с retry (макс. 3 попытки, экспоненциальная
    задержка — конвенция AGENTS.md для внешних API)."""
    payload = _build_payload(data)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    raw_sources = _raw_sources(data)

    delay = RETRY_BASE_DELAY_SECONDS
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            raw_content = response.json()["choices"][0]["message"]["content"]
            return _parse_response(raw_content, raw_sources)
        except (requests.RequestException, KeyError, IndexError, AggregationError) as exc:
            # AggregationError здесь — пустой/невалидный content (см. reasoning-модели
            # ниже): подтверждено вживую, что это перемежающийся сбой, а не системная
            # ошибка формата, — тот же запрос на повторной попытке отрабатывает штатно.
            last_error = exc
            logger.warning(
                "text-aggregation: попытка %d/%d вызова GLM не удалась: %s",
                attempt, MAX_RETRIES, exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(delay)
                delay *= 2

    raise AggregationError(f"GLM text API недоступен после {MAX_RETRIES} попыток: {last_error}")


async def aggregate_location(data: AggregationInput, api_key: str, base_url: str) -> AggregatedLocation:
    """Агрегирует caption/transcript/vision-данные в структуру локации.

    `api_key`/`base_url` передаются явно вызывающим кодом — модуль остаётся
    чистой функцией и не читает переменные окружения сам (как и vision.py).

    Решение "нужно ли переспрашивать пользователя" (confidence <
    CONFIDENCE_THRESHOLD) принимает вызывающий код (user-clarification /
    orchestrator, следующая итерация) — здесь только считается и логируется.
    """
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _call_glm_sync, data, api_key, base_url)
    logger.info(
        "text-aggregation: name=%r region=%r place_type=%s confidence=%.2f needs_clarification=%s",
        result.name, result.region, result.place_type, result.confidence,
        result.confidence < CONFIDENCE_THRESHOLD,
    )
    return result
