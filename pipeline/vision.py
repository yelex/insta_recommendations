"""vision-analysis: GLM-4.6V через GLM Coding Plan (SKILLS.md, скилл 4).

Вызывается прямой OpenAI-совместимый эндпоинт /chat/completions (см.
AGENTS.md, раздел "какой API-ключ использовать" — намеренное решение
владельца проекта использовать здесь Coding Plan ключ, а не pay-as-you-go).

Официально "рекомендуемый" для Coding Plan способ — Vision MCP Server, но
это MCP-инструмент для чата IDE-агента (Claude Code/Cline), а не HTTP API
для вызова из бэкенд-кода, поэтому здесь используется прямой вызов эндпоинта.

Кадры батчатся в минимально возможное число вызовов (см. SKILLS.md,
"ограничить число кадров, отправляемых за один вызов") — не один кадр на
один запрос.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

GLM_VISION_MODEL = "glm-4.6v"

# Не технический лимит модели (контекст 128K), а защита от слишком тяжёлых
# payload'ов при MAX_FRAMES=60 кадров на видео.
VISION_MAX_IMAGES_PER_BATCH = 20

REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 1.0

VISION_PROMPT = (
    "Посмотри на эти кадры из видео о путешествии по Дагестану.\n"
    "1. Есть ли на кадрах текстовые наклейки/подписи с названием места? Если "
    "да — выпиши точный текст.\n"
    "2. Узнаёшь ли ты конкретную достопримечательность (каньон, крепость, "
    "водопад, аул, кафе и т.д.)? Если да — назови её.\n"
    "Если ничего не удаётся определить — так и скажи, не выдумывай.\n\n"
    "Ответь строго в формате JSON без пояснений и без markdown-обрамления:\n"
    '{"overlay_text": "строка (пусто, если нет текста)", '
    '"recognized_place": "строка или null", "vision_confidence": 0.0}'
)


class VisionAnalysisError(RuntimeError):
    """GLM vision API недоступен или вернул нераспознаваемый ответ."""


@dataclass
class VisionAnalysisResult:
    overlay_text: str
    recognized_place: str | None
    vision_confidence: float


def _image_to_data_url(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _build_payload(image_paths: list[Path]) -> dict:
    content: list[dict] = [{"type": "text", "text": VISION_PROMPT}]
    for path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(path)}})
    return {
        "model": GLM_VISION_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1024,
        "temperature": 0.2,
    }


def _parse_response(raw_content: str) -> VisionAnalysisResult:
    text = raw_content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[len("json"):]
        text = text.strip()

    try:
        data = json.loads(text)
        return VisionAnalysisResult(
            overlay_text=data.get("overlay_text") or "",
            recognized_place=data.get("recognized_place") or None,
            vision_confidence=float(data.get("vision_confidence") or 0.0),
        )
    except (json.JSONDecodeError, ValueError, AttributeError) as exc:
        raise VisionAnalysisError(f"Не удалось разобрать ответ GLM vision: {raw_content!r}") from exc


def _call_glm_batch_sync(image_paths: list[Path], api_key: str, base_url: str) -> VisionAnalysisResult:
    """Синхронный HTTP-вызов с retry (макс. 3 попытки, экспоненциальная
    задержка — конвенция AGENTS.md для внешних API)."""
    payload = _build_payload(image_paths)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

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
            return _parse_response(raw_content)
        except (requests.RequestException, KeyError, IndexError) as exc:
            last_error = exc
            logger.warning(
                "vision-analysis: попытка %d/%d вызова GLM не удалась: %s",
                attempt, MAX_RETRIES, exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(delay)
                delay *= 2

    raise VisionAnalysisError(f"GLM vision API недоступен после {MAX_RETRIES} попыток: {last_error}")


def _merge_batches(results: list[VisionAnalysisResult]) -> VisionAnalysisResult:
    if len(results) == 1:
        return results[0]
    overlay_text = "\n".join(r.overlay_text for r in results if r.overlay_text)
    best = max(results, key=lambda r: r.vision_confidence)
    return VisionAnalysisResult(
        overlay_text=overlay_text,
        recognized_place=best.recognized_place,
        vision_confidence=best.vision_confidence,
    )


async def analyze_frames(image_paths: list[Path], api_key: str, base_url: str) -> VisionAnalysisResult:
    """Анализирует набор кадров (или один скриншот) через GLM-4.6V.

    `api_key`/`base_url` передаются явно вызывающим кодом (см. bot/config.py)
    — модуль остаётся чистой функцией и не читает переменные окружения сам.
    """
    if not image_paths:
        return VisionAnalysisResult(overlay_text="", recognized_place=None, vision_confidence=0.0)

    batches = [
        image_paths[i : i + VISION_MAX_IMAGES_PER_BATCH]
        for i in range(0, len(image_paths), VISION_MAX_IMAGES_PER_BATCH)
    ]

    loop = asyncio.get_running_loop()
    results = [
        await loop.run_in_executor(None, _call_glm_batch_sync, batch, api_key, base_url)
        for batch in batches
    ]

    merged = _merge_batches(results)
    logger.info(
        "vision-analysis: %d кадров, %d batch-вызов(а), recognized_place=%r, confidence=%.2f",
        len(image_paths), len(batches), merged.recognized_place, merged.vision_confidence,
    )
    return merged
