"""geocoding: поиск координат по названию места (SKILLS.md, скилл 7).

По умолчанию — Nominatim (OpenStreetMap): бесплатный сервис, не требует
API-ключа. Выбран, чтобы весь пайплайн, кроме GLM Coding Plan, оставался
бесплатным для личного использования (см. обсуждение в README). Публичный
инстанс Nominatim ограничивает частоту запросов (~1 в секунду) — при
последовательной обработке видео в оркестраторе этого достаточно; если
понадобится параллельная обработка, здесь стоит добавить троттлинг или
перейти на платный провайдер (Yandex/Google — тогда пригодится
GEOCODING_API_KEY из .env.example, сейчас не используется).

Если место не найдено — это не ошибка (возвращается None): вызывающий код
должен проставить needs_manual_location=true и всё равно сохранить запись
(SKILLS.md, скилл 7, "не блокировать сохранение").
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Nominatim usage policy требует осмысленный User-Agent с контактом проекта.
USER_AGENT = "dagestan-trip-bot/0.1 (personal travel-planning project)"

REQUEST_TIMEOUT_SECONDS = 15
MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 1.0


class GeocodingError(RuntimeError):
    """Geocoding API недоступен — не путать с «место не найдено» (это None)."""


@dataclass
class GeocodingResult:
    lat: float
    lng: float
    formatted_address: str


def _build_query(name: str, region: str | None) -> str:
    return f"{name}, {region}" if region else name


def _call_nominatim_sync(query: str) -> GeocodingResult | None:
    headers = {"User-Agent": USER_AGENT}
    params = {"q": query, "format": "jsonv2", "limit": 1}

    delay = RETRY_BASE_DELAY_SECONDS
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                NOMINATIM_URL, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            results = response.json()
            if not results:
                return None
            top = results[0]
            return GeocodingResult(
                lat=float(top["lat"]),
                lng=float(top["lon"]),
                formatted_address=top.get("display_name", query),
            )
        except (requests.RequestException, KeyError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "geocoding: попытка %d/%d запроса к Nominatim не удалась: %s",
                attempt, MAX_RETRIES, exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(delay)
                delay *= 2

    raise GeocodingError(f"Geocoding API недоступен после {MAX_RETRIES} попыток: {last_error}")


async def geocode_location(name: str, region: str | None = None) -> GeocodingResult | None:
    """Ищет координаты по названию (+ опционально региону).

    Возвращает None, если место не найдено — это штатный случай, не
    исключение (см. docstring модуля).
    """
    query = _build_query(name, region)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _call_nominatim_sync, query)
