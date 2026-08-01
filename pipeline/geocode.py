"""geocoding: поиск координат по названию места (SKILLS.md, скилл 7).

Многоуровневая стратегия:
1. Yandex Geocoder — лучший для России (знает заведения, улицы, регионы).
   Требует API-ключ (GEOCODING_API_KEY). Бесплатно до 1000 запросов/сутки.
2. Nominatim (OSM) — бесплатно, без ключа. Хороши для гео-объектов.
   Пытаемся несколькими вариантами запроса. Фильтруем по bbox Дагестана.
3. LLM fallback — просим GLM назвать координаты. Последний рубеж.

Если место не найдено — возвращается None: вызывающий код проставляет
needs_manual_location=true и всё равно сохраняет запись.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
YANDEX_GEOCODE_URL = "https://geocode-maps.yandex.ru/1.x/"
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
    source: str  # "yandex" | "nominatim" | "llm"


# ---------------------------------------------------------------------------
# Yandex Geocoder
# ---------------------------------------------------------------------------

def _yandex_geocode_sync(query: str, api_key: str) -> GeocodingResult | None:
    """Запрос к Yandex Geocoder API.
    Яндекс лучше всех знает российские заведения и адреса.
    """
    params = {
        "apikey": api_key,
        "geocode": query,
        "format": "json",
        "results": 1,
        "lang": "ru_RU",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                YANDEX_GEOCODE_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            data = response.json()
            members = (
                data.get("response", {})
                .get("GeoObjectCollection", {})
                .get("featureMember", [])
            )
            if not members:
                return None
            geo = members[0]["GeoObject"]
            point = geo["Point"]["pos"]  # "lon lat"
            lon, lat = point.split()
            address = (
                geo.get("metaDataProperty", {})
                .get("GeocoderMetaData", {})
                .get("text", query)
            )
            return GeocodingResult(
                lat=float(lat),
                lng=float(lon),
                formatted_address=address,
                source="yandex",
            )
        except (requests.RequestException, KeyError, ValueError) as exc:
            logger.warning(
                "geocoding: попытка %d/%d Yandex не удалась: %s",
                attempt, MAX_RETRIES, exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)))
    return None


def _yandex_multi_query(name: str, region: str | None, api_key: str) -> GeocodingResult | None:
    """Пытается найти место в Яндексе несколькими вариантами запроса."""
    queries: list[str] = []
    if region:
        queries.append(f"{name}, {region}")
        city = region.split(",")[0].strip()
        if city != region:
            queries.append(f"{name}, {city}")
    queries.append(name)

    seen: set[str] = set()
    for q in queries:
        if q in seen:
            continue
        seen.add(q)
        result = _yandex_geocode_sync(q, api_key)
        if result is not None:
            logger.info("geocoding: Yandex нашёл %r запросом %r", name, q)
            return result
    return None


# ---------------------------------------------------------------------------
# Nominatim (OSM)
# ---------------------------------------------------------------------------

def _in_dagestan_bbox(lat: float, lng: float) -> bool:
    """Грубый bounding box Республики Дагестан."""
    return 40.8 <= lat <= 43.7 and 44.5 <= lng <= 48.7


def _nominatim_search(query: str, lang: str = "ru") -> GeocodingResult | None:
    """Один запрос к Nominatim, возвращает None если ничего не найдено.
    Ограничиваем поиск Россией (countrycodes=ru) и фильтруем по bbox Дагестана.
    """
    headers = {"User-Agent": USER_AGENT}
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": 3,
        "accept-language": lang,
        "countrycodes": "ru",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                NOMINATIM_URL, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            results = response.json()
            if not results:
                return None
            for top in results:
                lat = float(top["lat"])
                lng = float(top["lon"])
                if _in_dagestan_bbox(lat, lng):
                    return GeocodingResult(
                        lat=lat,
                        lng=lng,
                        formatted_address=top.get("display_name", query),
                        source="nominatim",
                    )
            return None
        except (requests.RequestException, KeyError, ValueError) as exc:
            logger.warning(
                "geocoding: попытка %d/%d Nominatim не удалась: %s",
                attempt, MAX_RETRIES, exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)))
    return None


def _nominatim_multi_query(name: str, region: str | None) -> GeocodingResult | None:
    """Пытается найти место несколькими вариантами запроса."""
    queries: list[str] = []
    if region:
        queries.append(f"{name}, {region}")
        city = region.split(",")[0].strip()
        if city != region:
            queries.append(f"{name}, {city}")
    queries.append(name)

    seen: set[str] = set()
    for q in queries:
        if q in seen:
            continue
        seen.add(q)
        result = _nominatim_search(q)
        if result is not None:
            logger.info("geocoding: Nominatim нашёл %r запросом %r", name, q)
            return result
        time.sleep(1.0)  # Nominatim rate limit

    return None


# ---------------------------------------------------------------------------
# LLM fallback (GLM)
# ---------------------------------------------------------------------------

_COORD_RE = re.compile(r"(-?\d{1,3}\.\d{3,8})\s*,\s*(-?\d{1,3}\.\d{3,8})")


def _llm_geocode_sync(
    name: str, region: str | None, api_key: str, base_url: str
) -> GeocodingResult | None:
    """Спрашивает LLM назвать координаты места. Последний fallback."""
    if not api_key or not base_url:
        return None

    context = f", {region}" if region else ""
    prompt = (
        f"Назови точные географические координаты (широту и долготу) "
        f"места: {name}{context}.\n"
        f"Ответь только двумя числами через запятую: lat, lng. "
        f"Если не знаешь — ответь: UNKNOWN"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "glm-5",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.1,
        "thinking": {"type": "disabled"},
    }

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()
        if text.upper().startswith("UNKNOWN"):
            return None
        match = _COORD_RE.search(text)
        if not match:
            logger.warning("geocoding: LLM вернул непонятный ответ: %s", text)
            return None
        lat = float(match.group(1))
        lng = float(match.group(2))
        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            logger.warning("geocoding: LLM вернул невалидные координаты: %s, %s", lat, lng)
            return None
        logger.info("geocoding: LLM нашёл %r → %.4f, %.4f", name, lat, lng)
        return GeocodingResult(lat=lat, lng=lng, formatted_address=f"{name}{context}", source="llm")
    except Exception as exc:
        logger.warning("geocoding: LLM fallback не удался: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def geocode_location(
    name: str,
    region: str | None = None,
    *,
    llm_api_key: str | None = None,
    llm_base_url: str | None = None,
) -> GeocodingResult | None:
    """Ищет координаты по названию (+ опционально региону).

    Стратегия: Yandex → Nominatim → LLM fallback.

    Возвращает None, если место не найдено — это штатный случай.
    """
    loop = asyncio.get_running_loop()

    # Уровень 1: Yandex Geocoder (если есть ключ)
    yandex_key = os.getenv("GEOCODING_API_KEY", "")
    if yandex_key:
        result = await loop.run_in_executor(None, _yandex_multi_query, name, region, yandex_key)
        if result is not None:
            return result

    # Уровень 2: Nominatim
    result = await loop.run_in_executor(None, _nominatim_multi_query, name, region)
    if result is not None:
        return result

    # Уровень 3: LLM fallback
    if llm_api_key and llm_base_url:
        result = await loop.run_in_executor(
            None, _llm_geocode_sync, name, region, llm_api_key, llm_base_url
        )
        if result is not None:
            return result

    logger.info("geocoding: место не найдено ни одним способом: %r, %r", name, region)
    return None
