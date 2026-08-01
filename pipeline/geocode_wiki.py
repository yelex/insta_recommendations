"""Геокодинг wiki-мест через Nominatim (OpenStreetMap).
Лимит: 1 запрос/сек — для десятков мест ок.

Запуск:
    .venv/bin/python -m pipeline.geocode_wiki
"""

from __future__ import annotations

import asyncio
import logging
import time

import requests

from storage.wiki import list_wiki_places, get_wiki_place_by_name

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "insta-recommendations-bot/1.0"
TIMEOUT_SECONDS = 10
DELAY_SECONDS = 1.2  # уважение к лимиту Nominatim


def _geocode_nominatim(query: str) -> tuple[float, float] | None:
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": query, "format": "json", "limit": 1, "accept-language": "ru"},
            headers={"User-Agent": USER_AGENT},
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        results = resp.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception as exc:
        logger.warning("geocode: Nominatim для «%s»: %s", query, exc)
    return None


def _update_coords_sync(database_path: str, place_id: str, lat: float, lng: float) -> None:
    import sqlite3
    conn = sqlite3.connect(database_path)
    conn.execute(
        "UPDATE wiki_places SET lat = ?, lng = ?, updated_at = ? WHERE id = ?",
        (lat, lng, __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), place_id),
    )
    conn.commit()
    conn.close()


async def geocode_all_wiki(database_path: str) -> int:
    """Заполняет пропущенные координаты для всех wiki-мест."""
    places = await list_wiki_places(database_path)
    missing = [p for p in places if p.lat is None]

    if not missing:
        logger.info("geocode: все места уже имеют координаты")
        return 0

    logger.info("geocode: %d мест без координат", len(missing))
    updated = 0

    for p in missing:
        # Формируем поисковый запрос
        query = p.canonical_name
        if p.region:
            query += f", {p.region}"
        query += ", Дагестан, Россия"

        logger.info("geocode: ищу «%s»", query)
        coords = await asyncio.get_running_loop().run_in_executor(None, _geocode_nominatim, query)

        if coords:
            lat, lng = coords
            await asyncio.get_running_loop().run_in_executor(
                None, _update_coords_sync, database_path, p.id, lat, lng
            )
            logger.info("geocode: ✅ %s → %.5f, %.5f", p.canonical_name, lat, lng)
            updated += 1
        else:
            logger.warning("geocode: ❌ %s — не найдено", p.canonical_name)

        await asyncio.sleep(DELAY_SECONDS)

    logger.info("geocode: обновлено %d/%d", updated, len(missing))
    return updated


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    from bot.config import load_settings
    settings = load_settings()
    db_path = settings.database_url.replace("sqlite:///", "")
    asyncio.run(geocode_all_wiki(db_path))


if __name__ == "__main__":
    main()
