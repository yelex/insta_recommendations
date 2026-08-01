"""Wiki-слой: LLM-дистиллированные статьи о местах.

Иерархия: parent_place_id (Дагестан → Махачкала → кофейня).
Кросс-ссылки: wiki_place_links (many-to-many).
Цены: prices_json (SearXNG + LLM).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

WIKI_NAMESPACE = uuid.UUID("a3f2c1b0-1234-5678-9abc-def012345678")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS wiki_places (
    id                       TEXT PRIMARY KEY,
    normalized_key           TEXT NOT NULL,
    canonical_name           TEXT NOT NULL,
    region                   TEXT,
    place_type               TEXT,
    lat                      REAL,
    lng                      REAL,
    description              TEXT,
    tags_json                TEXT,
    best_photo_paths_json    TEXT,
    prices_json              TEXT,
    parent_place_id          TEXT REFERENCES wiki_places(id),
    source_location_ids_json TEXT NOT NULL,
    post_count               INTEGER NOT NULL DEFAULT 0,
    distill_prompt_version   TEXT NOT NULL,
    distilled_at             TEXT NOT NULL,
    created_at               TEXT NOT NULL,
    updated_at               TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS wiki_place_sources (
    wiki_place_id    TEXT NOT NULL REFERENCES wiki_places(id),
    raw_location_id  TEXT NOT NULL,
    PRIMARY KEY (wiki_place_id, raw_location_id)
);

CREATE TABLE IF NOT EXISTS wiki_place_links (
    source_place_id   TEXT NOT NULL REFERENCES wiki_places(id),
    target_place_id   TEXT NOT NULL REFERENCES wiki_places(id),
    link_type         TEXT NOT NULL DEFAULT 'related',
    PRIMARY KEY (source_place_id, target_place_id)
);

CREATE INDEX IF NOT EXISTS idx_wiki_places_key ON wiki_places(normalized_key);
CREATE INDEX IF NOT EXISTS idx_wiki_places_parent ON wiki_places(parent_place_id);
CREATE INDEX IF NOT EXISTS idx_wiki_place_links_source ON wiki_place_links(source_place_id);
CREATE INDEX IF NOT EXISTS idx_wiki_place_links_target ON wiki_place_links(target_place_id);

-- Триггеры: автосинхронизация координат wiki_places ← locations
CREATE TRIGGER IF NOT EXISTS sync_wiki_coords_on_update
AFTER UPDATE OF lat, lng ON locations
FOR EACH ROW
WHEN NEW.lat IS NOT NULL AND NEW.lng IS NOT NULL
BEGIN
  UPDATE wiki_places
  SET lat = NEW.lat,
      lng = NEW.lng,
      updated_at = datetime('now')
  WHERE id IN (
    SELECT wiki_place_id FROM wiki_place_sources WHERE raw_location_id = NEW.id
  );
END;

CREATE TRIGGER IF NOT EXISTS sync_wiki_coords_on_insert
AFTER INSERT ON locations
FOR EACH ROW
WHEN NEW.lat IS NOT NULL AND NEW.lng IS NOT NULL
BEGIN
  UPDATE wiki_places
  SET lat = NEW.lat,
      lng = NEW.lng,
      updated_at = datetime('now')
  WHERE id IN (
    SELECT wiki_place_id FROM wiki_place_sources WHERE raw_location_id = NEW.id
  );
END;
"""


class StorageError(RuntimeError):
    pass


@dataclass
class WikiPlaceRecord:
    id: str
    normalized_key: str
    canonical_name: str
    source_location_ids_json: str
    distill_prompt_version: str
    region: str | None = None
    place_type: str | None = None
    lat: float | None = None
    lng: float | None = None
    description: str | None = None
    tags_json: str | None = None
    best_photo_paths_json: str | None = None
    prices_json: str | None = None
    parent_place_id: str | None = None
    post_count: int = 0
    distilled_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


def make_wiki_place_id(normalized_key: str) -> str:
    return str(uuid.uuid5(WIKI_NAMESPACE, normalized_key))


def normalize_key(name: str, region: str | None = None) -> str:
    parts = [name.strip().lower()]
    if region:
        parts.append(region.strip().lower())
    return "|".join(parts)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(database_path: Path) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


def _row_to_record(row: tuple) -> WikiPlaceRecord:
    return WikiPlaceRecord(
        id=row[0], normalized_key=row[1], canonical_name=row[2],
        region=row[3], place_type=row[4], lat=row[5], lng=row[6],
        description=row[7], tags_json=row[8], best_photo_paths_json=row[9],
        prices_json=row[10], parent_place_id=row[11],
        source_location_ids_json=row[12], post_count=row[13],
        distill_prompt_version=row[14], distilled_at=row[15],
        created_at=datetime.fromisoformat(row[16]) if row[16] else None,
        updated_at=datetime.fromisoformat(row[17]) if row[17] else None,
    )

_SELECT_ALL = (
    "SELECT id, normalized_key, canonical_name, region, place_type, "
    "lat, lng, description, tags_json, best_photo_paths_json, "
    "prices_json, parent_place_id, source_location_ids_json, "
    "post_count, distill_prompt_version, distilled_at, created_at, updated_at "
    "FROM wiki_places"
)


def _find_candidates_sync(
    database_path: Path, normalized_key: str,
    lat: float | None, lng: float | None, radius_km: float = 5.0,
) -> list[WikiPlaceRecord]:
    conn = _connect(database_path)
    try:
        rows = conn.execute(
            f"{_SELECT_ALL} WHERE normalized_key = ?", (normalized_key,)
        ).fetchall()
        if lat is not None and lng is not None:
            deg = radius_km / 111.0
            rows += conn.execute(
                f"""{_SELECT_ALL}
                   WHERE lat IS NOT NULL AND lng IS NOT NULL
                     AND lat BETWEEN ? AND ?
                     AND lng BETWEEN ? AND ?
                     AND normalized_key != ?""",
                (lat - deg, lat + deg, lng - deg, lng + deg, normalized_key),
            ).fetchall()
    finally:
        conn.close()
    seen: set[str] = set()
    result: list[WikiPlaceRecord] = []
    for row in rows:
        r = _row_to_record(row)
        if r.id not in seen:
            seen.add(r.id)
            result.append(r)
    return result


async def find_candidates(
    database_path: str | Path, normalized_key: str,
    lat: float | None = None, lng: float | None = None,
    radius_km: float = 5.0,
) -> list[WikiPlaceRecord]:
    database_path = Path(database_path)
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            None, _find_candidates_sync, database_path, normalized_key, lat, lng, radius_km
        )
    except sqlite3.Error as exc:
        raise StorageError(f"wiki: find_candidates: {exc}") from exc


def _upsert_sync(database_path: Path, record: WikiPlaceRecord, source_location_ids: list[str]) -> None:
    now = _now_iso()
    conn = _connect(database_path)
    try:
        conn.execute(
            """INSERT INTO wiki_places
               (id, normalized_key, canonical_name, region, place_type,
                lat, lng, description, tags_json, best_photo_paths_json,
                prices_json, parent_place_id,
                source_location_ids_json, post_count,
                distill_prompt_version, distilled_at,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                normalized_key=excluded.normalized_key,
                canonical_name=excluded.canonical_name,
                region=excluded.region,
                place_type=excluded.place_type,
                lat=excluded.lat,
                lng=excluded.lng,
                description=excluded.description,
                tags_json=excluded.tags_json,
                best_photo_paths_json=excluded.best_photo_paths_json,
                prices_json=COALESCE(excluded.prices_json, wiki_places.prices_json),
                parent_place_id=COALESCE(excluded.parent_place_id, wiki_places.parent_place_id),
                source_location_ids_json=excluded.source_location_ids_json,
                post_count=excluded.post_count,
                distill_prompt_version=excluded.distill_prompt_version,
                distilled_at=excluded.distilled_at,
                updated_at=excluded.updated_at""",
            (
                record.id, record.normalized_key, record.canonical_name,
                record.region, record.place_type, record.lat, record.lng,
                record.description, record.tags_json, record.best_photo_paths_json,
                record.prices_json, record.parent_place_id,
                record.source_location_ids_json, record.post_count,
                record.distill_prompt_version,
                record.distilled_at.isoformat() if record.distilled_at else now,
                now, now,
            ),
        )
        conn.execute(
            "DELETE FROM wiki_place_sources WHERE wiki_place_id = ?", (record.id,)
        )
        for loc_id in source_location_ids:
            conn.execute(
                "INSERT OR IGNORE INTO wiki_place_sources (wiki_place_id, raw_location_id) VALUES (?, ?)",
                (record.id, loc_id),
            )
        conn.commit()
    finally:
        conn.close()


async def upsert_wiki_place(
    database_path: str | Path, record: WikiPlaceRecord, source_location_ids: list[str]
) -> None:
    database_path = Path(database_path)
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _upsert_sync, database_path, record, source_location_ids)
    except sqlite3.Error as exc:
        raise StorageError(f"wiki: upsert: {exc}") from exc


# --- Связи между местами ---

def _set_links_sync(database_path: Path, place_id: str, links: list[tuple[str, str]]) -> None:
    """links = [(target_place_id, link_type), ...]"""
    conn = _connect(database_path)
    try:
        conn.execute("DELETE FROM wiki_place_links WHERE source_place_id = ?", (place_id,))
        for target_id, link_type in links:
            conn.execute(
                "INSERT OR IGNORE INTO wiki_place_links (source_place_id, target_place_id, link_type) VALUES (?, ?, ?)",
                (place_id, target_id, link_type),
            )
        conn.commit()
    finally:
        conn.close()


async def set_place_links(database_path: str | Path, place_id: str, links: list[tuple[str, str]]) -> None:
    database_path = Path(database_path)
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _set_links_sync, database_path, place_id, links)
    except sqlite3.Error as exc:
        raise StorageError(f"wiki: set_links: {exc}") from exc


def _get_links_sync(database_path: Path, place_id: str) -> list[tuple[str, str, str]]:
    """Возвращает [(target_id, target_name, link_type), ...]"""
    conn = _connect(database_path)
    try:
        rows = conn.execute(
            """SELECT l.target_place_id, p.canonical_name, l.link_type
               FROM wiki_place_links l
               JOIN wiki_places p ON p.id = l.target_place_id
               WHERE l.source_place_id = ?
               UNION
               SELECT l.source_place_id, p.canonical_name, l.link_type
               FROM wiki_place_links l
               JOIN wiki_places p ON p.id = l.source_place_id
               WHERE l.target_place_id = ?""",
            (place_id, place_id),
        ).fetchall()
    finally:
        conn.close()
    return [(r[0], r[1], r[2]) for r in rows]


async def get_place_links(database_path: str | Path, place_id: str) -> list[tuple[str, str, str]]:
    database_path = Path(database_path)
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _get_links_sync, database_path, place_id)
    except sqlite3.Error as exc:
        raise StorageError(f"wiki: get_links: {exc}") from exc


# --- Чтение ---

def _list_sync(database_path: Path, parent_id: str | None = None) -> list[WikiPlaceRecord]:
    conn = _connect(database_path)
    try:
        if parent_id is not None:
            rows = conn.execute(
                f"{_SELECT_ALL} WHERE parent_place_id = ? ORDER BY canonical_name", (parent_id,)
            ).fetchall()
        elif parent_id == "__root__":
            rows = conn.execute(
                f"{_SELECT_ALL} WHERE parent_place_id IS NULL ORDER BY canonical_name"
            ).fetchall()
        else:
            rows = conn.execute(
                f"{_SELECT_ALL} ORDER BY updated_at DESC"
            ).fetchall()
    finally:
        conn.close()
    return [_row_to_record(r) for r in rows]


async def list_wiki_places(
    database_path: str | Path, parent_id: str | None = None
) -> list[WikiPlaceRecord]:
    """parent_id=None — все; '__root__' — только верхний уровень; конкретный id — дети."""
    database_path = Path(database_path)
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _list_sync, database_path, parent_id)
    except sqlite3.Error as exc:
        raise StorageError(f"wiki: list: {exc}") from exc


def _get_sync(database_path: Path, place_id: str) -> WikiPlaceRecord | None:
    conn = _connect(database_path)
    try:
        row = conn.execute(f"{_SELECT_ALL} WHERE id = ?", (place_id,)).fetchone()
    finally:
        conn.close()
    return _row_to_record(row) if row else None


async def get_wiki_place(database_path: str | Path, place_id: str) -> WikiPlaceRecord | None:
    database_path = Path(database_path)
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _get_sync, database_path, place_id)
    except sqlite3.Error as exc:
        raise StorageError(f"wiki: get: {exc}") from exc


def _get_by_name_sync(database_path: Path, name: str) -> WikiPlaceRecord | None:
    conn = _connect(database_path)
    try:
        row = conn.execute(
            f"{_SELECT_ALL} WHERE canonical_name = ? COLLATE NOCASE OR normalized_key = ?",
            (name, normalize_key(name)),
        ).fetchone()
    finally:
        conn.close()
    return _row_to_record(row) if row else None


async def get_wiki_place_by_name(database_path: str | Path, name: str) -> WikiPlaceRecord | None:
    database_path = Path(database_path)
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _get_by_name_sync, database_path, name)
    except sqlite3.Error as exc:
        raise StorageError(f"wiki: get_by_name: {exc}") from exc


def _list_stale_sync(database_path: Path, current_prompt_version: str) -> list[WikiPlaceRecord]:
    conn = _connect(database_path)
    try:
        rows = conn.execute(
            f"{_SELECT_ALL} WHERE distill_prompt_version != ?",
            (current_prompt_version,),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_record(r) for r in rows]


async def list_stale_wiki_places(
    database_path: str | Path, current_prompt_version: str
) -> list[WikiPlaceRecord]:
    database_path = Path(database_path)
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            None, _list_stale_sync, database_path, current_prompt_version
        )
    except sqlite3.Error as exc:
        raise StorageError(f"wiki: stale: {exc}") from exc


def _rebuild_all_sync(database_path: Path) -> None:
    conn = _connect(database_path)
    try:
        conn.execute("DELETE FROM wiki_place_links")
        conn.execute("DELETE FROM wiki_place_sources")
        conn.execute("DELETE FROM wiki_places")
        conn.commit()
    finally:
        conn.close()


async def rebuild_all(database_path: str | Path) -> None:
    database_path = Path(database_path)
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _rebuild_all_sync, database_path)
    except sqlite3.Error as exc:
        raise StorageError(f"wiki: rebuild: {exc}") from exc


# --- Цены ---

def _update_prices_sync(database_path: Path, place_id: str, prices_json: str) -> None:
    conn = _connect(database_path)
    try:
        conn.execute(
            "UPDATE wiki_places SET prices_json = ?, updated_at = ? WHERE id = ?",
            (prices_json, _now_iso(), place_id),
        )
        conn.commit()
    finally:
        conn.close()


async def update_prices(database_path: str | Path, place_id: str, prices_json: str) -> None:
    database_path = Path(database_path)
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _update_prices_sync, database_path, place_id, prices_json)
    except sqlite3.Error as exc:
        raise StorageError(f"wiki: update_prices: {exc}") from exc
