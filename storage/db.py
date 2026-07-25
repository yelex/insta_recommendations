"""storage: доступ к SQLite (SKILLS.md, скилл 8).

Финальный шаг для каждой обработанной локации. Схема — минимум из
SKILLS.md: id, name, region, place_type, lat, lng, confidence,
needs_manual_location, source_message_id, created_at.
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DATABASE_URL_PREFIX = "sqlite:///"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS locations (
    id TEXT PRIMARY KEY,
    name TEXT,
    region TEXT,
    place_type TEXT NOT NULL,
    lat REAL,
    lng REAL,
    confidence REAL NOT NULL,
    needs_manual_location INTEGER NOT NULL DEFAULT 0,
    source_message_id TEXT,
    created_at TEXT NOT NULL
);
"""


class StorageError(RuntimeError):
    """Ошибка доступа к БД."""


@dataclass
class LocationRecord:
    id: str
    place_type: str
    confidence: float
    name: str | None = None
    region: str | None = None
    lat: float | None = None
    lng: float | None = None
    needs_manual_location: bool = False
    source_message_id: str | None = None
    created_at: datetime | None = None


def path_from_database_url(database_url: str) -> Path:
    """Извлекает путь к файлу из DATABASE_URL вида sqlite:///storage/locations.db."""
    if not database_url.startswith(DATABASE_URL_PREFIX):
        raise StorageError(
            f"Ожидался DATABASE_URL вида '{DATABASE_URL_PREFIX}...', получено: {database_url!r}"
        )
    return Path(database_url[len(DATABASE_URL_PREFIX):])


def _connect(database_path: Path) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database_path)
    conn.execute(_SCHEMA)
    return conn


def _save_sync(database_path: Path, record: LocationRecord) -> None:
    created_at = (record.created_at or datetime.now(timezone.utc)).isoformat()
    conn = _connect(database_path)
    try:
        conn.execute(
            """
            INSERT INTO locations
                (id, name, region, place_type, lat, lng, confidence,
                 needs_manual_location, source_message_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.name,
                record.region,
                record.place_type,
                record.lat,
                record.lng,
                record.confidence,
                int(record.needs_manual_location),
                record.source_message_id,
                created_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()


async def save_location(database_path: str | Path, record: LocationRecord) -> None:
    database_path = Path(database_path)
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _save_sync, database_path, record)
    except sqlite3.Error as exc:
        raise StorageError(f"Не удалось сохранить локацию {record.id}: {exc}") from exc


def _row_to_record(row: tuple) -> LocationRecord:
    (
        id_, name, region, place_type, lat, lng, confidence,
        needs_manual_location, source_message_id, created_at,
    ) = row
    return LocationRecord(
        id=id_,
        name=name,
        region=region,
        place_type=place_type,
        lat=lat,
        lng=lng,
        confidence=confidence,
        needs_manual_location=bool(needs_manual_location),
        source_message_id=source_message_id,
        created_at=datetime.fromisoformat(created_at),
    )


def _list_sync(database_path: Path, only_geocoded: bool) -> list[LocationRecord]:
    conn = _connect(database_path)
    try:
        query = (
            "SELECT id, name, region, place_type, lat, lng, confidence, "
            "needs_manual_location, source_message_id, created_at FROM locations"
        )
        if only_geocoded:
            query += " WHERE needs_manual_location = 0"
        query += " ORDER BY created_at"
        rows = conn.execute(query).fetchall()
    finally:
        conn.close()
    return [_row_to_record(row) for row in rows]


async def list_locations(database_path: str | Path, only_geocoded: bool = False) -> list[LocationRecord]:
    """only_geocoded=True — как того требует route-planner (SKILLS.md, скилл 9):
    только локации с needs_manual_location = false."""
    database_path = Path(database_path)
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _list_sync, database_path, only_geocoded)
    except sqlite3.Error as exc:
        raise StorageError(f"Не удалось прочитать локации: {exc}") from exc
