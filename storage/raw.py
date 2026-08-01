"""Raw-слой: неизменяемый лог всех промежуточных данных пайплайна.

Только INSERT, никогда UPDATE. Источник для pipeline/distill.py.
Сбои записи не должны ломать основной пайплайн — вызывающий код
должен ловить StorageError и логировать, но не ререйзить.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Схема — отдельный файл от storage/db.py, но та же БД
_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_posts (
    id                 TEXT PRIMARY KEY,
    source_type        TEXT NOT NULL,
    source_url         TEXT,
    source_message_id  TEXT,
    caption_text       TEXT,
    transcript         TEXT,
    media_items_json   TEXT NOT NULL,
    pipeline_version   TEXT,
    received_at        TEXT NOT NULL,
    processed_at       TEXT,
    created_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_media_items (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_post_id              TEXT NOT NULL REFERENCES raw_posts(id),
    item_index               INTEGER NOT NULL,
    kind                     TEXT NOT NULL,
    ocr_text                 TEXT,
    vision_overlay_text      TEXT,
    vision_recognized_place  TEXT,
    vision_confidence        REAL,
    created_at               TEXT NOT NULL,
    UNIQUE (raw_post_id, item_index)
);

CREATE TABLE IF NOT EXISTS raw_frames (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_media_item_id    INTEGER NOT NULL REFERENCES raw_media_items(id),
    frame_index          INTEGER NOT NULL,
    frame_path           TEXT,
    ocr_text             TEXT,
    is_archived          INTEGER NOT NULL DEFAULT 0,
    created_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_locations (
    id                    TEXT PRIMARY KEY,
    raw_post_id           TEXT NOT NULL REFERENCES raw_posts(id),
    item_index            INTEGER,
    caption               TEXT,
    transcript            TEXT,
    overlay_text          TEXT,
    recognized_place      TEXT,
    aggregated_name       TEXT,
    aggregated_region     TEXT,
    aggregated_place_type TEXT,
    aggregated_confidence REAL,
    raw_sources_json      TEXT,
    created_at            TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_raw_locations_post ON raw_locations(raw_post_id);
CREATE INDEX IF NOT EXISTS idx_raw_media_items_post ON raw_media_items(raw_post_id);
"""


class StorageError(RuntimeError):
    pass


@dataclass
class RawPostRecord:
    id: str
    source_type: str
    media_items_json: str
    received_at: datetime
    source_url: str | None = None
    source_message_id: str | None = None
    caption_text: str | None = None
    transcript: str | None = None
    pipeline_version: str | None = None
    processed_at: datetime | None = None


@dataclass
class RawMediaItemRecord:
    raw_post_id: str
    item_index: int
    kind: str
    ocr_text: str | None = None
    vision_overlay_text: str | None = None
    vision_recognized_place: str | None = None
    vision_confidence: float | None = None


@dataclass
class RawFrameRecord:
    raw_media_item_id: int
    frame_index: int
    frame_path: str | None = None
    ocr_text: str | None = None
    is_archived: bool = False


@dataclass
class RawLocationRecord:
    id: str
    raw_post_id: str
    aggregated_place_type: str
    aggregated_confidence: float
    item_index: int | None = None
    caption: str | None = None
    transcript: str | None = None
    overlay_text: str | None = None
    recognized_place: str | None = None
    aggregated_name: str | None = None
    aggregated_region: str | None = None
    raw_sources_json: str | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(database_path: Path) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


def _save_raw_post_sync(database_path: Path, record: RawPostRecord) -> None:
    conn = _connect(database_path)
    try:
        conn.execute(
            """INSERT OR REPLACE INTO raw_posts
               (id, source_type, source_url, source_message_id, caption_text,
                transcript, media_items_json, pipeline_version,
                received_at, processed_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id, record.source_type, record.source_url,
                record.source_message_id, record.caption_text,
                record.transcript, record.media_items_json,
                record.pipeline_version,
                record.received_at.isoformat(),
                record.processed_at.isoformat() if record.processed_at else None,
                _now_iso(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


async def save_raw_post(database_path: str | Path, record: RawPostRecord) -> None:
    database_path = Path(database_path)
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _save_raw_post_sync, database_path, record)
    except sqlite3.Error as exc:
        raise StorageError(f"raw: не удалось сохранить пост {record.id}: {exc}") from exc


def _save_raw_media_item_sync(database_path: Path, record: RawMediaItemRecord) -> int:
    conn = _connect(database_path)
    try:
        cursor = conn.execute(
            """INSERT OR REPLACE INTO raw_media_items
               (raw_post_id, item_index, kind, ocr_text,
                vision_overlay_text, vision_recognized_place,
                vision_confidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.raw_post_id, record.item_index, record.kind,
                record.ocr_text, record.vision_overlay_text,
                record.vision_recognized_place, record.vision_confidence,
                _now_iso(),
            ),
        )
        conn.commit()
        return cursor.lastrowid or 0
    finally:
        conn.close()


async def save_raw_media_item(database_path: str | Path, record: RawMediaItemRecord) -> int:
    """Возвращает id вставленной строки raw_media_items."""
    database_path = Path(database_path)
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _save_raw_media_item_sync, database_path, record)
    except sqlite3.Error as exc:
        raise StorageError(f"raw: не сохранить media item: {exc}") from exc


def _save_raw_location_sync(database_path: Path, record: RawLocationRecord) -> None:
    conn = _connect(database_path)
    try:
        conn.execute(
            """INSERT OR REPLACE INTO raw_locations
               (id, raw_post_id, item_index, caption, transcript,
                overlay_text, recognized_place,
                aggregated_name, aggregated_region,
                aggregated_place_type, aggregated_confidence,
                raw_sources_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id, record.raw_post_id, record.item_index,
                record.caption, record.transcript,
                record.overlay_text, record.recognized_place,
                record.aggregated_name, record.aggregated_region,
                record.aggregated_place_type, record.aggregated_confidence,
                record.raw_sources_json, _now_iso(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


async def save_raw_location(database_path: str | Path, record: RawLocationRecord) -> None:
    database_path = Path(database_path)
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _save_raw_location_sync, database_path, record)
    except sqlite3.Error as exc:
        raise StorageError(f"raw: не удалось сохранить локацию {record.id}: {exc}") from exc


# --- Читающие функции для distill.py ---

def _row_to_location(row: tuple) -> RawLocationRecord:
    return RawLocationRecord(
        id=row[0], raw_post_id=row[1], item_index=row[2],
        caption=row[3], transcript=row[4],
        overlay_text=row[5], recognized_place=row[6],
        aggregated_name=row[7], aggregated_region=row[8],
        aggregated_place_type=row[9], aggregated_confidence=row[10],
        raw_sources_json=row[11],
    )


def _list_raw_locations_without_wiki_sync(database_path: Path) -> list[RawLocationRecord]:
    conn = _connect(database_path)
    try:
        rows = conn.execute(
            """SELECT rl.id, rl.raw_post_id, rl.item_index, rl.caption,
                      rl.transcript, rl.overlay_text, rl.recognized_place,
                      rl.aggregated_name, rl.aggregated_region,
                      rl.aggregated_place_type, rl.aggregated_confidence,
                      rl.raw_sources_json
               FROM raw_locations rl
               LEFT JOIN wiki_place_sources ws ON ws.raw_location_id = rl.id
               WHERE ws.raw_location_id IS NULL
               ORDER BY rl.created_at"""
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_location(r) for r in rows]


async def list_raw_locations_without_wiki(database_path: str | Path) -> list[RawLocationRecord]:
    """raw_locations, ещё не привязанные ни к одной wiki-статье."""
    database_path = Path(database_path)
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _list_raw_locations_without_wiki_sync, database_path)
    except sqlite3.Error as exc:
        raise StorageError(f"raw: не удалось прочитать локации: {exc}") from exc


def _list_all_raw_locations_sync(database_path: Path) -> list[RawLocationRecord]:
    conn = _connect(database_path)
    try:
        rows = conn.execute(
            """SELECT id, raw_post_id, item_index, caption,
                      transcript, overlay_text, recognized_place,
                      aggregated_name, aggregated_region,
                      aggregated_place_type, aggregated_confidence,
                      raw_sources_json
               FROM raw_locations ORDER BY created_at"""
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_location(r) for r in rows]


async def list_all_raw_locations(database_path: str | Path) -> list[RawLocationRecord]:
    database_path = Path(database_path)
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _list_all_raw_locations_sync, database_path)
    except sqlite3.Error as exc:
        raise StorageError(f"raw: не удалось прочитать локации: {exc}") from exc
