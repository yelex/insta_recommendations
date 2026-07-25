"""video-ingest: нормализация входа в RawItem (SKILLS.md, скилл 1).

Скачивание файла из Telegram — ответственность слоя `bot` (там же, где
Telegram-специфичный код). Этот модуль работает только с уже сохранённым
локальным файлом и не знает про Telegram.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from pipeline.models import RawItem

logger = logging.getLogger(__name__)


def create_video_raw_item(
    item_id: str,
    file_path: Path,
    caption_text: str | None,
    source_message_id: str | None = None,
) -> RawItem:
    """Строит RawItem для уже сохранённого локально видеофайла."""
    raw_item = RawItem(
        id=item_id,
        type="video",
        received_at=datetime.now(timezone.utc),
        file_path=file_path,
        caption_text=caption_text,
        source_message_id=source_message_id,
    )
    logger.info(
        "video-ingest: получено видео id=%s file_path=%s caption=%r",
        raw_item.id,
        raw_item.file_path,
        raw_item.caption_text,
    )
    return raw_item
