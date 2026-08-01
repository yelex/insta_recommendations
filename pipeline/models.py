"""Типизированные модели данных пайплайна (см. SKILLS.md)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

RawItemType = Literal["video", "link", "screenshot", "carousel"]


@dataclass
class MediaItem:
    """Один медиа-элемент из карусели или одиночное видео/фото."""
    path: Path
    kind: Literal["video", "image"]


@dataclass
class RawItem:
    """Нормализованный вход из video-ingest (SKILLS.md, скилл 1)."""

    id: str
    type: RawItemType
    received_at: datetime
    file_path: Path | None = None
    url: str | None = None
    caption_text: str | None = None
    # "chat_id:message_id" исходного Telegram-сообщения — для storage.source_message_id
    # (SKILLS.md, скилл 8) и для будущего user-clarification (ответить в тот же чат).
    source_message_id: str | None = None
    # Для каруселей — список медиа-элементов (фото + видео).
    media_items: list[MediaItem] | None = None
