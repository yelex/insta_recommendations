"""Типизированные модели данных пайплайна (см. SKILLS.md)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

RawItemType = Literal["video", "link", "screenshot"]


@dataclass
class RawItem:
    """Нормализованный вход из video-ingest (SKILLS.md, скилл 1)."""

    id: str
    type: RawItemType
    received_at: datetime
    file_path: Path | None = None
    url: str | None = None
    caption_text: str | None = None
