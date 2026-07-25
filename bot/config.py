from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    telegram_bot_token: str
    media_storage_dir: Path


def load_settings() -> Settings:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN не задан. Скопируйте .env.example в .env и заполните значения."
        )
    media_storage_dir = Path(os.getenv("MEDIA_STORAGE_DIR", "media/videos"))
    return Settings(telegram_bot_token=token, media_storage_dir=media_storage_dir)
