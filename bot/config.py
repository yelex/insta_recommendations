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
    glm_api_key: str
    glm_base_url: str
    database_url: str


def load_settings() -> Settings:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN не задан. Скопируйте .env.example в .env и заполните значения."
        )
    glm_api_key = os.getenv("GLM_API_KEY")
    glm_base_url = os.getenv("GLM_API_BASE_URL")
    if not glm_api_key or not glm_base_url:
        raise RuntimeError(
            "GLM_API_KEY/GLM_API_BASE_URL не заданы. Скопируйте .env.example в .env и заполните значения."
        )
    media_storage_dir = Path(os.getenv("MEDIA_STORAGE_DIR", "media/videos"))
    database_url = os.getenv("DATABASE_URL", "sqlite:///storage/locations.db")
    return Settings(
        telegram_bot_token=token,
        media_storage_dir=media_storage_dir,
        glm_api_key=glm_api_key,
        glm_base_url=glm_base_url,
        database_url=database_url,
    )
