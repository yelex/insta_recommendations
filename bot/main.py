from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher

from bot.config import load_settings
from bot.handlers import router
from pipeline.orchestrator import PipelineConfig
from storage.db import path_from_database_url

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    settings = load_settings()
    bot = Bot(token=settings.telegram_bot_token)
    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    pipeline_config = PipelineConfig(
        glm_api_key=settings.glm_api_key,
        glm_base_url=settings.glm_base_url,
        database_path=str(path_from_database_url(settings.database_url)),
    )

    logger.info(
        "Бот запускается, media_storage_dir=%s, database=%s",
        settings.media_storage_dir, pipeline_config.database_path,
    )
    await dispatcher.start_polling(
        bot, media_storage_dir=settings.media_storage_dir, pipeline_config=pipeline_config
    )


if __name__ == "__main__":
    asyncio.run(main())
