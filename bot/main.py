from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher

from bot.config import load_settings
from bot.handlers import router

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

    logger.info("Бот запускается, media_storage_dir=%s", settings.media_storage_dir)
    await dispatcher.start_polling(bot, media_storage_dir=settings.media_storage_dir)


if __name__ == "__main__":
    asyncio.run(main())
