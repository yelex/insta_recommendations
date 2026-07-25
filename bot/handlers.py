"""Telegram-обработчики: приём видео и заглушка для остальных типов входа.

Реальная обработка (frame-extraction, vision, transcribe, aggregate,
geocode) подключается в следующих итерациях через оркестратор в
`pipeline/orchestrator.py`. Здесь бот только сохраняет файл и логирует событие.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from aiogram import F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.types import Message

from pipeline.ingest import create_video_raw_item

logger = logging.getLogger(__name__)

router = Router(name="dagestan_trip_bot")

DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_RETRY_BASE_DELAY_SECONDS = 1.0


async def _download_with_retry(message: Message, file_id: str, destination: Path) -> None:
    """Скачивает файл из Telegram с retry (макс. 3 попытки, экспоненциальная
    задержка) — см. AGENTS.md, конвенция для внешних API-вызовов."""
    delay = DOWNLOAD_RETRY_BASE_DELAY_SECONDS
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            await message.bot.download(file_id, destination=destination)
            return
        except TelegramAPIError:
            logger.warning(
                "Попытка %d/%d скачать файл %s не удалась", attempt, DOWNLOAD_ATTEMPTS, file_id
            )
            if attempt == DOWNLOAD_ATTEMPTS:
                raise
            await asyncio.sleep(delay)
            delay *= 2


@router.message(Command("start"))
async def handle_start(message: Message) -> None:
    await message.answer(
        "Привет! Пересылай сюда видео из Instagram про Дагестан — я их сохраню.\n"
        "Пока умею только принимать и сохранять видео, распознавание мест "
        "и построение маршрута появятся в следующих итерациях."
    )


@router.message(F.video)
async def handle_video(message: Message, media_storage_dir: Path) -> None:
    video = message.video
    item_id = uuid.uuid4().hex
    media_storage_dir.mkdir(parents=True, exist_ok=True)
    destination = media_storage_dir / f"{item_id}.mp4"

    try:
        await _download_with_retry(message, video.file_id, destination)
    except TelegramAPIError:
        logger.exception("Не удалось скачать видео file_id=%s", video.file_id)
        await message.answer("Не получилось скачать видео, попробуй переслать ещё раз.")
        return

    raw_item = create_video_raw_item(
        item_id=item_id,
        file_path=destination,
        caption_text=message.caption,
    )

    logger.info(
        "Видео принято: chat_id=%s user_id=%s forwarded=%s raw_item_id=%s",
        message.chat.id,
        message.from_user.id if message.from_user else None,
        message.forward_origin is not None,
        raw_item.id,
    )

    await message.answer("Видео получено и сохранено ✅\nОбработка появится в следующих итерациях.")


@router.message(F.photo | F.text)
async def handle_unsupported(message: Message) -> None:
    logger.info("Пока не обрабатываемый тип сообщения: chat_id=%s", message.chat.id)
    await message.answer(
        "Пока умею принимать только видео. Ссылки и скриншоты — в следующих итерациях."
    )
