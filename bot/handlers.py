"""Telegram-обработчики: приём видео и заглушка для остальных типов входа.

Видео теперь прогоняется через весь пайплайн (`pipeline/orchestrator.py`):
extract → transcribe/vision → aggregate → geocode → store. Диалог
уточнения (SKILLS.md, скилл 6, user-clarification) пока не реализован —
при низкой уверенности бот честно сообщает об этом и сохраняет черновик,
не переспрашивая (это отдельная следующая итерация).
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

from pipeline.aggregate import AggregationError
from pipeline.frame_extraction import FrameExtractionError
from pipeline.geocode import GeocodingError
from pipeline.ingest import create_video_raw_item
from pipeline.orchestrator import PipelineConfig, ProcessingResult, process_video_item
from pipeline.transcribe import TranscriptionError
from pipeline.vision import VisionAnalysisError
from storage.db import StorageError

logger = logging.getLogger(__name__)

# Ошибки, которые могут прилететь из process_video_item — ловим их здесь,
# чтобы одно неудачное видео не роняло цикл обработки и пользователь
# получал внятный ответ (AGENTS.md: try/except с понятным логом на внешние
# вызовы; ValueError — из orchestrator.process_video_item на RawItem без файла).
PIPELINE_ERRORS = (
    FrameExtractionError,
    TranscriptionError,
    VisionAnalysisError,
    AggregationError,
    GeocodingError,
    StorageError,
    ValueError,
)

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


def _format_result_message(result: ProcessingResult) -> str:
    location = result.location

    if result.needs_clarification:
        return (
            "Сохранил, но не смог уверенно определить место 🤔\n"
            f"Черновик: {location.name or '—'}"
            + (f", {location.region}" if location.region else "")
            + f" (уверенность {location.confidence:.2f}).\n"
            "Уточнение вручную — в одной из следующих итераций."
        )

    coords = f"{location.lat:.5f}, {location.lng:.5f}" if location.lat is not None else "не найдены"
    region_part = f", {location.region}" if location.region else ""
    return (
        f"Готово ✅ {location.name}{region_part}\n"
        f"Тип: {location.place_type}\n"
        f"Координаты: {coords}"
    )


@router.message(F.video)
async def handle_video(message: Message, media_storage_dir: Path, pipeline_config: PipelineConfig) -> None:
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
        source_message_id=f"{message.chat.id}:{message.message_id}",
    )

    logger.info(
        "Видео принято: chat_id=%s user_id=%s forwarded=%s raw_item_id=%s",
        message.chat.id,
        message.from_user.id if message.from_user else None,
        message.forward_origin is not None,
        raw_item.id,
    )

    await message.answer("Видео получено, начинаю обработку — это может занять минуту...")

    try:
        result = await process_video_item(raw_item, pipeline_config)
    except PIPELINE_ERRORS:
        logger.exception("Ошибка обработки видео raw_item_id=%s", raw_item.id)
        await message.answer(
            "Видео сохранено, но обработать не получилось (сбой при распознавании места). "
            "Файл остался на диске, можно будет обработать позже."
        )
        return

    await message.answer(_format_result_message(result))


@router.message(F.photo | F.text)
async def handle_unsupported(message: Message) -> None:
    logger.info("Пока не обрабатываемый тип сообщения: chat_id=%s", message.chat.id)
    await message.answer(
        "Пока умею принимать только видео. Ссылки и скриншоты — в следующих итерациях."
    )
