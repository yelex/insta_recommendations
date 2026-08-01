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
import re
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
from pipeline.ingest_url import download_video_from_url
from pipeline.orchestrator import CarouselResult, PipelineConfig, ProcessingResult, process_video_item
from pipeline.transcribe import TranscriptionError
from pipeline.vision import VisionAnalysisError
from storage.db import StorageError
from storage.wiki import list_wiki_places, get_wiki_place

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

# Instagram Reels / Posts / Share URLs
INSTAGRAM_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:instagram\.com|instagr\.am)/(?:reel|reels|p|share)/[\w\-]+",
    re.IGNORECASE,
)


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


def _format_result_message(result: ProcessingResult | CarouselResult) -> str:
    if isinstance(result, CarouselResult):
        if not result.locations:
            return "Не удалось найти места в этом посте 🤔"
        lines: list[str] = []
        for i, loc in enumerate(result.locations, 1):
            coords = f"{loc.lat:.5f}, {loc.lng:.5f}" if loc.lat is not None else "координаты не найдены"
            region_part = f", {loc.region}" if loc.region else ""
            lines.append(f"{i}. {loc.name or '—'}{region_part} ({loc.place_type}) — {coords}")
        header = "Найдено мест: " + str(len(result.locations)) + " 🎯\n\n"
        return header + "\n".join(lines)

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


@router.message(F.text & ~F.text.startswith("/"))
async def handle_text(message: Message, media_storage_dir: Path, pipeline_config: PipelineConfig) -> None:
    """Ловит Instagram-ссылки и скачивает видео через yt-dlp."""
    text = (message.text or "").strip()
    match = INSTAGRAM_URL_RE.search(text)
    if not match:
        logger.info("Текст без Instagram-ссылки: chat_id=%s", message.chat.id)
        await message.answer(
            "Пришли ссылку на Instagram Reels/пост — я скачаю видео и найду локацию."
        )
        return

    url = match.group(0)
    item_id = uuid.uuid4().hex
    media_storage_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Instagram-ссылка: chat_id=%s url=%s raw_item_id=%s",
        message.chat.id, url, item_id,
    )
    await message.answer("Скачиваю видео по ссылке...")

    try:
        raw_item = await asyncio.to_thread(
            download_video_from_url,
            url=url,
            item_id=item_id,
            media_storage_dir=media_storage_dir,
            caption_text=text if text != url else None,
            source_message_id=f"{message.chat.id}:{message.message_id}",
        )
    except RuntimeError:
        logger.exception("Не удалось скачать видео по ссылке: %s", url)
        await message.answer(
            "Не получилось скачать видео по ссылке. Проверь, что ссылка корректная и не приватная."
        )
        return

    await message.answer("Видео скачано, начинаю обработку — это может занять минуту...")

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


@router.message(F.photo)
async def handle_unsupported(message: Message) -> None:
    logger.info("Пока не обрабатываемый тип сообщения: chat_id=%s", message.chat.id)
    await message.answer(
        "Пока умею принимать только видео и Instagram-ссылки. Скриншоты — в следующих итерациях."
    )


# --- Wiki команды ---

WIKI_PAGE_SIZE = 10


@router.message(Command("wiki"))
async def handle_wiki(message: Message, pipeline_config: PipelineConfig) -> None:
    """Список wiki-статей с пагинацией."""
    args = (message.text or "").split()
    page = 1
    if len(args) > 1:
        try:
            page = max(1, int(args[1]))
        except ValueError:
            pass

    places = await list_wiki_places(pipeline_config.database_path)
    if not places:
        await message.answer("Wiki пока пуста. Обработай несколько постов и запусти distill.")
        return

    total = len(places)
    total_pages = (total + WIKI_PAGE_SIZE - 1) // WIKI_PAGE_SIZE
    page = min(page, total_pages)
    offset = (page - 1) * WIKI_PAGE_SIZE
    page_items = places[offset:offset + WIKI_PAGE_SIZE]

    lines = [f"📚 Wiki мест Дагестана ({total} мест, стр. {page}/{total_pages})\n"]
    for i, p in enumerate(page_items, offset + 1):
        region_part = f", {p.region}" if p.region else ""
        count_part = f" [{p.post_count} постов]" if p.post_count > 1 else ""
        lines.append(f"{i}. {p.canonical_name}{region_part} — {p.place_type}{count_part}")

    lines.append(f"\n/wiki {page + 1} — следующая страница" if page < total_pages else "")
    lines.append("/place <название> — подробно о месте")
    await message.answer("\n".join(lines))


@router.message(Command("place"))
async def handle_place(message: Message, pipeline_config: PipelineConfig) -> None:
    """Подробная статья о месте по названию или id."""
    query = (message.text or "").removeprefix("/place").strip()
    if not query:
        await message.answer("Использование: /place <название места>")
        return

    places = await list_wiki_places(pipeline_config.database_path)

    # Сначала точное вхождение по id, потом по названию
    match = None
    for p in places:
        if p.id == query:
            match = p
            break
    if not match:
        q_lower = query.lower()
        for p in places:
            if q_lower in p.canonical_name.lower():
                match = p
                break

    if not match:
        await message.answer(f"Не нашёл место «{query}». /wiki — список всех мест.")
        return

    import json
    tags = json.loads(match.tags_json) if match.tags_json else []
    coords = f"{match.lat:.5f}, {match.lng:.5f}" if match.lat is not None else "—"

    lines = [
        f"📍 {match.canonical_name}",
        f"Регион: {match.region or '—'}",
        f"Тип: {match.place_type or '—'}",
        f"Координаты: {coords}",
        f"Постов: {match.post_count}",
    ]
    if tags:
        lines.append(f"Теги: {', '.join(tags)}")
    if match.description:
        lines.append(f"\n{match.description}")

    await message.answer("\n".join(lines))
