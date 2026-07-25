"""Оркестратор: связывает скиллы в одну цепочку обработки видео
(AGENTS.md, диаграмма архитектуры; SKILLS.md, "Общие правила для всех
скиллов" — каждый скилл чистая функция, вся связка живёт здесь).

extract → (transcribe, vision) → aggregate → geocode/needs_manual_location
→ store.

Пока НЕ подключён к bot/handlers.py — вызывается явно (см. тесты). Решение
"нужно ли переспросить пользователя" (`needs_clarification` в
ProcessingResult) здесь только вычисляется; сам диалог с пользователем
(SKILLS.md, скилл 6, user-clarification) — Telegram-специфичная логика,
будущая работа в слое `bot`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pipeline.aggregate import CONFIDENCE_THRESHOLD, AggregationInput, aggregate_location
from pipeline.frame_extraction import ExtractedMedia, extract_frames_and_audio
from pipeline.geocode import geocode_location
from pipeline.models import RawItem
from pipeline.transcribe import transcribe_audio
from pipeline.vision import analyze_frames
from storage.db import LocationRecord, save_location

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    glm_api_key: str
    glm_base_url: str
    database_path: str
    work_dir: str = "media/work"


@dataclass
class ProcessingResult:
    location: LocationRecord
    needs_clarification: bool


def _cleanup_temp_files(extracted: ExtractedMedia) -> None:
    """AGENTS.md: временные файлы (кадры, аудио) удаляются после обработки —
    не копить на диске."""
    for frame in extracted.frames:
        frame.unlink(missing_ok=True)
    if extracted.audio_path is not None:
        extracted.audio_path.unlink(missing_ok=True)


async def process_video_item(raw_item: RawItem, config: PipelineConfig) -> ProcessingResult:
    if raw_item.file_path is None:
        raise ValueError("orchestrator: RawItem без file_path — поддерживается только видео")

    output_dir = Path(config.work_dir) / raw_item.id
    extracted = await extract_frames_and_audio(raw_item.file_path, output_dir)

    transcript: str | None = None
    if extracted.audio_path is not None:
        transcription = await transcribe_audio(extracted.audio_path)
        transcript = transcription.transcript or None

    vision_result = await analyze_frames(
        extracted.frames, api_key=config.glm_api_key, base_url=config.glm_base_url
    )

    aggregated = await aggregate_location(
        AggregationInput(
            caption=raw_item.caption_text,
            transcript=transcript,
            overlay_text=vision_result.overlay_text or None,
            recognized_place=vision_result.recognized_place,
        ),
        api_key=config.glm_api_key,
        base_url=config.glm_base_url,
    )

    # SKILLS.md, скилл 6: переспросить, если уверенность низкая ИЛИ name и
    # region оба не определены.
    needs_clarification = aggregated.confidence < CONFIDENCE_THRESHOLD or (
        aggregated.name is None and aggregated.region is None
    )

    lat: float | None = None
    lng: float | None = None
    needs_manual_location = True
    if aggregated.name and not needs_clarification:
        geo = await geocode_location(aggregated.name, aggregated.region)
        if geo is not None:
            lat, lng = geo.lat, geo.lng
            needs_manual_location = False

    record = LocationRecord(
        id=raw_item.id,
        name=aggregated.name,
        region=aggregated.region,
        place_type=aggregated.place_type,
        confidence=aggregated.confidence,
        lat=lat,
        lng=lng,
        needs_manual_location=needs_manual_location,
        source_message_id=raw_item.source_message_id,
    )
    await save_location(config.database_path, record)
    _cleanup_temp_files(extracted)

    logger.info(
        "orchestrator: raw_item=%s name=%r confidence=%.2f needs_clarification=%s "
        "needs_manual_location=%s",
        raw_item.id, record.name, record.confidence, needs_clarification, needs_manual_location,
    )

    return ProcessingResult(location=record, needs_clarification=needs_clarification)
