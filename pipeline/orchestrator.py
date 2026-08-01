"""Оркестратор: связывает скиллы в одну цепочку обработки видео."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from pipeline.aggregate import CONFIDENCE_THRESHOLD, AggregationInput, aggregate_location
from pipeline.frame_extraction import ExtractedMedia, extract_frames_and_audio
from pipeline.geocode import geocode_location
from pipeline.models import MediaItem, RawItem
from pipeline.transcribe import transcribe_audio
from pipeline.ocr import extract_text_from_frames
from pipeline.vision import VisionAnalysisResult, analyze_frames
from storage.db import LocationRecord, save_location
from storage.raw import (
    RawLocationRecord as RawLocRec,
    RawMediaItemRecord,
    RawPostRecord,
    StorageError as RawStorageError,
    save_raw_location,
    save_raw_media_item,
    save_raw_post,
)

logger = logging.getLogger(__name__)


async def _trigger_distill(config: PipelineConfig) -> None:
    """Fire-and-forget: запускает distill после сохранения raw_location.
    Ошибки ловим — не должны ронять пайплайн."""
    try:
        from pipeline.distill import run_distill_job, sync_all_coordinates, deduplicate_wiki_places
        await run_distill_job(
            config.database_path, config.glm_api_key, config.glm_base_url,
        )
        await sync_all_coordinates(config.database_path)
        await deduplicate_wiki_places(config.database_path, config.glm_api_key, config.glm_base_url)
    except Exception as exc:
        logger.warning("orchestrator: distill-хук не отработал: %s", exc)


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


@dataclass
class CarouselResult:
    """Результат карусели — несколько локаций."""
    locations: list[LocationRecord]
    needs_clarification: bool


def _cleanup_temp_files(extracted: ExtractedMedia) -> None:
    for frame in extracted.frames:
        frame.unlink(missing_ok=True)
    if extracted.audio_path is not None:
        extracted.audio_path.unlink(missing_ok=True)


async def _save_raw_post_safe(config: PipelineConfig, raw_item: RawItem) -> None:
    """Сохраняет raw_post. Сбой не должен ронять пайплайн."""
    media_json = "[]"
    if raw_item.media_items:
        media_json = json.dumps(
            [{"path": str(m.path), "kind": m.kind} for m in raw_item.media_items]
        )

    record = RawPostRecord(
        id=raw_item.id,
        source_type=raw_item.type,
        source_url=raw_item.url,
        source_message_id=raw_item.source_message_id,
        caption_text=raw_item.caption_text,
        media_items_json=media_json,
        received_at=raw_item.received_at,
    )
    try:
        await save_raw_post(config.database_path, record)
    except RawStorageError as exc:
        logger.warning("orchestrator: raw_post не сохранён (%s): %s", raw_item.id, exc)


async def _save_raw_location_safe(
    config: PipelineConfig,
    loc_id: str,
    raw_post_id: str,
    item_index: int | None,
    agg_input: AggregationInput,
    aggregated,
) -> None:
    """Сохраняет raw_location. Сбой не роняет пайплайн."""
    record = RawLocRec(
        id=loc_id,
        raw_post_id=raw_post_id,
        item_index=item_index,
        caption=agg_input.caption,
        transcript=agg_input.transcript,
        overlay_text=agg_input.overlay_text,
        recognized_place=agg_input.recognized_place,
        aggregated_name=aggregated.name,
        aggregated_region=aggregated.region,
        aggregated_place_type=aggregated.place_type,
        aggregated_confidence=aggregated.confidence,
        raw_sources_json=json.dumps(aggregated.raw_sources),
    )
    try:
        await save_raw_location(config.database_path, record)
    except RawStorageError as exc:
        logger.warning("orchestrator: raw_location не сохранён (%s): %s", loc_id, exc)


async def process_video_item(
    raw_item: RawItem, config: PipelineConfig
) -> ProcessingResult | CarouselResult:
    if raw_item.file_path is None and not raw_item.media_items:
        raise ValueError("orchestrator: RawItem без file_path и media_items")

    # Raw-запись поста — в самом начале, пока все данные на месте
    await _save_raw_post_safe(config, raw_item)

    if raw_item.type == "carousel" and raw_item.media_items:
        return await _process_carousel(raw_item, config)

    if raw_item.file_path is None:
        raise ValueError("orchestrator: RawItem без file_path — поддерживается только видео")

    return await _process_single_video(raw_item, config)


async def _run_ocr(frames: list[Path]) -> str:
    """Запускает OCR в потоке (Tesseract синхронный)."""
    import asyncio
    return await asyncio.to_thread(extract_text_from_frames, frames)


async def _process_carousel(raw_item: RawItem, config: PipelineConfig) -> CarouselResult:
    """Обработка карусели: каждый элемент анализируется отдельно.
    Видео → 10 кадров → OCR + vision → aggregate → место.
    Фото → OCR + vision → aggregate → место.
    """
    all_transcripts: list[str] = []
    extracted_media: list[ExtractedMedia] = []
    # Для каждого элемента — свой набор кадров
    item_frames: list[list[Path]] = []
    item_ocr: list[str] = []  # OCR-текст для каждого элемента

    for item in raw_item.media_items or []:
        if item.kind == "image":
            item_frames.append([item.path])
            ocr_text = await _run_ocr([item.path])
            item_ocr.append(ocr_text)
        elif item.kind == "video":
            output_dir = Path(config.work_dir) / f"{raw_item.id}_{item.path.stem}"
            extracted = await extract_frames_and_audio(item.path, output_dir)
            item_frames.append(extracted.frames)
            extracted_media.append(extracted)

            ocr_text = await _run_ocr(extracted.frames)
            item_ocr.append(ocr_text)

            if extracted.audio_path is not None:
                transcription = await transcribe_audio(extracted.audio_path)
                if transcription.transcript:
                    all_transcripts.append(transcription.transcript)

    combined_transcript = "\n".join(all_transcripts) if all_transcripts else None

    # Vision — каждый элемент отдельно
    vision_results: list[VisionAnalysisResult] = []
    for frames in item_frames:
        if not frames:
            vision_results.append(VisionAnalysisResult())
            continue
        result = await analyze_frames(
            frames, api_key=config.glm_api_key, base_url=config.glm_base_url,
        )
        vision_results.append(result)

    # Объединяем vision + OCR для каждого элемента
    # Если OCR нашёл текст — используем его как overlay_text (приоритет над vision)
    combined_places: list[VisionAnalysisResult] = []
    for vplace, ocr_text in zip(vision_results, item_ocr):
        if ocr_text:
            # OCR нашёл подпись — это приоритетный источник
            combined = VisionAnalysisResult(
                overlay_text=ocr_text,
                recognized_place=vplace.recognized_place,
                vision_confidence=vplace.vision_confidence,
            )
        else:
            combined = vplace
        combined_places.append(combined)

    # Дедупликация до aggregation — по raw overlay_text/recognized_place.
    # Убираем полностью пустые элементы и точные дубликаты сырого текста.
    seen_raw: set[str] = set()
    unique_places: list[VisionAnalysisResult] = []
    for vplace in combined_places:
        key_text = (vplace.overlay_text or "").strip().lower()[:50]
        key_place = (vplace.recognized_place or "").strip().lower()
        key = key_text or key_place
        if not key or key == "null":
            continue  # пропускаем пустые
        if key not in seen_raw:
            seen_raw.add(key)
            unique_places.append(vplace)

    # Aggregation + дедупликация по результату.
    # LLM-агрегатор может превратить разные overlay-тексты в одно и то же
    # имя (например, 3 слайда про еду в Дербенте → все «Дербент, Дагестан»).
    # Дедуплицируем по (name, region) после агрегации.
    locations: list[LocationRecord] = []
    seen_agg: set[str] = set()
    any_needs_clarification = False

    for vplace in unique_places[:15]:
        aggregated = await aggregate_location(
            AggregationInput(
                caption=raw_item.caption_text,
                transcript=combined_transcript,
                overlay_text=vplace.overlay_text or None,
                recognized_place=vplace.recognized_place,
            ),
            api_key=config.glm_api_key,
            base_url=config.glm_base_url,
        )

        # Дедупликация по агрегированному имени + региону
        agg_key = (
            (aggregated.name or "").strip().lower() + "|" +
            (aggregated.region or "").strip().lower()
        )
        if agg_key in seen_agg:
            logger.info("orchestrator: carousel — дубликат после aggregation: %r, пропускаем", agg_key)
            continue
        seen_agg.add(agg_key)

        needs_clarification = aggregated.confidence < CONFIDENCE_THRESHOLD or (
            aggregated.name is None and aggregated.region is None
        )
        if needs_clarification:
            any_needs_clarification = True

        lat: float | None = None
        lng: float | None = None
        needs_manual_location = True
        if aggregated.name and not needs_clarification:
            geo = await geocode_location(
                aggregated.name, aggregated.region,
                llm_api_key=config.glm_api_key, llm_base_url=config.glm_base_url,
            )
            if geo is not None:
                lat, lng = geo.lat, geo.lng
                needs_manual_location = False

        idx = len(locations)
        record = LocationRecord(
            id=f"{raw_item.id}_{idx}",
            name=aggregated.name,
            region=aggregated.region,
            place_type=aggregated.place_type,
            confidence=aggregated.confidence,
            lat=lat,
            lng=lng,
            needs_manual_location=needs_manual_location,
            source_message_id=raw_item.source_message_id,
        )
        # Raw-запись локации — до save_location, но не блокирует при сбое
        await _save_raw_location_safe(
            config, record.id, raw_item.id, idx,
            AggregationInput(
                caption=raw_item.caption_text,
                transcript=combined_transcript,
                overlay_text=vplace.overlay_text or None,
                recognized_place=vplace.recognized_place,
            ),
            aggregated,
        )
        await save_location(config.database_path, record)
        locations.append(record)

    # Чистим временные файлы
    for extracted in extracted_media:
        _cleanup_temp_files(extracted)

    # Distill-хук: обновляем wiki после новых raw_locations
    await _trigger_distill(config)

    logger.info(
        "orchestrator: carousel raw_item=%s найдено мест: %d",
        raw_item.id, len(locations),
    )

    return CarouselResult(locations=locations, needs_clarification=any_needs_clarification)


async def _process_single_video(raw_item: RawItem, config: PipelineConfig) -> ProcessingResult:
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

    needs_clarification = aggregated.confidence < CONFIDENCE_THRESHOLD or (
        aggregated.name is None and aggregated.region is None
    )

    lat: float | None = None
    lng: float | None = None
    needs_manual_location = True
    if aggregated.name and not needs_clarification:
        geo = await geocode_location(
            aggregated.name, aggregated.region,
            llm_api_key=config.glm_api_key, llm_base_url=config.glm_base_url,
        )
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
    # Raw-запись локации — до save_location, но не блокирует при сбое
    await _save_raw_location_safe(
        config, record.id, raw_item.id, None,
        AggregationInput(
            caption=raw_item.caption_text,
            transcript=transcript,
            overlay_text=vision_result.overlay_text or None,
            recognized_place=vision_result.recognized_place,
        ),
        aggregated,
    )
    await save_location(config.database_path, record)
    _cleanup_temp_files(extracted)

    # Distill-хук: обновляем wiki после новой raw_location
    await _trigger_distill(config)

    logger.info(
        "orchestrator: raw_item=%s name=%r confidence=%.2f needs_clarification=%s "
        "needs_manual_location=%s",
        raw_item.id, record.name, record.confidence, needs_clarification, needs_manual_location,
    )

    return ProcessingResult(location=record, needs_clarification=needs_clarification)
