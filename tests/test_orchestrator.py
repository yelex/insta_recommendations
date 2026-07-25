"""Юнит-тесты для pipeline.orchestrator.

Все скиллы замоканы (у каждого уже есть свои юнит-тесты) — здесь
проверяется только логика связки: ветвление по confidence, вызов
geocoding, очистка временных файлов, форма итоговой записи.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

import pipeline.orchestrator as orchestrator_module
from pipeline.aggregate import AggregatedLocation
from pipeline.frame_extraction import ExtractedMedia
from pipeline.geocode import GeocodingResult
from pipeline.models import RawItem
from pipeline.orchestrator import PipelineConfig, process_video_item
from pipeline.transcribe import TranscriptionResult
from pipeline.vision import VisionAnalysisResult


def _make_raw_item(tmp_path: Path, with_file: bool = True) -> RawItem:
    file_path = None
    if with_file:
        file_path = tmp_path / "video.mp4"
        file_path.write_bytes(b"fake video")
    return RawItem(
        id="item-1",
        type="video",
        received_at=datetime.now(timezone.utc),
        file_path=file_path,
        caption_text="caption",
        source_message_id="123:456",
    )


def _make_extracted(tmp_path: Path, with_audio: bool = True) -> ExtractedMedia:
    frames = []
    for i in range(2):
        frame = tmp_path / f"frame_{i}.jpg"
        frame.write_bytes(b"fake jpeg")
        frames.append(frame)
    audio_path = None
    if with_audio:
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake wav")
    return ExtractedMedia(frames=frames, audio_path=audio_path)


def _patch_common(
    monkeypatch: pytest.MonkeyPatch,
    extracted: ExtractedMedia,
    aggregated: AggregatedLocation,
    geocode_result: GeocodingResult | None = None,
) -> dict[str, list]:
    calls: dict[str, list] = {"transcribe": [], "vision": [], "aggregate": [], "geocode": [], "save": []}

    async def fake_extract(video_path, output_dir):
        return extracted

    async def fake_transcribe(audio_path):
        calls["transcribe"].append(audio_path)
        return TranscriptionResult(transcript="транскрипт", language="ru", confidence=0.9)

    async def fake_analyze(image_paths, api_key, base_url):
        calls["vision"].append(image_paths)
        return VisionAnalysisResult(overlay_text="оверлей", recognized_place=None, vision_confidence=0.7)

    async def fake_aggregate(data, api_key, base_url):
        calls["aggregate"].append(data)
        return aggregated

    async def fake_geocode(name, region=None):
        calls["geocode"].append((name, region))
        return geocode_result

    async def fake_save(database_path, record):
        calls["save"].append(record)

    monkeypatch.setattr(orchestrator_module, "extract_frames_and_audio", fake_extract)
    monkeypatch.setattr(orchestrator_module, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(orchestrator_module, "analyze_frames", fake_analyze)
    monkeypatch.setattr(orchestrator_module, "aggregate_location", fake_aggregate)
    monkeypatch.setattr(orchestrator_module, "geocode_location", fake_geocode)
    monkeypatch.setattr(orchestrator_module, "save_location", fake_save)
    return calls


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        glm_api_key="key",
        glm_base_url="https://api.example/v4",
        database_path=str(tmp_path / "locations.db"),
        work_dir=str(tmp_path / "work"),
    )


async def test_high_confidence_geocode_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    extracted = _make_extracted(tmp_path)
    aggregated = AggregatedLocation(
        name="Сулакский каньон", region="Дагестан", place_type="природа", confidence=0.9
    )
    geo = GeocodingResult(lat=43.2, lng=46.8, formatted_address="Сулакский каньон, Дагестан")
    calls = _patch_common(monkeypatch, extracted, aggregated, geo)

    result = await process_video_item(_make_raw_item(tmp_path), _config(tmp_path))

    assert result.needs_clarification is False
    assert result.location.needs_manual_location is False
    assert result.location.lat == pytest.approx(43.2)
    assert result.location.lng == pytest.approx(46.8)
    assert result.location.source_message_id == "123:456"
    assert len(calls["geocode"]) == 1
    assert len(calls["save"]) == 1

    assert not any(frame.exists() for frame in extracted.frames)
    assert not extracted.audio_path.exists()


async def test_high_confidence_geocode_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    extracted = _make_extracted(tmp_path)
    aggregated = AggregatedLocation(
        name="Неизвестное место", region="Дагестан", place_type="другое", confidence=0.9
    )
    calls = _patch_common(monkeypatch, extracted, aggregated, geocode_result=None)

    result = await process_video_item(_make_raw_item(tmp_path), _config(tmp_path))

    assert result.needs_clarification is False
    assert result.location.needs_manual_location is True
    assert result.location.lat is None
    assert len(calls["geocode"]) == 1


async def test_low_confidence_skips_geocoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    extracted = _make_extracted(tmp_path)
    aggregated = AggregatedLocation(name="X", region="Y", place_type="другое", confidence=0.2)
    calls = _patch_common(monkeypatch, extracted, aggregated)

    result = await process_video_item(_make_raw_item(tmp_path), _config(tmp_path))

    assert result.needs_clarification is True
    assert result.location.needs_manual_location is True
    assert len(calls["geocode"]) == 0


async def test_missing_name_and_region_needs_clarification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extracted = _make_extracted(tmp_path)
    aggregated = AggregatedLocation(name=None, region=None, place_type="другое", confidence=0.9)
    calls = _patch_common(monkeypatch, extracted, aggregated)

    result = await process_video_item(_make_raw_item(tmp_path), _config(tmp_path))

    assert result.needs_clarification is True
    assert len(calls["geocode"]) == 0


async def test_no_audio_skips_transcription(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    extracted = _make_extracted(tmp_path, with_audio=False)
    aggregated = AggregatedLocation(name="X", region="Y", place_type="другое", confidence=0.9)
    calls = _patch_common(monkeypatch, extracted, aggregated, GeocodingResult(1.0, 2.0, "x"))

    await process_video_item(_make_raw_item(tmp_path), _config(tmp_path))

    assert len(calls["transcribe"]) == 0
    assert calls["aggregate"][0].transcript is None


async def test_missing_file_path_raises_without_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extracted = _make_extracted(tmp_path)
    aggregated = AggregatedLocation(name="X", region="Y", place_type="другое", confidence=0.9)
    calls = _patch_common(monkeypatch, extracted, aggregated)

    with pytest.raises(ValueError):
        await process_video_item(_make_raw_item(tmp_path, with_file=False), _config(tmp_path))

    assert calls["save"] == []
