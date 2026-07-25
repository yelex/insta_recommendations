"""Юнит-тесты для pipeline.frame_extraction (SKILLS.md, скилл 2).

Тестовое видео генерируется на лету через ffmpeg (testsrc + синус-тон),
чтобы не хранить бинарный файл в репозитории.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from pipeline import frame_extraction
from pipeline.frame_extraction import FrameExtractionError, extract_frames_and_audio


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    video_path = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-c:v", "libx264", "-c:a", "aac", "-shortest",
            str(video_path),
        ],
        check=True,
        capture_output=True,
    )
    return video_path


@pytest.fixture
def silent_video(tmp_path: Path) -> Path:
    video_path = tmp_path / "silent.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
            "-c:v", "libx264",
            str(video_path),
        ],
        check=True,
        capture_output=True,
    )
    return video_path


async def test_extracts_frames_and_audio(sample_video: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "output"

    result = await extract_frames_and_audio(sample_video, output_dir)

    assert len(result.frames) > 0
    assert all(frame.exists() and frame.suffix == ".jpg" for frame in result.frames)
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert result.audio_path.stat().st_size > 0


async def test_respects_max_frames_cap(sample_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(frame_extraction, "MAX_FRAMES", 2)

    result = await extract_frames_and_audio(sample_video, tmp_path / "output")

    assert len(result.frames) <= 2


async def test_video_without_audio_returns_none_audio_path(silent_video: Path, tmp_path: Path) -> None:
    result = await extract_frames_and_audio(silent_video, tmp_path / "output")

    assert len(result.frames) > 0
    assert result.audio_path is None


async def test_missing_video_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FrameExtractionError):
        await extract_frames_and_audio(tmp_path / "does_not_exist.mp4", tmp_path / "output")
