"""frame-extraction: ffmpeg-обёртка (SKILLS.md, скилл 2).

Вход: путь к видеофайлу.
Выход: список путей к кадрам (JPEG, ~1 кадр/сек, не больше MAX_FRAMES) +
путь к аудиодорожке (WAV, 16kHz mono — формат, ожидаемый faster-whisper).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

FRAME_RATE_FPS = 1
# Ограничение числа кадров на видео — чтобы не раздувать счёт за vision-вызовы
# (см. AGENTS.md, "не увеличивать частоту/объём LLM-вызовов без явного запроса").
MAX_FRAMES = 60


class FrameExtractionError(RuntimeError):
    """ffmpeg не смог извлечь кадры или аудио из видео."""


@dataclass
class ExtractedMedia:
    frames: list[Path]
    audio_path: Path | None


def _ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise FrameExtractionError(
            "ffmpeg не найден в PATH. Установите ffmpeg (см. README.md)."
        )


async def _run_ffmpeg(args: list[str]) -> None:
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise FrameExtractionError(
            f"ffmpeg завершился с ошибкой (код {process.returncode}): "
            f"{stderr.decode(errors='ignore')[-2000:]}"
        )


async def extract_frames_and_audio(video_path: str | Path, output_dir: str | Path) -> ExtractedMedia:
    """Извлекает кадры (1 fps, максимум MAX_FRAMES) и аудиодорожку из видео.

    Если у видео нет аудиодорожки — это не ошибка, `audio_path` будет None
    (аналогично правилу из SKILLS.md для transcription: отсутствие звука
    не считается ошибкой пайплайна).
    """
    _ensure_ffmpeg_available()

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FrameExtractionError(f"Видеофайл не найден: {video_path}")

    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_pattern = str(frames_dir / "frame_%04d.jpg")
    await _run_ffmpeg(
        [
            "-y",
            "-i", str(video_path),
            "-vf", f"fps={FRAME_RATE_FPS}",
            "-frames:v", str(MAX_FRAMES),
            frame_pattern,
        ]
    )
    frames = sorted(frames_dir.glob("frame_*.jpg"))

    audio_path = output_dir / "audio.wav"
    try:
        await _run_ffmpeg(
            [
                "-y",
                "-i", str(video_path),
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                str(audio_path),
            ]
        )
    except FrameExtractionError:
        logger.warning("Аудиодорожка не найдена или не извлечена: %s", video_path)
        audio_path = None

    logger.info(
        "frame-extraction: %d кадров, аудио=%s, источник=%s",
        len(frames),
        audio_path,
        video_path,
    )
    return ExtractedMedia(frames=frames, audio_path=audio_path)
