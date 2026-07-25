"""transcription: локальный faster-whisper (SKILLS.md, скилл 3).

Вход: путь к WAV-файлу.
Выход: `{ transcript: str, language: str, confidence: float }`.

Если в аудио нет речи (только музыка/тишина) — это не ошибка пайплайна,
возвращается пустая строка (VAD-фильтр faster-whisper отсекает не-речевые
сегменты).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# "small"/"medium" — компромисс между качеством и скоростью на CPU (SKILLS.md).
WHISPER_MODEL_SIZE = "small"

_model: WhisperModel | None = None


class TranscriptionError(RuntimeError):
    """faster-whisper не смог обработать аудиофайл."""


@dataclass
class TranscriptionResult:
    transcript: str
    language: str | None
    confidence: float


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        logger.info("Загрузка модели faster-whisper (%s)...", WHISPER_MODEL_SIZE)
        _model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def _transcribe_sync(audio_path: Path) -> TranscriptionResult:
    model = _get_model()
    try:
        segments, info = model.transcribe(str(audio_path), vad_filter=True)
        transcript = "".join(segment.text for segment in segments).strip()
        return TranscriptionResult(
            transcript=transcript,
            language=info.language,
            # language_probability — уверенность распознавания языка, единственная
            # величина такого рода, которую отдаёт faster-whisper из коробки.
            confidence=info.language_probability,
        )
    except ValueError:
        # faster-whisper падает с ValueError("max() iterable argument is empty"),
        # если VAD отфильтровал вообще всё аудио как не-речь — тогда для
        # автоопределения языка не остаётся ни одного сегмента. Это не ошибка,
        # а ровно случай "речи нет" из SKILLS.md (скилл 3).
        logger.info("Речь в аудио не обнаружена (VAD не нашёл сегментов): %s", audio_path)
        return TranscriptionResult(transcript="", language=None, confidence=0.0)


async def transcribe_audio(audio_path: str | Path) -> TranscriptionResult:
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise TranscriptionError(f"Аудиофайл не найден: {audio_path}")

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, _transcribe_sync, audio_path)
    except Exception as exc:  # noqa: BLE001 — оборачиваем любую ошибку faster-whisper
        raise TranscriptionError(f"Не удалось транскрибировать {audio_path}: {exc}") from exc

    logger.info(
        "transcription: %d символов, язык=%s, confidence=%.2f",
        len(result.transcript),
        result.language,
        result.confidence,
    )
    return result
