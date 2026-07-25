"""Юнит-тесты для pipeline.transcribe (SKILLS.md, скилл 3).

Реальная модель faster-whisper не грузится и не скачивается — по конвенции
AGENTS.md внешние/тяжёлые зависимости в юнит-тестах мокаются.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import pipeline.transcribe as transcribe_module
from pipeline.transcribe import TranscriptionError, transcribe_audio


class _FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeInfo:
    def __init__(self, language: str = "ru", language_probability: float = 0.95) -> None:
        self.language = language
        self.language_probability = language_probability


class _FakeWhisperModel:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def transcribe(self, path: str, vad_filter: bool = True):
        return [_FakeSegment(" привет "), _FakeSegment("мир")], _FakeInfo()


@pytest.fixture(autouse=True)
def reset_model_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(transcribe_module, "_model", None)
    yield
    monkeypatch.setattr(transcribe_module, "_model", None)


def _write_dummy_audio(tmp_path: Path) -> Path:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake wav content")
    return audio_path


async def test_transcribe_audio_returns_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcribe_module, "WhisperModel", _FakeWhisperModel)

    result = await transcribe_audio(_write_dummy_audio(tmp_path))

    assert result.transcript == "привет мир"
    assert result.language == "ru"
    assert result.confidence == pytest.approx(0.95)


async def test_no_speech_returns_empty_transcript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _SilentModel(_FakeWhisperModel):
        def transcribe(self, path: str, vad_filter: bool = True):
            return [], _FakeInfo(language="ru", language_probability=0.5)

    monkeypatch.setattr(transcribe_module, "WhisperModel", _SilentModel)

    result = await transcribe_audio(_write_dummy_audio(tmp_path))

    assert result.transcript == ""


async def test_missing_audio_file_raises(tmp_path: Path) -> None:
    with pytest.raises(TranscriptionError):
        await transcribe_audio(tmp_path / "missing.wav")


async def test_model_error_wrapped_in_transcription_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BrokenModel(_FakeWhisperModel):
        def transcribe(self, path: str, vad_filter: bool = True):
            raise RuntimeError("boom")

    monkeypatch.setattr(transcribe_module, "WhisperModel", _BrokenModel)

    with pytest.raises(TranscriptionError):
        await transcribe_audio(_write_dummy_audio(tmp_path))


async def test_model_loaded_once_across_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    load_count = 0

    class _CountingModel(_FakeWhisperModel):
        def __init__(self, *args, **kwargs) -> None:
            nonlocal load_count
            load_count += 1

    monkeypatch.setattr(transcribe_module, "WhisperModel", _CountingModel)
    audio_path = _write_dummy_audio(tmp_path)

    await transcribe_audio(audio_path)
    await transcribe_audio(audio_path)

    assert load_count == 1
