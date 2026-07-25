"""Юнит-тесты для pipeline.vision (SKILLS.md, скилл 4).

Реальный GLM API не вызывается — по конвенции AGENTS.md внешние API
в юнит-тестах мокаются (здесь мокается `requests.post`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import requests

import pipeline.vision as vision_module
from pipeline.vision import (
    VISION_MAX_IMAGES_PER_BATCH,
    VisionAnalysisError,
    VisionAnalysisResult,
    analyze_frames,
)


def _make_response(payload: dict[str, Any]) -> Any:
    class _FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return payload

    return _FakeResponse()


def _content_response(content: str) -> Any:
    return _make_response({"choices": [{"message": {"content": content}}]})


def _write_dummy_image(tmp_path: Path, name: str = "frame.jpg") -> Path:
    path = tmp_path / name
    path.write_bytes(b"fake-jpeg-bytes")
    return path


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vision_module.time, "sleep", lambda _seconds: None)


async def test_analyze_frames_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = _write_dummy_image(tmp_path)
    response_content = (
        '{"overlay_text": "Сулакский каньон", "recognized_place": "Сулакский каньон", '
        '"vision_confidence": 0.9}'
    )
    calls: list[tuple[str, dict]] = []

    def fake_post(url, headers, json, timeout):
        calls.append((url, json))
        return _content_response(response_content)

    monkeypatch.setattr(vision_module.requests, "post", fake_post)

    result = await analyze_frames([image], api_key="key", base_url="https://api.example/v4")

    assert result == VisionAnalysisResult(
        overlay_text="Сулакский каньон",
        recognized_place="Сулакский каньон",
        vision_confidence=0.9,
    )
    assert len(calls) == 1
    assert calls[0][0] == "https://api.example/v4/chat/completions"


async def test_analyze_frames_empty_input_skips_api_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_post(*args, **kwargs):
        raise AssertionError("не должен вызываться для пустого списка кадров")

    monkeypatch.setattr(vision_module.requests, "post", fail_post)

    result = await analyze_frames([], api_key="key", base_url="https://api.example/v4")

    assert result == VisionAnalysisResult(overlay_text="", recognized_place=None, vision_confidence=0.0)


async def test_parses_markdown_wrapped_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = _write_dummy_image(tmp_path)
    wrapped = '```json\n{"overlay_text": "", "recognized_place": null, "vision_confidence": 0.0}\n```'

    monkeypatch.setattr(vision_module.requests, "post", lambda *a, **k: _content_response(wrapped))

    result = await analyze_frames([image], api_key="key", base_url="https://api.example/v4")

    assert result == VisionAnalysisResult(overlay_text="", recognized_place=None, vision_confidence=0.0)


async def test_batches_frames_and_merges_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images = [
        _write_dummy_image(tmp_path, f"frame_{i}.jpg") for i in range(VISION_MAX_IMAGES_PER_BATCH + 5)
    ]
    responses = [
        '{"overlay_text": "Дербент", "recognized_place": null, "vision_confidence": 0.3}',
        '{"overlay_text": "крепость Нарын-Кала", "recognized_place": "Нарын-Кала", '
        '"vision_confidence": 0.85}',
    ]
    call_count = {"n": 0}

    def fake_post(url, headers, json, timeout):
        content = responses[call_count["n"]]
        call_count["n"] += 1
        return _content_response(content)

    monkeypatch.setattr(vision_module.requests, "post", fake_post)

    result = await analyze_frames(images, api_key="key", base_url="https://api.example/v4")

    assert call_count["n"] == 2
    assert result.recognized_place == "Нарын-Кала"
    assert result.vision_confidence == pytest.approx(0.85)
    assert "Дербент" in result.overlay_text
    assert "крепость Нарын-Кала" in result.overlay_text


async def test_retries_then_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = _write_dummy_image(tmp_path)
    attempts = {"n": 0}

    def flaky_post(url, headers, json, timeout):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise requests.exceptions.ConnectionError("boom")
        return _content_response('{"overlay_text": "", "recognized_place": null, "vision_confidence": 0.0}')

    monkeypatch.setattr(vision_module.requests, "post", flaky_post)

    result = await analyze_frames([image], api_key="key", base_url="https://api.example/v4")

    assert attempts["n"] == 3
    assert result.vision_confidence == 0.0


async def test_all_retries_exhausted_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = _write_dummy_image(tmp_path)

    def always_fail(*args, **kwargs):
        raise requests.exceptions.ConnectionError("boom")

    monkeypatch.setattr(vision_module.requests, "post", always_fail)

    with pytest.raises(VisionAnalysisError):
        await analyze_frames([image], api_key="key", base_url="https://api.example/v4")


async def test_malformed_json_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = _write_dummy_image(tmp_path)

    monkeypatch.setattr(vision_module.requests, "post", lambda *a, **k: _content_response("это не JSON"))

    with pytest.raises(VisionAnalysisError):
        await analyze_frames([image], api_key="key", base_url="https://api.example/v4")
