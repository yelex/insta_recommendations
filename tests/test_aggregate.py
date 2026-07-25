"""Юнит-тесты для pipeline.aggregate (SKILLS.md, скилл 5).

Реальный GLM API не вызывается — мокается `requests.post`, как и в
tests/test_vision.py.
"""

from __future__ import annotations

from typing import Any

import pytest
import requests

import pipeline.aggregate as aggregate_module
from pipeline.aggregate import (
    AggregatedLocation,
    AggregationError,
    AggregationInput,
    aggregate_location,
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


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aggregate_module.time, "sleep", lambda _seconds: None)


async def test_aggregate_happy_path_all_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    response_content = (
        '{"name": "Сулакский каньон", "region": "Дагестан", '
        '"place_type": "природа", "confidence": 0.9}'
    )
    monkeypatch.setattr(
        aggregate_module.requests, "post", lambda *a, **k: _content_response(response_content)
    )

    data = AggregationInput(
        caption="самый глубокий каньон Европы",
        transcript="это Сулакский каньон",
        overlay_text="Сулакский каньон",
        recognized_place="Сулакский каньон",
    )

    result = await aggregate_location(data, api_key="key", base_url="https://api.example/v4")

    assert result == AggregatedLocation(
        name="Сулакский каньон",
        region="Дагестан",
        place_type="природа",
        confidence=0.9,
        raw_sources=["caption", "transcript", "vision"],
    )


async def test_raw_sources_reflects_only_present_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    response_content = '{"name": null, "region": null, "place_type": "другое", "confidence": 0.1}'
    monkeypatch.setattr(
        aggregate_module.requests, "post", lambda *a, **k: _content_response(response_content)
    )

    data = AggregationInput(transcript="что-то невнятное")

    result = await aggregate_location(data, api_key="key", base_url="https://api.example/v4")

    assert result.raw_sources == ["transcript"]
    assert result.name is None
    assert result.confidence == pytest.approx(0.1)


async def test_parses_markdown_wrapped_json(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapped = (
        '```json\n{"name": "Дербент", "region": "Дагестан", '
        '"place_type": "достопримечательность", "confidence": 0.8}\n```'
    )
    monkeypatch.setattr(aggregate_module.requests, "post", lambda *a, **k: _content_response(wrapped))

    result = await aggregate_location(
        AggregationInput(caption="крепость"), api_key="key", base_url="https://api.example/v4"
    )

    assert result.name == "Дербент"
    assert result.place_type == "достопримечательность"


async def test_unknown_place_type_falls_back_to_other(monkeypatch: pytest.MonkeyPatch) -> None:
    response_content = '{"name": "X", "region": null, "place_type": "неизвестно", "confidence": 0.5}'
    monkeypatch.setattr(
        aggregate_module.requests, "post", lambda *a, **k: _content_response(response_content)
    )

    result = await aggregate_location(
        AggregationInput(caption="что-то"), api_key="key", base_url="https://api.example/v4"
    )

    assert result.place_type == "другое"


async def test_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"n": 0}

    def flaky_post(url, headers, json, timeout):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise requests.exceptions.ConnectionError("boom")
        return _content_response(
            '{"name": null, "region": null, "place_type": "другое", "confidence": 0.0}'
        )

    monkeypatch.setattr(aggregate_module.requests, "post", flaky_post)

    result = await aggregate_location(
        AggregationInput(caption="что-то"), api_key="key", base_url="https://api.example/v4"
    )

    assert attempts["n"] == 3
    assert result.confidence == 0.0


async def test_all_retries_exhausted_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_fail(*args, **kwargs):
        raise requests.exceptions.ConnectionError("boom")

    monkeypatch.setattr(aggregate_module.requests, "post", always_fail)

    with pytest.raises(AggregationError):
        await aggregate_location(
            AggregationInput(caption="что-то"), api_key="key", base_url="https://api.example/v4"
        )


async def test_malformed_json_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        aggregate_module.requests, "post", lambda *a, **k: _content_response("это не JSON")
    )

    with pytest.raises(AggregationError):
        await aggregate_location(
            AggregationInput(caption="что-то"), api_key="key", base_url="https://api.example/v4"
        )
