"""Юнит-тесты для pipeline.geocode (SKILLS.md, скилл 7).

Реальный Nominatim не вызывается — мокается `requests.get`.
"""

from __future__ import annotations

from typing import Any

import pytest
import requests

import pipeline.geocode as geocode_module
from pipeline.geocode import GeocodingError, GeocodingResult, geocode_location


def _make_response(payload: Any) -> Any:
    class _FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> Any:
            return payload

    return _FakeResponse()


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(geocode_module.time, "sleep", lambda _seconds: None)


async def test_found_location_returns_result(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_get(url, headers, params, timeout):
        calls.append(params)
        return _make_response(
            [{"lat": "43.2105", "lon": "46.8340", "display_name": "Сулакский каньон, Дагестан"}]
        )

    monkeypatch.setattr(geocode_module.requests, "get", fake_get)

    result = await geocode_location("Сулакский каньон", region="Дагестан")

    assert result == GeocodingResult(lat=43.2105, lng=46.8340, formatted_address="Сулакский каньон, Дагестан")
    assert calls[0]["q"] == "Сулакский каньон, Дагестан"


async def test_not_found_returns_none_not_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(geocode_module.requests, "get", lambda *a, **k: _make_response([]))

    result = await geocode_location("НесуществующееМесто12345")

    assert result is None


async def test_query_without_region(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_get(url, headers, params, timeout):
        calls.append(params)
        return _make_response([])

    monkeypatch.setattr(geocode_module.requests, "get", fake_get)

    await geocode_location("Дербент")

    assert calls[0]["q"] == "Дербент"


async def test_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"n": 0}

    def flaky_get(url, headers, params, timeout):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise requests.exceptions.ConnectionError("boom")
        return _make_response([{"lat": "1.0", "lon": "2.0", "display_name": "Тест"}])

    monkeypatch.setattr(geocode_module.requests, "get", flaky_get)

    result = await geocode_location("Тест")

    assert attempts["n"] == 3
    assert result == GeocodingResult(lat=1.0, lng=2.0, formatted_address="Тест")


async def test_all_retries_exhausted_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_fail(*args, **kwargs):
        raise requests.exceptions.ConnectionError("boom")

    monkeypatch.setattr(geocode_module.requests, "get", always_fail)

    with pytest.raises(GeocodingError):
        await geocode_location("Тест")


async def test_malformed_result_raises_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        geocode_module.requests, "get", lambda *a, **k: _make_response([{"unexpected": "shape"}])
    )

    with pytest.raises(GeocodingError):
        await geocode_location("Тест")
