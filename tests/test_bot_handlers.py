"""Юнит-тесты для bot.handlers._format_result_message.

Полный aiogram-хендлер (Message/Bot) не тестируется юнит-тестами — это
требует отдельной инфраструктуры мокания aiogram и не даёт много ценности
по сравнению с тестированием самой логики форматирования. Проверено вручную
через реальный чат с ботом (см. README).
"""

from __future__ import annotations

from bot.handlers import _format_result_message
from pipeline.orchestrator import ProcessingResult
from storage.db import LocationRecord


def test_needs_clarification_message_mentions_draft_and_confidence() -> None:
    result = ProcessingResult(
        location=LocationRecord(
            id="1", name=None, region=None, place_type="другое", confidence=0.2,
            needs_manual_location=True,
        ),
        needs_clarification=True,
    )

    text = _format_result_message(result)

    assert "не смог уверенно определить" in text
    assert "0.20" in text


def test_success_message_includes_coordinates() -> None:
    result = ProcessingResult(
        location=LocationRecord(
            id="1", name="Сулакский каньон", region="Дагестан", place_type="природа",
            confidence=0.9, lat=43.2, lng=46.8, needs_manual_location=False,
        ),
        needs_clarification=False,
    )

    text = _format_result_message(result)

    assert "Сулакский каньон" in text
    assert "Дагестан" in text
    assert "43.20000" in text and "46.80000" in text


def test_success_message_without_coordinates_says_not_found() -> None:
    result = ProcessingResult(
        location=LocationRecord(
            id="1", name="Неизвестное место", region="Дагестан", place_type="другое",
            confidence=0.9, lat=None, lng=None, needs_manual_location=True,
        ),
        needs_clarification=False,
    )

    text = _format_result_message(result)

    assert "не найдены" in text
