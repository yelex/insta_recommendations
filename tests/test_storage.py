"""Юнит-тесты для storage.db (SKILLS.md, скилл 8).

Локальная SQLite — не внешний API, поэтому тесты работают с настоящим
(временным) файлом БД, а не с моками.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from storage.db import (
    LocationRecord,
    StorageError,
    list_locations,
    path_from_database_url,
    save_location,
)


def test_path_from_database_url_parses_sqlite_scheme() -> None:
    assert path_from_database_url("sqlite:///storage/locations.db") == Path("storage/locations.db")


def test_path_from_database_url_rejects_other_schemes() -> None:
    with pytest.raises(StorageError):
        path_from_database_url("postgres://user:pass@host/db")


async def test_save_and_list_round_trip(tmp_path: Path) -> None:
    db_path = tmp_path / "locations.db"
    record = LocationRecord(
        id="loc-1",
        name="Сулакский каньон",
        region="Дагестан",
        place_type="природа",
        confidence=0.9,
        lat=43.2,
        lng=46.8,
        source_message_id="msg-1",
    )

    await save_location(db_path, record)
    locations = await list_locations(db_path)

    assert len(locations) == 1
    saved = locations[0]
    assert saved.id == "loc-1"
    assert saved.name == "Сулакский каньон"
    assert saved.lat == pytest.approx(43.2)
    assert saved.needs_manual_location is False
    assert isinstance(saved.created_at, datetime)


async def test_list_locations_filters_needs_manual_location(tmp_path: Path) -> None:
    db_path = tmp_path / "locations.db"
    await save_location(
        db_path,
        LocationRecord(id="geocoded", place_type="природа", confidence=0.9, needs_manual_location=False),
    )
    await save_location(
        db_path,
        LocationRecord(id="not-geocoded", place_type="другое", confidence=0.2, needs_manual_location=True),
    )

    all_locations = await list_locations(db_path)
    geocoded_only = await list_locations(db_path, only_geocoded=True)

    assert {loc.id for loc in all_locations} == {"geocoded", "not-geocoded"}
    assert {loc.id for loc in geocoded_only} == {"geocoded"}


async def test_save_location_creates_schema_on_first_use(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "locations.db"
    assert not db_path.exists()

    await save_location(
        db_path, LocationRecord(id="loc-1", place_type="еда", confidence=0.5)
    )

    assert db_path.exists()


async def test_duplicate_id_raises_storage_error(tmp_path: Path) -> None:
    db_path = tmp_path / "locations.db"
    record = LocationRecord(id="loc-1", place_type="еда", confidence=0.5)

    await save_location(db_path, record)
    with pytest.raises(StorageError):
        await save_location(db_path, record)


async def test_created_at_defaults_to_now_utc(tmp_path: Path) -> None:
    db_path = tmp_path / "locations.db"
    before = datetime.now(timezone.utc)

    await save_location(db_path, LocationRecord(id="loc-1", place_type="еда", confidence=0.5))

    after = datetime.now(timezone.utc)
    [saved] = await list_locations(db_path)
    assert before <= saved.created_at <= after
