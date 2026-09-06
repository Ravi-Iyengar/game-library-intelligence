"""
Sprint 2 tests: parsing IGDB's response shape, and the enrichment loop
against a fake client, so this runs without network access.
"""
import datetime as dt
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage.db import get_connection, init_schema
from src.enrichment.enrich_games import parse_igdb_game, enrich_games

# A representative (trimmed) IGDB /games response for Baldur's Gate 3,
# shaped the way the real API returns expanded fields.
RAW_IGDB_GAME = {
    "id": 119171,
    "name": "Baldur's Gate III",
    "genres": [{"id": 12, "name": "Role-playing (RPG)"}],
    "themes": [{"id": 1, "name": "Fantasy"}],
    "keywords": [{"id": 5, "name": "party-based"}],
    "franchises": [{"id": 9, "name": "Baldur's Gate"}],
    "collections": [{"id": 3, "name": "Baldur's Gate Collection"}],
    "game_modes": [{"id": 1, "name": "Single player"}, {"id": 2, "name": "Multiplayer"}],
    "player_perspectives": [{"id": 4, "name": "Bird view / Isometric"}],
    "involved_companies": [
        {"company": {"name": "Larian Studios"}, "developer": True, "publisher": True},
        {"company": {"name": "Some Distributor"}, "developer": False, "publisher": True},
    ],
    "first_release_date": 1691625600,  # 2023-08-10
    "aggregated_rating": 96.5,
    "aggregated_rating_count": 42,
    "platforms": [{"id": 6, "name": "PC (Microsoft Windows)"}],
}


def test_parse_igdb_game_maps_fields_correctly():
    parsed = parse_igdb_game(RAW_IGDB_GAME)

    assert parsed["igdb_id"] == "119171"
    assert json.loads(parsed["genres"]) == ["Role-playing (RPG)"]
    assert json.loads(parsed["themes"]) == ["Fantasy"]
    assert json.loads(parsed["developers"]) == ["Larian Studios"]
    assert json.loads(parsed["publishers"]) == ["Larian Studios", "Some Distributor"]
    assert json.loads(parsed["game_modes"]) == ["Single player", "Multiplayer"]
    assert parsed["release_date"] == "2023-08-10"
    assert parsed["aggregated_rating"] == 96.5
    assert parsed["aggregated_rating_count"] == 42


def test_parse_igdb_game_handles_missing_optional_fields():
    minimal = {"id": 1, "name": "Untitled Prototype"}
    parsed = parse_igdb_game(minimal)

    assert parsed["igdb_id"] == "1"
    assert json.loads(parsed["genres"]) == []
    assert json.loads(parsed["developers"]) == []
    assert parsed["release_date"] is None
    assert parsed["aggregated_rating"] is None


def test_enrich_games_updates_db_and_reports_not_found(tmp_path):
    db_path = tmp_path / "glip.db"
    conn = get_connection(db_path)
    init_schema(conn)

    conn.execute("INSERT INTO games (igdb_id, title) VALUES ('119171', 'Baldur''s Gate III')")
    conn.execute("INSERT INTO games (igdb_id, title) VALUES ('999999', 'Delisted Game')")
    conn.commit()

    fake_client = MagicMock()
    # IGDB simply omits ids it has no match for — only one of the two comes back.
    fake_client.query.return_value = [RAW_IGDB_GAME]

    summary = enrich_games(conn, fake_client)

    assert summary == {"requested": 2, "matched": 1, "not_found": 1}

    row = conn.execute("SELECT * FROM games WHERE igdb_id = '119171'").fetchone()
    assert json.loads(row["genres"]) == ["Role-playing (RPG)"]
    assert row["enriched_at"] is not None

    not_found_row = conn.execute(
        "SELECT * FROM games WHERE igdb_id = '999999'"
    ).fetchone()
    # Stamped as attempted (so it isn't retried every run) but metadata stays NULL.
    assert not_found_row["enriched_at"] is not None
    assert not_found_row["genres"] is None

    # Second call: nothing left pending, so the API isn't hit again.
    fake_client.query.reset_mock()
    summary2 = enrich_games(conn, fake_client)
    assert summary2 == {"requested": 0, "matched": 0, "not_found": 0}
    fake_client.query.assert_not_called()
