"""
Sprint 1 tests: ingestion loads a small synthetic export correctly and
is idempotent on re-run.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage.db import get_connection, init_schema
from src.ingestion.ingest_json import ingest_export

FIXTURE = [
    {
        "id": "1001",
        "name": "Example Game",
        "game_log": {
            "status": "completed",
            "rating": 9,
            "total_hours": 12,
            "total_minutes": 30,
            "is_backlog": False,
            "is_playing": False,
            "is_wishlist": False,
            "game_liked": True,
            "last_edited_at": 1700000000,
        },
        "playthroughs": {
            "5001": {
                "id": 5001,
                "rating": 9,
                "hours_played": 12,
                "mins_played": 30,
                "hours_finished": 12,
                "mins_finished": 0,
                "hours_mastered": None,
                "mins_mastered": None,
                "is_replay": False,
                "is_master": False,
                "start_date": "2024-01-01",
                "finish_date": "2024-01-05",
                "platform": "PC",
                "played_platform": "PC",
                "review": "Great game.",
                "review_spoilers": False,
                "created_at": "2024-01-05",
                "updated_at": "2024-01-05",
                "play_dates": [
                    {"id": 9001, "range_start_date": "2024-01-01", "range_end_date": "2024-01-02",
                     "start_date": "2024-01-01", "finish_date": None, "hours": 3, "minutes": 0, "note": ""},
                    {"id": 9002, "range_start_date": "2024-01-04", "range_end_date": "2024-01-05",
                     "start_date": None, "finish_date": "2024-01-05", "hours": 9, "minutes": 30, "note": ""},
                ],
            }
        },
    }
]


def _build_db(tmp_path):
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(FIXTURE))

    db_path = tmp_path / "glip.db"
    conn = get_connection(db_path)
    init_schema(conn)
    return conn, export_path


def test_ingest_counts(tmp_path):
    conn, export_path = _build_db(tmp_path)
    summary = ingest_export(conn, export_path)

    assert summary == {"games": 1, "playthroughs": 1, "sessions": 2, "skipped": 0}

    game = conn.execute("SELECT * FROM games WHERE igdb_id = '1001'").fetchone()
    assert game["title"] == "Example Game"

    ug = conn.execute("SELECT * FROM user_games WHERE igdb_id = '1001'").fetchone()
    assert ug["status"] == "completed"
    assert ug["total_hours"] == 12.5
    assert ug["is_liked"] == 1

    pt = conn.execute("SELECT * FROM playthroughs WHERE igdb_id = '1001'").fetchone()
    assert pt["hours_played"] == 12.5

    sessions = conn.execute("SELECT * FROM sessions WHERE playthrough_id = 5001").fetchall()
    assert len(sessions) == 2


def test_ingest_is_idempotent(tmp_path):
    conn, export_path = _build_db(tmp_path)
    ingest_export(conn, export_path)
    summary_second_run = ingest_export(conn, export_path)

    assert summary_second_run == {"games": 1, "playthroughs": 1, "sessions": 2, "skipped": 0}
    assert conn.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2
