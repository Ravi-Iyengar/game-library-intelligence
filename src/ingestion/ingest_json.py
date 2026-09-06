"""
Sprint 1 — Database + JSON ingestion.

Loads a Backloggd-style library export (a JSON array of games, each with
a `game_log` block and a `playthroughs` dict keyed by playthrough id,
each playthrough carrying a `play_dates` list) into the SQLite tables
defined in schema.sql.

Design rules this module follows (per the project spec / EDD):
  * igdb_id is taken verbatim from each entry's "id" field and used as
    the primary key everywhere. No title matching, ever.
  * This is a RAW load: no status re-interpretation (e.g. mapping
    "retired" -> "abandoned"), no derived features. Those belong to
    later sprints (analytics / feature engineering) that read from
    these tables, not to ingestion.
  * Re-running ingestion is safe: each table is loaded with
    INSERT OR REPLACE, so importing the same export twice is a no-op,
    and importing an updated export just refreshes the rows.

Usage (also see scripts/run_ingestion.py for a CLI entry point):

    from src.storage.db import get_connection, init_schema
    from src.ingestion.ingest_json import ingest_export

    conn = get_connection("data/processed/glip.db")
    init_schema(conn)
    summary = ingest_export(conn, "data/raw/export.json")
    print(summary)
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _hours(hours: Any, minutes: Any) -> float | None:
    """Combine an hours/minutes pair into a single float-hours value."""
    if hours is None and minutes is None:
        return None
    return (hours or 0) + (minutes or 0) / 60.0


def _bool_to_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(bool(value))


def load_export(path: str | Path) -> list[dict]:
    """Read and parse the raw JSON export. Raises if it isn't a list."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected the export to be a JSON array of game entries, "
            f"got {type(data).__name__} instead."
        )
    return data


def _upsert_game(conn: sqlite3.Connection, entry: dict) -> None:
    conn.execute(
        """
        INSERT INTO games (igdb_id, title)
        VALUES (:igdb_id, :title)
        ON CONFLICT(igdb_id) DO UPDATE SET title = excluded.title
        """,
        {"igdb_id": str(entry["id"]), "title": entry.get("name")},
    )


def _upsert_user_game(conn: sqlite3.Connection, entry: dict) -> None:
    gl = entry.get("game_log") or {}
    conn.execute(
        """
        INSERT OR REPLACE INTO user_games (
            igdb_id, status, rating, total_hours,
            is_backlog, is_playing, is_wishlist, is_liked, last_edited_at
        ) VALUES (
            :igdb_id, :status, :rating, :total_hours,
            :is_backlog, :is_playing, :is_wishlist, :is_liked, :last_edited_at
        )
        """,
        {
            "igdb_id": str(entry["id"]),
            "status": gl.get("status"),
            "rating": gl.get("rating"),
            "total_hours": _hours(gl.get("total_hours"), gl.get("total_minutes")),
            "is_backlog": _bool_to_int(gl.get("is_backlog")),
            "is_playing": _bool_to_int(gl.get("is_playing")),
            "is_wishlist": _bool_to_int(gl.get("is_wishlist")),
            "is_liked": _bool_to_int(gl.get("game_liked")),
            "last_edited_at": gl.get("last_edited_at"),
        },
    )


def _insert_playthrough(conn: sqlite3.Connection, igdb_id: str, pt: dict) -> int:
    playthrough_id = int(pt["id"])
    conn.execute(
        """
        INSERT OR REPLACE INTO playthroughs (
            playthrough_id, igdb_id, rating,
            hours_played, hours_finished, hours_mastered,
            is_replay, is_master, start_date, finish_date,
            platform, played_platform, review, review_spoilers,
            created_at, updated_at
        ) VALUES (
            :playthrough_id, :igdb_id, :rating,
            :hours_played, :hours_finished, :hours_mastered,
            :is_replay, :is_master, :start_date, :finish_date,
            :platform, :played_platform, :review, :review_spoilers,
            :created_at, :updated_at
        )
        """,
        {
            "playthrough_id": playthrough_id,
            "igdb_id": igdb_id,
            "rating": pt.get("rating"),
            "hours_played": _hours(pt.get("hours_played"), pt.get("mins_played")),
            "hours_finished": _hours(pt.get("hours_finished"), pt.get("mins_finished")),
            "hours_mastered": _hours(pt.get("hours_mastered"), pt.get("mins_mastered")),
            "is_replay": _bool_to_int(pt.get("is_replay")),
            "is_master": _bool_to_int(pt.get("is_master")),
            "start_date": pt.get("start_date"),
            "finish_date": pt.get("finish_date"),
            "platform": pt.get("platform"),
            "played_platform": pt.get("played_platform"),
            "review": pt.get("review"),
            "review_spoilers": _bool_to_int(pt.get("review_spoilers")),
            "created_at": pt.get("created_at"),
            "updated_at": pt.get("updated_at"),
        },
    )
    return playthrough_id


def _insert_sessions(conn: sqlite3.Connection, playthrough_id: int, play_dates: list[dict]) -> int:
    count = 0
    for pd in play_dates or []:
        conn.execute(
            """
            INSERT OR REPLACE INTO sessions (
                session_id, playthrough_id,
                range_start_date, range_end_date,
                session_start_date, session_finish_date,
                hours, minutes, note
            ) VALUES (
                :session_id, :playthrough_id,
                :range_start_date, :range_end_date,
                :session_start_date, :session_finish_date,
                :hours, :minutes, :note
            )
            """,
            {
                "session_id": int(pd["id"]),
                "playthrough_id": playthrough_id,
                "range_start_date": pd.get("range_start_date"),
                "range_end_date": pd.get("range_end_date"),
                "session_start_date": pd.get("start_date"),
                "session_finish_date": pd.get("finish_date"),
                "hours": pd.get("hours"),
                "minutes": pd.get("minutes"),
                "note": pd.get("note"),
            },
        )
        count += 1
    return count


def ingest_export(conn: sqlite3.Connection, export_path: str | Path) -> dict[str, int]:
    """
    Load one export file into the database. Returns row-count summary.
    Safe to call multiple times (upserts throughout).
    """
    entries = load_export(export_path)

    n_games = n_playthroughs = n_sessions = 0
    skipped: list[str] = []

    for entry in entries:
        if "id" not in entry:
            skipped.append(entry.get("name", "<unknown>"))
            continue

        igdb_id = str(entry["id"])
        _upsert_game(conn, entry)
        _upsert_user_game(conn, entry)
        n_games += 1

        for pt in (entry.get("playthroughs") or {}).values():
            playthrough_id = _insert_playthrough(conn, igdb_id, pt)
            n_playthroughs += 1
            n_sessions += _insert_sessions(conn, playthrough_id, pt.get("play_dates"))

    conn.commit()

    if skipped:
        logger.warning("Skipped %d entries with no 'id' field: %s", len(skipped), skipped)

    summary = {
        "games": n_games,
        "playthroughs": n_playthroughs,
        "sessions": n_sessions,
        "skipped": len(skipped),
    }
    logger.info("Ingestion complete: %s", summary)
    return summary
