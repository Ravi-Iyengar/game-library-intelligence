"""
Loads the raw data every analytics function needs, in plain Python
structures (no pandas/polars dependency for Sprint 4 — these are small
enough that stdlib is plenty, and it keeps the analysis functions easy
to unit test without a DataFrame fixture).
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from typing import TypedDict

from src.feature_engineering.completion import signal_category
from src.utils.ratings import effective_rating
from src.utils.tags import parse_tag_list

TAG_CATEGORIES = ("genres", "themes", "developers", "publishers", "franchises")


class GameRecord(TypedDict):
    igdb_id: str
    title: str
    rating: float | None
    total_hours: float | None
    status: str | None
    signal_category: str
    tags: dict[str, list[str]]
    playthrough_count: int
    replay_count: int


class PlaythroughYearRecord(TypedDict):
    igdb_id: str
    year: int
    hours: float


def load_games(conn: sqlite3.Connection) -> list[GameRecord]:
    playthrough_counts: dict[str, int] = {}
    replay_counts: dict[str, int] = {}
    for row in conn.execute(
        "SELECT igdb_id, COUNT(*) AS n, SUM(is_replay) AS n_replay FROM playthroughs GROUP BY igdb_id"
    ):
        playthrough_counts[row["igdb_id"]] = row["n"]
        replay_counts[row["igdb_id"]] = row["n_replay"] or 0

    games: list[GameRecord] = []
    for row in conn.execute(
        """
        SELECT g.igdb_id, g.title, g.genres, g.themes, g.developers, g.publishers, g.franchises,
               ug.rating, ug.total_hours, ug.status
        FROM games g
        LEFT JOIN user_games ug ON ug.igdb_id = g.igdb_id
        """
    ):
        games.append(
            {
                "igdb_id": row["igdb_id"],
                "title": row["title"],
                "rating": effective_rating(row["rating"]),
                "total_hours": row["total_hours"],
                "status": row["status"],
                "signal_category": signal_category(row["status"]),
                "tags": {
                    "genres": parse_tag_list(row["genres"]),
                    "themes": parse_tag_list(row["themes"]),
                    "developers": parse_tag_list(row["developers"]),
                    "publishers": parse_tag_list(row["publishers"]),
                    "franchises": parse_tag_list(row["franchises"]),
                },
                "playthrough_count": playthrough_counts.get(row["igdb_id"], 0),
                "replay_count": replay_counts.get(row["igdb_id"], 0),
            }
        )
    return games


def load_playthrough_years(conn: sqlite3.Connection) -> list[PlaythroughYearRecord]:
    """
    One record per playthrough with a parseable start_date and a logged
    hours_played, used for time-windowed analysis (genre evolution,
    taste drift). Playthroughs missing either are silently skipped —
    there's no reasonable year or weight to attribute them to.
    """
    records: list[PlaythroughYearRecord] = []
    for row in conn.execute("SELECT igdb_id, start_date, hours_played FROM playthroughs"):
        if not row["start_date"] or row["hours_played"] is None:
            continue
        try:
            year = dt.date.fromisoformat(row["start_date"][:10]).year
        except ValueError:
            continue
        records.append({"igdb_id": row["igdb_id"], "year": year, "hours": row["hours_played"]})
    return records
