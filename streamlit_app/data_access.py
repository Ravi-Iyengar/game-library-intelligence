"""
Data access for the Sprint 7 dashboard, deliberately kept free of any
`streamlit` import. This sandbox has no streamlit/plotly installed and
no browser, so app.py itself can't be run or visually verified here —
but everything in this module is plain Python + pandas + sqlite3, and
is unit-tested the same way every other sprint's logic has been (see
tests/test_dashboard_data.py).

app.py should do almost nothing but call these functions and hand the
result to st.dataframe/st.bar_chart/etc — keeping the untestable
(streamlit rendering) surface as thin as possible.
"""
from __future__ import annotations

import json
import sqlite3

import pandas as pd

from src.analytics.completion_analysis import completion_rate_by_tag
from src.analytics.counts import library_overview
from src.analytics.data_loader import load_games, load_playthrough_years
from src.analytics.replay_analysis import most_replayed
from src.analytics.tag_preferences import tag_preference_table
from src.analytics.taste_drift import (
    build_year_vectors,
    taste_similarity_matrix,
    top_tags_by_year,
    yearly_tag_hours,
)
from src.feature_engineering.vocabulary import build_vocabulary
from src.recommendation.generate import generate_recommendations

TAG_CATEGORIES = ("genres", "themes", "developers", "publishers")


def load_overview(conn: sqlite3.Connection) -> dict:
    games = load_games(conn)
    return library_overview(games)


def load_tag_table(conn: sqlite3.Connection, category: str, min_games: int = 2) -> pd.DataFrame:
    games = load_games(conn)
    rows = tag_preference_table(games, category, min_games=min_games)
    return pd.DataFrame(rows)


def load_replay_table(conn: sqlite3.Connection, category: str) -> pd.DataFrame:
    games = load_games(conn)
    rows = most_replayed(games, category)
    return pd.DataFrame(rows)


def load_completion_table(conn: sqlite3.Connection, category: str, min_games: int = 2) -> pd.DataFrame:
    games = load_games(conn)
    rows = completion_rate_by_tag(games, category, min_games=min_games)
    return pd.DataFrame(rows)


def load_taste_drift(conn: sqlite3.Connection, min_occurrences: int = 5, top_n: int = 5):
    """Returns (top_genres_by_year: dict, similarity_df: pd.DataFrame | None)."""
    games = load_games(conn)
    playthrough_years = load_playthrough_years(conn)
    vocab = build_vocabulary(conn, "genres", min_occurrences=min_occurrences)

    hours_by_year = yearly_tag_hours(games, playthrough_years, category="genres")
    top_genres = top_tags_by_year(hours_by_year, top_n=top_n)

    if not vocab or not hours_by_year:
        return top_genres, None

    vectors = build_year_vectors(hours_by_year, vocab)
    matrix = taste_similarity_matrix(vectors)
    similarity_df = pd.DataFrame(matrix).sort_index().sort_index(axis=1)
    return top_genres, similarity_df


def load_recommendations(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Reads whatever is currently in the `recommendations` table (written
    by scripts/run_ml_training.py + scripts/run_recommendations.py, or
    by the "Regenerate" button below), joined with title. Empty
    DataFrame (not an error) if recommendations haven't been generated
    yet — the app should show guidance, not crash, in that case.
    """
    rows = conn.execute(
        """
        SELECT r.*, g.title
        FROM recommendations r
        JOIN games g ON g.igdb_id = r.igdb_id
        ORDER BY r.recommendation_score DESC
        """
    ).fetchall()

    records = []
    for row in rows:
        record = dict(row)
        record["explanation"] = json.loads(record.pop("explanation_json") or "{}")
        records.append(record)
    return pd.DataFrame(records)


def regenerate_recommendations(conn: sqlite3.Connection, models_dir: str = "data/models") -> pd.DataFrame:
    """Re-runs Sprint 6 scoring live and returns the freshly ranked DataFrame.
    Raises ValueError with a clear message if no models are trained yet or
    the backlog pool is empty — app.py should catch this and show it
    with st.error rather than letting the page crash."""
    return generate_recommendations(conn, models_dir=models_dir)
