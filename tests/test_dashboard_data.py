"""
Sprint 7 tests. Tests everything the dashboard depends on EXCEPT the
actual streamlit rendering in app.py, which can't be exercised here —
no streamlit or browser in this sandbox. See app.py's module docstring.

Uses its own seeding helper (_seed_analytics_game) rather than
tests/test_ml.py's _seed_game: that one populates the `features` table
(what Sprint 5/6 read from); the dashboard's analytics functions read
genres/themes/etc directly from `games` and rating/status/hours from
`user_games` (what Sprint 4 reads from) — a different data path, same
as tests/test_analytics.py already does it.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage.db import get_connection, init_schema
from streamlit_app.data_access import (
    load_overview,
    load_tag_table,
    load_replay_table,
    load_completion_table,
    load_taste_drift,
    load_recommendations,
)


def _seed_analytics_game(
    conn, igdb_id, title, genres=None, status=None, rating=None, hours=None,
    replay_count=0, enriched=True,
):
    conn.execute(
        "INSERT INTO games (igdb_id, title, genres, enriched_at) VALUES (?, ?, ?, ?)",
        (igdb_id, title, json.dumps(genres or []), "x" if enriched else None),
    )
    conn.execute(
        "INSERT INTO user_games (igdb_id, status, rating, total_hours) VALUES (?, ?, ?, ?)",
        (igdb_id, status, rating, hours),
    )
    for i in range(1 + replay_count):
        conn.execute(
            "INSERT INTO playthroughs (playthrough_id, igdb_id, is_replay) VALUES (?, ?, ?)",
            (int(igdb_id) * 100 + i, igdb_id, int(i > 0)),
        )


def test_load_overview_on_empty_db(tmp_path):
    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)
    overview = load_overview(conn)
    assert overview["total_games"] == 0


def test_load_overview_matches_real_counts(tmp_path):
    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)
    _seed_analytics_game(conn, "1", "A", ["RPG"], status="completed", rating=9)
    _seed_analytics_game(conn, "2", "B", ["RPG"], status="played", rating=0)
    conn.commit()

    overview = load_overview(conn)
    assert overview["total_games"] == 2
    assert overview["rated_games"] == 1  # game 2's rating=0 correctly excluded


def test_load_tag_table_returns_dataframe_with_expected_columns(tmp_path):
    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)
    _seed_analytics_game(conn, "1", "A", ["RPG"], rating=9, hours=10)
    _seed_analytics_game(conn, "2", "B", ["RPG"], rating=7, hours=20)
    conn.commit()

    table = load_tag_table(conn, "genres")
    assert list(table.columns) == ["tag", "game_count", "total_hours", "avg_rating", "hours_weighted_rating"]
    assert table.iloc[0]["tag"] == "RPG"
    assert table.iloc[0]["game_count"] == 2


def test_load_tag_table_empty_when_no_enrichment(tmp_path):
    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)
    conn.execute("INSERT INTO games (igdb_id, title) VALUES ('1', 'Unenriched')")
    conn.commit()

    table = load_tag_table(conn, "genres")
    assert table.empty


def test_load_replay_table(tmp_path):
    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)
    _seed_analytics_game(conn, "1", "A", ["RPG"], replay_count=2)
    _seed_analytics_game(conn, "2", "B", ["RPG"], replay_count=0)
    conn.commit()

    table = load_replay_table(conn, "genres")
    assert len(table) == 1
    assert table.iloc[0]["tag"] == "RPG"
    assert table.iloc[0]["total_replays"] == 2


def test_load_completion_table(tmp_path):
    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)
    _seed_analytics_game(conn, "1", "A", ["RPG"], status="completed")
    _seed_analytics_game(conn, "2", "B", ["RPG"], status="abandoned")
    conn.commit()

    table = load_completion_table(conn, "genres")
    assert table.iloc[0]["tag"] == "RPG"
    assert table.iloc[0]["games_logged"] == 2
    assert table.iloc[0]["completion_rate"] == 0.5


def test_load_taste_drift_returns_none_similarity_without_enrichment(tmp_path):
    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)
    conn.execute("INSERT INTO games (igdb_id, title) VALUES ('1', 'Unenriched')")
    conn.commit()

    top_genres, similarity_df = load_taste_drift(conn)
    assert top_genres == {}
    assert similarity_df is None


def test_load_taste_drift_with_real_data(tmp_path):
    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)
    for i in range(6):
        _seed_analytics_game(conn, str(i), f"Game {i}", ["RPG"])
    conn.execute(
        "INSERT INTO playthroughs (playthrough_id, igdb_id, start_date, hours_played) "
        "VALUES (9001, '0', '2023-01-01', 20)"
    )
    conn.execute(
        "INSERT INTO playthroughs (playthrough_id, igdb_id, start_date, hours_played) "
        "VALUES (9002, '1', '2024-01-01', 30)"
    )
    conn.commit()

    top_genres, similarity_df = load_taste_drift(conn, min_occurrences=2)
    assert 2023 in top_genres
    assert 2024 in top_genres
    assert similarity_df is not None
    assert similarity_df.loc[2023, 2023] == 1.0


def test_load_recommendations_empty_when_none_generated(tmp_path):
    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)
    table = load_recommendations(conn)
    assert table.empty


def test_load_recommendations_reads_written_rows(tmp_path):
    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)
    conn.execute("INSERT INTO games (igdb_id, title) VALUES ('1', 'Some Game')")
    conn.execute(
        """INSERT INTO recommendations (igdb_id, recommendation_score, explanation_json, generated_at)
           VALUES ('1', 87.5, ?, 'x')""",
        (json.dumps({"method": "shap", "top_factors": [{"feature": "genre__RPG", "contribution": 0.4}]}),),
    )
    conn.commit()

    table = load_recommendations(conn)
    assert len(table) == 1
    assert table.iloc[0]["title"] == "Some Game"
    assert table.iloc[0]["recommendation_score"] == 87.5
    assert table.iloc[0]["explanation"]["method"] == "shap"
