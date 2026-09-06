"""Sprint 4 tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analytics.counts import library_overview, status_counts, signal_category_counts
from src.analytics.data_loader import load_games
from src.storage.db import get_connection, init_schema
from src.feature_engineering.completion import signal_category
from src.analytics.tag_preferences import tag_preference_table
from src.analytics.replay_analysis import most_replayed
from src.analytics.completion_analysis import completion_rate_by_tag
from src.analytics.taste_drift import (
    yearly_tag_hours,
    build_year_vectors,
    taste_similarity_matrix,
    top_tags_by_year,
)


def _game(igdb_id, title, status, rating, hours, genres, developers=None, replay_count=0):
    return {
        "igdb_id": igdb_id,
        "title": title,
        "rating": rating,
        "total_hours": hours,
        "status": status,
        "signal_category": signal_category(status),
        "tags": {"genres": genres, "themes": [], "developers": developers or [], "publishers": [], "franchises": []},
        "playthrough_count": 1 + replay_count,
        "replay_count": replay_count,
    }


GAMES = [
    _game("1", "BG3", "completed", 9, 60, ["RPG"], ["Larian"]),
    _game("2", "DOS2", "completed", 10, 90, ["RPG"], ["Larian"], replay_count=2),
    _game("3", "Elden Ring", "played", 8, 120, ["RPG", "Action"], ["FromSoftware"], replay_count=1),
    _game("4", "Celeste", "abandoned", 4, 3, ["Platformer"]),
    _game("5", "Stardew Valley", "shelved", None, 15, ["Simulator"]),
]


def test_library_overview():
    overview = library_overview(GAMES)
    assert overview["total_games"] == 5
    assert overview["rated_games"] == 4
    assert overview["status_counts"] == {
        "completed": 2, "played": 1, "abandoned": 1, "shelved": 1,
    }
    assert overview["signal_category_counts"] == {
        "positive": 2, "negative": 1, "neutral": 2,
    }


def test_tag_preference_table_weights_by_hours():
    rows = tag_preference_table(GAMES, "genres", min_games=1)
    rpg = next(r for r in rows if r["tag"] == "RPG")
    assert rpg["game_count"] == 3
    # (9*60 + 10*90 + 8*120) / (60+90+120)
    expected = (9 * 60 + 10 * 90 + 8 * 120) / (60 + 90 + 120)
    assert round(rpg["hours_weighted_rating"], 4) == round(expected, 4)

    # min_games filters out single-appearance tags
    rows_pruned = tag_preference_table(GAMES, "genres", min_games=2)
    tags_present = {r["tag"] for r in rows_pruned}
    assert "Platformer" not in tags_present  # only 1 game
    assert "RPG" in tags_present


def test_most_replayed():
    rows = most_replayed(GAMES, "developers")
    larian = next(r for r in rows if r["tag"] == "Larian")
    assert larian["total_replays"] == 2  # from DOS2 only
    assert larian["games_with_replay"] == 1
    assert larian["games_total"] == 2  # BG3 + DOS2 both tagged Larian


def test_completion_rate_by_tag():
    rows = completion_rate_by_tag(GAMES, "genres", min_games=1)
    rpg = next(r for r in rows if r["tag"] == "RPG")
    assert rpg["games_logged"] == 3
    assert round(rpg["completion_rate"], 4) == round(2 / 3, 4)  # BG3, DOS2 completed; Elden Ring "played"
    assert round(rpg["positive_rate"], 4) == round(2 / 3, 4)  # BG3, DOS2 completed are positive; Elden Ring "played" is neutral


def test_load_games_treats_rating_zero_as_unrated(tmp_path):
    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)
    conn.execute("INSERT INTO games (igdb_id, title) VALUES ('1', 'Fire Emblem Awakening')")
    conn.execute(
        "INSERT INTO user_games (igdb_id, status, rating, total_hours) VALUES ('1', 'completed', 0, 0)"
    )
    conn.commit()

    games = load_games(conn)
    assert games[0]["rating"] is None  # raw 0 -> None, not a genuine zero score


def test_taste_drift_similarity():
    playthrough_years = [
        {"igdb_id": "1", "year": 2023, "hours": 60},
        {"igdb_id": "2", "year": 2023, "hours": 90},
        {"igdb_id": "4", "year": 2024, "hours": 3},
    ]
    hours_by_year = yearly_tag_hours(GAMES, playthrough_years, category="genres")
    assert hours_by_year[2023] == {"RPG": 150}
    assert hours_by_year[2024] == {"Platformer": 3}

    vocab = ["RPG", "Platformer"]
    vectors = build_year_vectors(hours_by_year, vocab)
    assert vectors[2023] == [150, 0]
    assert vectors[2024] == [0, 3]

    matrix = taste_similarity_matrix(vectors)
    # Completely disjoint genre vectors -> orthogonal -> similarity 0
    assert matrix[2023][2024] == 0
    assert matrix[2023][2023] == 1.0

    top = top_tags_by_year(hours_by_year, top_n=1)
    assert top[2023] == [("RPG", 150)]
