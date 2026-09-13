"""
Sprint 3 tests. Embeddings are mocked (patch get_embedder) so these run
without network access or the sentence-transformers package installed.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage.db import get_connection, init_schema
from src.feature_engineering.vocabulary import build_vocabulary, one_hot
from src.feature_engineering.affinity import compute_tag_affinities
from src.feature_engineering.engagement import compute_engagement_metrics
from src.feature_engineering.reviews import select_representative_review, extract_review_text_features
from src.feature_engineering.build_features import build_feature_store


# ---------------------------------------------------------------------------
# vocabulary.py
# ---------------------------------------------------------------------------

def test_build_vocabulary_prunes_rare_tags(tmp_path):
    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)

    rows = [
        ("1", json.dumps(["RPG", "Rare Tag A"])),
        ("2", json.dumps(["RPG", "Rare Tag B"])),
        ("3", json.dumps(["RPG"])),
        ("4", json.dumps(["RPG"])),
        ("5", json.dumps(["RPG"])),
    ]
    for igdb_id, genres in rows:
        conn.execute(
            "INSERT INTO games (igdb_id, title, genres, enriched_at) VALUES (?, ?, ?, 'x')",
            (igdb_id, f"Game {igdb_id}", genres),
        )
    conn.commit()

    vocab = build_vocabulary(conn, "genres", min_occurrences=5)
    assert vocab == ["RPG"]  # rare tags (count 1 each) pruned out

    one_hot_result = one_hot(["RPG", "Rare Tag A"], vocab)
    assert one_hot_result == {"RPG": 1}


# ---------------------------------------------------------------------------
# affinity.py
# ---------------------------------------------------------------------------

def test_affinity_is_leave_one_out_and_hours_weighted():
    games = [
        {"igdb_id": "A", "rating": 10, "total_hours": 100, "tags": {"genres": ["RPG"], "themes": [], "developers": [], "publishers": [], "franchises": []}},
        {"igdb_id": "B", "rating": 6, "total_hours": 10, "tags": {"genres": ["RPG"], "themes": [], "developers": [], "publishers": [], "franchises": []}},
        {"igdb_id": "C", "rating": None, "total_hours": None, "tags": {"genres": ["RPG"], "themes": [], "developers": [], "publishers": [], "franchises": []}},
        {"igdb_id": "D", "rating": 5, "total_hours": 5, "tags": {"genres": ["Platformer"], "themes": [], "developers": [], "publishers": [], "franchises": []}},
    ]
    result = compute_tag_affinities(games)

    # A's genre_affinity should reflect only B (weight 10) since C has no rating
    # and A itself is excluded: (6*10)/10 = 6.0
    assert result["A"]["genre_affinity"] == 6.0

    # B's genre_affinity should reflect only A (weight 100): (10*100)/100 = 10.0
    assert result["B"]["genre_affinity"] == 10.0

    # C has no rating but should still get an affinity computed from A and B:
    # (10*100 + 6*10) / (100+10) = 1060/110
    assert round(result["C"]["genre_affinity"], 4) == round(1060 / 110, 4)

    # D shares no tag with anyone else -> None
    assert result["D"]["genre_affinity"] is None
    # And D has no themes/developers/etc at all -> also None
    assert result["D"]["theme_affinity"] is None


# ---------------------------------------------------------------------------
# engagement.py
# ---------------------------------------------------------------------------

def test_engagement_metrics():
    playthroughs = [
        {"playthrough_id": 1, "is_replay": 0, "start_date": "2024-01-01"},
        {"playthrough_id": 2, "is_replay": 1, "start_date": "2024-03-01"},
    ]
    sessions = [
        {"playthrough_id": 1, "range_start_date": "2024-01-01", "hours": 2, "minutes": 30},
        {"playthrough_id": 1, "range_start_date": "2024-01-02", "hours": 1, "minutes": 0},
        {"playthrough_id": 2, "range_start_date": "2024-03-01", "hours": 3, "minutes": 0},
    ]
    metrics = compute_engagement_metrics(playthroughs, sessions)

    assert metrics["playthrough_count"] == 2
    assert metrics["replay_count"] == 1
    assert metrics["session_count"] == 3
    assert round(metrics["average_session_length_hours"], 4) == round((2.5 + 1 + 3) / 3, 4)
    assert metrics["days_played"] == 3
    assert metrics["avg_days_between_replays"] == 60  # Jan 1 -> Mar 1 (leap year 2024)


# ---------------------------------------------------------------------------
# reviews.py
# ---------------------------------------------------------------------------

def test_select_representative_review_prefers_master():
    playthroughs = [
        {"is_master": 0, "updated_at": "2024-05-01", "review": "later but not master"},
        {"is_master": 1, "updated_at": "2024-01-01", "review": "the master review"},
    ]
    assert select_representative_review(playthroughs) == "the master review"


def test_select_representative_review_falls_back_to_most_recent():
    playthroughs = [
        {"is_master": 0, "updated_at": "2024-01-01", "review": "older"},
        {"is_master": 0, "updated_at": "2024-05-01", "review": "newer"},
    ]
    assert select_representative_review(playthroughs) == "newer"


def test_extract_review_text_features_handles_missing_review():
    assert extract_review_text_features(None) == {
        "review_exists": 0, "review_length": 0, "review_word_count": 0,
    }
    features = extract_review_text_features("four little words")
    assert features == {"review_exists": 1, "review_length": 17, "review_word_count": 3}


# ---------------------------------------------------------------------------
# build_features.py — full build against a small synthetic library
# ---------------------------------------------------------------------------

def _seed_small_library(conn):
    games = [
        ("1", "Baldur's Gate III", ["RPG"], ["Fantasy"], ["Larian Studios"], ["Larian Studios"]),
        ("2", "Divinity: Original Sin 2", ["RPG"], ["Fantasy"], ["Larian Studios"], ["Larian Studios"]),
        ("3", "Elden Ring", ["RPG"], ["Fantasy"], ["FromSoftware"], ["Bandai Namco"]),
        ("4", "Celeste", ["Platformer"], ["Difficult"], ["Extremely OK Games"], ["Extremely OK Games"]),
        ("5", "Stardew Valley", ["Simulator"], ["Farming"], ["ConcernedApe"], ["ConcernedApe"]),
    ]
    for igdb_id, title, genres, themes, devs, pubs in games:
        conn.execute(
            """INSERT INTO games (igdb_id, title, genres, themes, developers, publishers, enriched_at)
               VALUES (?, ?, ?, ?, ?, ?, 'x')""",
            (igdb_id, title, json.dumps(genres), json.dumps(themes), json.dumps(devs), json.dumps(pubs)),
        )

    user_games = [
        ("1", "completed", 9, 60.0, 0, 0, 0, 1),
        ("2", "completed", 10, 90.0, 0, 0, 0, 1),
        ("3", "played", 8, 120.0, 0, 0, 0, 1),
        ("4", "abandoned", 4, 3.0, 0, 0, 0, 0),
        ("5", "shelved", None, 15.0, 1, 0, 0, 0),
    ]
    for igdb_id, status, rating, hours, backlog, playing, wishlist, liked in user_games:
        conn.execute(
            """INSERT INTO user_games (igdb_id, status, rating, total_hours, is_backlog, is_playing, is_wishlist, is_liked)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (igdb_id, status, rating, hours, backlog, playing, wishlist, liked),
        )

    conn.execute(
        """INSERT INTO playthroughs (playthrough_id, igdb_id, rating, is_replay, is_master, start_date, updated_at, review)
           VALUES (100, '1', 9, 0, 1, '2024-01-01', '2024-02-01', 'Loved every second of this game and its choices.')"""
    )
    conn.execute(
        """INSERT INTO sessions (session_id, playthrough_id, range_start_date, hours, minutes)
           VALUES (900, 100, '2024-01-01', 5, 0)"""
    )
    conn.commit()


def test_build_feature_store_end_to_end(tmp_path):
    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)
    _seed_small_library(conn)

    with patch("src.feature_engineering.embeddings.get_embedder", return_value=None):
        summary = build_feature_store(conn, min_occurrences=2)  # 2 RPGs qualifies "RPG"

    assert summary["games"] == 5
    assert summary["with_review_embedding"] == 0  # embedder mocked to None
    assert summary["vocab_genres"] == 1  # only "RPG" appears >=2 times

    row = conn.execute("SELECT feature_json FROM features WHERE igdb_id = '1'").fetchone()
    features = json.loads(row["feature_json"])

    assert features["own_rating"] == 9
    assert features["genre_one_hot"] == {"RPG": 1}
    # BG3's genre_affinity should come from Divinity OS2 (10) and Elden Ring (8),
    # weighted by their hours (90 and 120): (10*90 + 8*120)/(90+120)
    expected = (10 * 90 + 8 * 120) / (90 + 120)
    assert round(features["genre_affinity"], 4) == round(expected, 4)
    # developer_affinity: only other Larian game is Divinity OS2 (rating 10)
    assert features["developer_affinity"] == 10.0
    assert features["review_exists"] == 1
    assert features["review_word_count"] == 9
    assert features["review_embedding"] is None
    assert features["signal_category"] == "positive"
    assert features["playthrough_count"] == 1
    assert features["session_count"] == 1

    stardew = json.loads(
        conn.execute("SELECT feature_json FROM features WHERE igdb_id = '5'").fetchone()["feature_json"]
    )
    assert stardew["own_rating"] is None
    assert stardew["signal_category"] == "neutral"  # shelved
    assert stardew["is_backlog"] == 1


def test_rating_zero_is_treated_as_unrated(tmp_path):
    """A raw rating of 0 (Backloggd's 'no rating given' default) must not
    be treated as a genuine zero score anywhere rating feeds a computation."""
    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)
    _seed_small_library(conn)

    # Overwrite game 4 (Celeste, abandoned) to have a raw rating of 0
    # instead of 4, simulating an unrated-but-logged entry.
    conn.execute("UPDATE user_games SET rating = 0 WHERE igdb_id = '4'")
    conn.commit()

    with patch("src.feature_engineering.embeddings.get_embedder", return_value=None):
        build_feature_store(conn, min_occurrences=2)

    celeste = json.loads(
        conn.execute("SELECT feature_json FROM features WHERE igdb_id = '4'").fetchone()["feature_json"]
    )
    assert celeste["own_rating"] is None  # not 0
