"""
Sprint 5 tests. Uses only RandomForest-scale data (no xgboost/catboost/
lightgbm in this environment) with a synthetic library large enough for
an actual 80/20 split + 5-fold CV to run meaningfully (~60 played games).
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage.db import get_connection, init_schema
from src.ml.dataset import build_dataset, has_been_played
from src.ml.targets import (
    completion_target, replay_target, rating_target, engagement_score_target,
)
from src.ml.train import train_target
from src.ml.model_candidates import get_candidates


# ---------------------------------------------------------------------------
# has_been_played
# ---------------------------------------------------------------------------

def test_has_been_played_requires_real_evidence():
    import pandas as pd

    # Labeled "completed" but zero hours, zero playthroughs, no real rating —
    # the exact pattern found in the real export.
    placeholder = pd.Series({"own_total_hours": 0, "playthrough_count": 0, "own_rating": None})
    assert has_been_played(placeholder) is False

    genuinely_played = pd.Series({"own_total_hours": 12.0, "playthrough_count": 1, "own_rating": None})
    assert has_been_played(genuinely_played) is True

    rated_but_no_hours_logged = pd.Series({"own_total_hours": 0, "playthrough_count": 0, "own_rating": 7})
    assert has_been_played(rated_but_no_hours_logged) is True


# ---------------------------------------------------------------------------
# dataset.py — flattening and pre-play/post-play separation
# ---------------------------------------------------------------------------

def _seed_game(conn, igdb_id, title, genres, developers, agg_rating, feature_overrides):
    conn.execute(
        "INSERT INTO games (igdb_id, title, genres, developers, aggregated_rating, aggregated_rating_count, release_date, enriched_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'x')",
        (igdb_id, title, json.dumps(genres), json.dumps(developers), agg_rating, 100, "2020-06-15"),
    )
    base = {
        "igdb_id": igdb_id, "own_rating": None, "own_total_hours": 0,
        "playthrough_count": 0, "replay_count": 0, "session_count": 0,
        "average_session_length_hours": None, "days_played": 0, "avg_days_between_replays": None,
        "status_raw": None, "signal_category": "unknown", "is_completed": 0, "is_abandoned": 0,
        "is_shelved": 0, "is_backlog": 0, "is_playing": 0, "is_wishlist": 0, "is_liked": 0,
        "replay_rate": None,
        "genre_one_hot": {tag: int(tag in genres) for tag in ["RPG", "Platformer", "Simulator"]},
        "theme_one_hot": {}, "developer_one_hot": {}, "publisher_one_hot": {},
        "genre_affinity": None, "theme_affinity": None, "developer_affinity": None,
        "publisher_affinity": None, "franchise_affinity": None,
        "review_exists": 0, "review_length": 0, "review_word_count": 0, "review_embedding": None,
    }
    base.update(feature_overrides)
    conn.execute(
        "INSERT INTO features (igdb_id, feature_json, computed_at) VALUES (?, ?, 'x')",
        (igdb_id, json.dumps(base)),
    )


def test_build_dataset_flattens_and_splits_pre_play_features(tmp_path):
    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)
    _seed_game(
        conn, "1", "Game A", ["RPG"], ["Larian"], 91.0,
        {"own_rating": 9, "own_total_hours": 40, "playthrough_count": 1, "genre_affinity": 8.5},
    )
    conn.commit()

    df, pre_play_cols = build_dataset(conn)

    assert "genre__RPG" in df.columns
    assert df.loc[0, "genre__RPG"] == 1
    assert "genre_affinity" in pre_play_cols
    assert "aggregated_rating" in pre_play_cols
    assert df.loc[0, "release_year"] == 2020

    # Post-play-only fields must NOT be offered as model inputs.
    for leaky in ("own_rating", "own_total_hours", "review_embedding", "playthrough_count", "status_raw"):
        assert leaky not in pre_play_cols


def test_build_dataset_has_been_played_flag(tmp_path):
    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)
    _seed_game(conn, "1", "Never touched", ["RPG"], [], 80.0, {})  # all defaults: 0 hours, no rating
    _seed_game(conn, "2", "Actually played", ["RPG"], [], 80.0, {"own_total_hours": 20, "playthrough_count": 1})
    conn.commit()

    df, _ = build_dataset(conn)
    df = df.set_index("igdb_id")
    assert df.loc["1", "has_been_played"] == False  # noqa: E712
    assert df.loc["2", "has_been_played"] == True  # noqa: E712


def test_has_settled_outcome_excludes_in_progress_backlog_games(tmp_path):
    """
    Confirmed with the library's owner: is_backlog=True means "haven't
    beaten this yet," even when real hours are logged (e.g. 45 hours
    into Red Dead Redemption 2, still marked is_backlog=True). That
    outcome isn't final — has_settled_outcome must exclude it, even
    though has_been_played correctly reports real engagement.
    """
    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)
    _seed_game(conn, "1", "Never touched", ["RPG"], [], 80.0, {})
    _seed_game(
        conn, "2", "Finished, settled", ["RPG"], [], 80.0,
        {"own_total_hours": 40, "playthrough_count": 1, "own_rating": 9, "is_backlog": 0},
    )
    _seed_game(
        conn, "3", "Started but not beaten", ["RPG"], [], 80.0,
        {"own_total_hours": 45, "playthrough_count": 1, "own_rating": 7, "is_backlog": 1},
    )
    conn.commit()

    df, _ = build_dataset(conn)
    df = df.set_index("igdb_id")

    assert df.loc["1", "has_been_played"] == False  # noqa: E712
    assert df.loc["1", "has_settled_outcome"] == False  # noqa: E712

    assert df.loc["2", "has_been_played"] == True  # noqa: E712
    assert df.loc["2", "has_settled_outcome"] == True  # noqa: E712

    # The key case: real evidence of play, but still in progress.
    assert df.loc["3", "has_been_played"] == True  # noqa: E712
    assert df.loc["3", "has_settled_outcome"] == False  # noqa: E712


# ---------------------------------------------------------------------------
# targets.py
# ---------------------------------------------------------------------------

def test_targets_only_defined_for_played_games(tmp_path):
    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)
    _seed_game(conn, "1", "Unplayed", ["RPG"], [], 80.0, {})
    _seed_game(
        conn, "2", "Completed and rated", ["RPG"], [], 80.0,
        {"own_rating": 9, "own_total_hours": 30, "playthrough_count": 1, "status_raw": "completed",
         "is_completed": 1, "replay_count": 0},
    )
    _seed_game(
        conn, "3", "Replayed", ["RPG"], [], 80.0,
        {"own_rating": 7, "own_total_hours": 15, "playthrough_count": 2, "status_raw": "played",
         "replay_count": 1},
    )
    conn.commit()
    df, _ = build_dataset(conn)
    df = df.set_index("igdb_id", drop=False)

    rating_y = rating_target(df)
    assert rating_y.loc[df["igdb_id"] == "1"].isna().all()
    assert rating_y[df["igdb_id"] == "2"].iloc[0] == 9

    completion_y = completion_target(df)
    assert completion_y[df["igdb_id"] == "2"].iloc[0] == 1
    assert completion_y[df["igdb_id"] == "3"].iloc[0] == 0  # played, not completed

    replay_y = replay_target(df)
    assert replay_y[df["igdb_id"] == "3"].iloc[0] == 1
    assert replay_y[df["igdb_id"] == "2"].iloc[0] == 0

    engagement_y = engagement_score_target(df)
    assert engagement_y[df["igdb_id"] == "1"].isna().all()
    assert 0 <= engagement_y[df["igdb_id"] == "2"].iloc[0] <= 1


# ---------------------------------------------------------------------------
# model_candidates.py
# ---------------------------------------------------------------------------

def test_get_candidates_always_includes_random_forest():
    regression_candidates = get_candidates("regression")
    classification_candidates = get_candidates("classification")
    assert "RandomForest" in regression_candidates
    assert "RandomForest" in classification_candidates


# ---------------------------------------------------------------------------
# train.py — small end-to-end run
# ---------------------------------------------------------------------------

def _synthetic_played_library(conn, n=60, seed=7):
    rng = random.Random(seed)
    genres_pool = ["RPG", "Platformer", "Simulator", "Action"]
    for i in range(n):
        genres = rng.sample(genres_pool, k=rng.randint(1, 2))
        is_rpg = "RPG" in genres
        # Make the signal learnable: RPGs tend to get rated higher and replayed more.
        rating = rng.randint(7, 10) if is_rpg else rng.randint(3, 8)
        hours = rng.randint(10, 100)
        replay_count = rng.randint(1, 3) if (is_rpg and rng.random() < 0.6) else 0
        completed = 1 if rating >= 6 else 0
        _seed_game(
            conn, str(i), f"Game {i}", genres, ["DevA" if is_rpg else "DevB"], rng.uniform(60, 95),
            {
                "own_rating": rating, "own_total_hours": hours,
                "playthrough_count": 1 + replay_count, "replay_count": replay_count,
                "status_raw": "completed" if completed else "played",
                "is_completed": completed,
                "genre_affinity": 8.0 if is_rpg else 5.0,
            },
        )
    conn.commit()


def test_train_target_runs_end_to_end_on_random_forest(tmp_path, monkeypatch):
    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)
    _synthetic_played_library(conn, n=60)

    df, feature_columns = build_dataset(conn)
    X = df[feature_columns]

    # Force RandomForest-only to keep this fast and dependency-free.
    import src.ml.train as train_module
    monkeypatch.setattr(train_module, "get_candidates", lambda task: get_candidates(task))

    y = rating_target(df)
    result = train_target(X, y, "regression", "rating")

    assert result["n_eligible"] == 60  # all synthetic games were "played"
    assert result["best_candidate"] == "RandomForest"  # only candidate available here
    assert "rmse" in result["holdout_metrics"]
    assert result["model"] is not None

    # The fitted pipeline should be usable for prediction on new rows.
    preds = result["model"].predict(X.head(3))
    assert len(preds) == 3


def test_train_target_skips_when_too_few_eligible_rows(tmp_path):
    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)
    _seed_game(
        conn, "1", "Only one played game", ["RPG"], [], 80.0,
        {"own_rating": 8, "own_total_hours": 10, "playthrough_count": 1, "status_raw": "completed"},
    )
    conn.commit()
    df, feature_columns = build_dataset(conn)
    X = df[feature_columns]
    y = rating_target(df)

    result = train_target(X, y, "regression", "rating")
    assert result["model"] is None
    assert result["best_candidate"] is None


def test_train_target_skips_when_no_usable_features(tmp_path):
    """Un-enriched library (Sprint 2 not run yet): every pre-play feature
    column is entirely missing. Must fail gracefully, not crash sklearn."""
    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)
    for i in range(25):
        conn.execute(
            "INSERT INTO games (igdb_id, title) VALUES (?, ?)", (str(i), f"Game {i}")
        )
        conn.execute(
            "INSERT INTO features (igdb_id, feature_json, computed_at) VALUES (?, ?, 'x')",
            (str(i), json.dumps({
                "igdb_id": str(i), "own_rating": 7, "own_total_hours": 10,
                "playthrough_count": 1, "replay_count": 0, "status_raw": "completed",
                "genre_one_hot": {}, "theme_one_hot": {}, "developer_one_hot": {},
                "publisher_one_hot": {}, "genre_affinity": None, "theme_affinity": None,
                "developer_affinity": None, "publisher_affinity": None, "franchise_affinity": None,
            })),
        )
    conn.commit()

    df, feature_columns = build_dataset(conn)
    X = df[feature_columns]
    y = rating_target(df)

    result = train_target(X, y, "regression", "rating")
    assert result["model"] is None
    assert result["best_candidate"] is None


def test_explain_predictions_handles_all_nan_feature_column(tmp_path):
    """
    A real bug caught during Sprint 6 integration: SimpleImputer silently
    DROPS any column that was entirely NaN during fit (e.g. an affinity
    category no game in the training set had a value for), so the
    imputer's transform() output can be narrower than its input. Naively
    re-labeling that output with the original (pre-drop) column names
    misattributes every feature name after the dropped one. This locks
    in the fix: explain_predictions must use the imputer's actual output
    columns, not the model's input columns.
    """
    import warnings
    from unittest.mock import patch
    from src.ml.explain import explain_predictions

    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)
    for i in range(25):
        rating = 8 if i % 2 == 0 else 4
        conn.execute("INSERT INTO games (igdb_id, title) VALUES (?, ?)", (str(i), f"Game {i}"))
        conn.execute(
            "INSERT INTO features (igdb_id, feature_json, computed_at) VALUES (?, ?, 'x')",
            (str(i), json.dumps({
                "igdb_id": str(i), "own_rating": rating, "own_total_hours": 10,
                "playthrough_count": 1, "replay_count": 0, "status_raw": "completed",
                "genre_one_hot": {"RPG": i % 2}, "theme_one_hot": {}, "developer_one_hot": {},
                "publisher_one_hot": {},
                "genre_affinity": float(rating),
                # These three are entirely None across every single row —
                # exactly the condition that makes SimpleImputer drop them.
                "theme_affinity": None, "developer_affinity": None, "publisher_affinity": None,
                "franchise_affinity": None,
            })),
        )
    conn.commit()

    df, feature_columns = build_dataset(conn)
    assert "theme_affinity" in feature_columns  # present in the schema, all-NaN in the data
    X = df[feature_columns]
    y = rating_target(df)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = train_target(X, y, "regression", "rating")

    imputer = result["model"].named_steps["imputer"]
    assert imputer.n_features_in_ == len(feature_columns)  # fit on the full input width
    surviving = imputer.get_feature_names_out(feature_columns)
    assert len(surviving) < len(feature_columns)  # confirms the drop actually happened here

    with patch("src.ml.explain._shap_available", return_value=False):
        explanations = explain_predictions(result["model"], X.head(3), top_n=3)

    assert len(explanations) == 3
    for exp in explanations:
        # Every reported feature name must be one that actually survived
        # imputation — never a name from the dropped, all-NaN columns.
        for factor in exp["top_factors"]:
            assert factor["feature"] in surviving
            assert factor["feature"] not in ("theme_affinity", "developer_affinity", "publisher_affinity")
