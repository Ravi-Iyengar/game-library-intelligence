"""Sprint 6 tests."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage.db import get_connection, init_schema
from src.ml.dataset import build_dataset
from src.ml.targets import rating_target, completion_target, replay_target, engagement_score_target
from src.ml.train import train_target
from src.recommendation.candidate_pool import candidate_pool
from src.recommendation.scorer import score_candidates, DEFAULT_WEIGHTS
from src.recommendation.generate import generate_recommendations

from tests.test_ml import _seed_game, _synthetic_played_library


# ---------------------------------------------------------------------------
# candidate_pool.py
# ---------------------------------------------------------------------------

def test_candidate_pool_excludes_played_and_wishlist(tmp_path):
    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)
    _seed_game(conn, "1", "Played game", ["RPG"], [], 80.0, {"own_total_hours": 20, "playthrough_count": 1})
    _seed_game(conn, "2", "Backlog game", ["RPG"], [], 80.0, {})
    _seed_game(conn, "3", "Wishlist game", ["RPG"], [], 80.0, {"is_wishlist": 1})
    conn.commit()

    df, _ = build_dataset(conn)
    pool = candidate_pool(df)

    assert set(pool["igdb_id"]) == {"2"}


def test_candidate_pool_includes_in_progress_backlog_games(tmp_path):
    """A game with real hours logged but is_backlog=True (started, not
    beaten) belongs in the recommendation pool, not excluded as 'played'."""
    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)
    _seed_game(
        conn, "1", "Finished, settled", ["RPG"], [], 80.0,
        {"own_total_hours": 40, "playthrough_count": 1, "own_rating": 9, "is_backlog": 0},
    )
    _seed_game(
        conn, "2", "Started but not beaten", ["RPG"], [], 80.0,
        {"own_total_hours": 45, "playthrough_count": 1, "own_rating": 7, "is_backlog": 1},
    )
    conn.commit()

    df, _ = build_dataset(conn)
    pool = candidate_pool(df)

    assert set(pool["igdb_id"]) == {"2"}


# ---------------------------------------------------------------------------
# scorer.py
# ---------------------------------------------------------------------------

class _FakeRegressor:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return [self.value] * len(X)


class _FakeClassifier:
    def __init__(self, proba):
        self.proba = proba

    def predict_proba(self, X):
        import numpy as np
        return np.array([[1 - self.proba, self.proba]] * len(X))

    def predict(self, X):
        return [int(self.proba > 0.5)] * len(X)


def test_score_candidates_combines_all_four_models():
    candidates = pd.DataFrame({"igdb_id": ["1", "2"], "title": ["A", "B"], "f1": [0, 1]})
    models = {
        "rating": _FakeRegressor(8.0),
        "completion": _FakeClassifier(0.7),
        "replay": _FakeClassifier(0.3),
        "engagement": _FakeRegressor(0.6),
    }
    result = score_candidates(candidates, ["f1"], models)

    expected = 100 * (0.40 * 0.8 + 0.25 * 0.7 + 0.20 * 0.3 + 0.15 * 0.6)
    assert round(result["recommendation_score"].iloc[0], 2) == round(expected, 2)
    assert result["predicted_rating"].iloc[0] == 8.0
    assert result["predicted_completion_probability"].iloc[0] == 0.7


def test_score_candidates_redistributes_weight_when_model_missing():
    candidates = pd.DataFrame({"igdb_id": ["1"], "title": ["A"], "f1": [0]})
    # Only rating and completion models available — replay/engagement missing.
    models = {"rating": _FakeRegressor(10.0), "completion": _FakeClassifier(1.0),
              "replay": None, "engagement": None}
    result = score_candidates(candidates, ["f1"], models)

    # Remaining weights (0.40 rating + 0.25 completion) renormalize to sum to 1:
    # 0.40/0.65 and 0.25/0.65 — both predictions are "perfect" (1.0 normalized), so score = 100.
    assert result["recommendation_score"].iloc[0] == 100.0


def test_score_candidates_raises_when_no_models_available():
    candidates = pd.DataFrame({"igdb_id": ["1"], "title": ["A"], "f1": [0]})
    models = {"rating": None, "completion": None, "replay": None, "engagement": None}
    try:
        score_candidates(candidates, ["f1"], models)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# generate.py — full pipeline with real trained models
# ---------------------------------------------------------------------------

def test_generate_recommendations_end_to_end(tmp_path):
    conn = get_connection(tmp_path / "glip.db")
    init_schema(conn)

    _synthetic_played_library(conn, n=60)
    # A handful of unplayed backlog candidates — some RPG, some not, so the
    # trained "RPGs score higher" signal has something to actually rank.
    _seed_game(conn, "bl_1", "Backlog RPG", ["RPG"], ["DevA"], 85.0, {"genre_affinity": 8.0})
    _seed_game(conn, "bl_2", "Backlog Platformer", ["Platformer"], ["DevB"], 70.0, {"genre_affinity": 5.0})
    conn.commit()

    df, feature_columns = build_dataset(conn)
    X = df[feature_columns]

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    import joblib
    for target_name, builder, task in [
        ("rating", rating_target, "regression"),
        ("completion", completion_target, "classification"),
        ("replay", replay_target, "classification"),
        ("engagement", engagement_score_target, "regression"),
    ]:
        y = builder(df)
        result = train_target(X, y, task, target_name)
        assert result["model"] is not None
        joblib.dump(result["model"], models_dir / f"{target_name}_model.joblib")

    ranked = generate_recommendations(conn, models_dir=str(models_dir))

    assert set(ranked["igdb_id"]) == {"bl_1", "bl_2"}
    assert list(ranked["recommendation_score"]) == sorted(ranked["recommendation_score"], reverse=True)
    assert "explanation" in ranked.columns
    assert ranked.iloc[0]["explanation"]["source_model"] == "rating"

    # Recommendations should be persisted to the DB too.
    saved = conn.execute("SELECT * FROM recommendations").fetchall()
    assert len(saved) == 2
    saved_by_id = {row["igdb_id"]: row for row in saved}
    assert saved_by_id["bl_1"]["recommendation_score"] is not None
    explanation = json.loads(saved_by_id["bl_1"]["explanation_json"])
    assert "top_factors" in explanation
