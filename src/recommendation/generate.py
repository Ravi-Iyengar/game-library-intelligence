"""
Sprint 6 — recommendation engine.

Input: your backlog (unplayed games, evidence-based — see
candidate_pool.py). Output: that pool ranked by a composite predicted-
enjoyment score, each with an explanation, written to the
`recommendations` table.

Usage (see scripts/run_recommendations.py for the CLI):

    from src.storage.db import get_connection
    from src.recommendation.generate import generate_recommendations

    conn = get_connection("data/processed/glip.db")
    ranked = generate_recommendations(conn, models_dir="data/models")
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3

import pandas as pd

from src.ml.dataset import build_dataset
from src.recommendation.candidate_pool import candidate_pool
from src.recommendation.explain_recommendations import explain_candidates
from src.recommendation.model_loader import load_models
from src.recommendation.scorer import score_candidates


def generate_recommendations(
    conn: sqlite3.Connection,
    models_dir: str = "data/models",
    weights: dict[str, float] | None = None,
    top_n_factors: int = 5,
) -> pd.DataFrame:
    df, feature_columns = build_dataset(conn)
    candidates = candidate_pool(df)

    if candidates.empty:
        raise ValueError(
            "No candidates in the backlog pool — every game in this library already has "
            "evidence of having been played. Nothing to recommend."
        )

    models = load_models(models_dir)
    ranked = score_candidates(candidates, feature_columns, models, weights=weights)
    explanations = explain_candidates(ranked, feature_columns, models, top_n=top_n_factors)
    ranked["explanation"] = explanations

    _write_recommendations(conn, ranked)
    return ranked


def _write_recommendations(conn: sqlite3.Connection, ranked: pd.DataFrame) -> None:
    now = dt.datetime.utcnow().isoformat(timespec="seconds")
    for _, row in ranked.iterrows():
        conn.execute(
            """
            INSERT INTO recommendations (
                igdb_id, predicted_rating, predicted_completion_prob,
                predicted_replay_prob, predicted_engagement,
                recommendation_score, explanation_json, generated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(igdb_id) DO UPDATE SET
                predicted_rating = excluded.predicted_rating,
                predicted_completion_prob = excluded.predicted_completion_prob,
                predicted_replay_prob = excluded.predicted_replay_prob,
                predicted_engagement = excluded.predicted_engagement,
                recommendation_score = excluded.recommendation_score,
                explanation_json = excluded.explanation_json,
                generated_at = excluded.generated_at
            """,
            (
                row["igdb_id"],
                row.get("predicted_rating"),
                row.get("predicted_completion_probability"),
                row.get("predicted_replay_probability"),
                row.get("predicted_engagement"),
                row["recommendation_score"],
                json.dumps(row["explanation"]),
                now,
            ),
        )
    conn.commit()
