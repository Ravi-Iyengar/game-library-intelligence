"""
Sprint 5 orchestrator. Builds the dataset once, trains all four
targets, saves each winning model to disk, and returns a summary
report (also used by scripts/run_ml_training.py for the CLI output).

Usage:

    from src.storage.db import get_connection
    from src.ml.run_training import train_all_models

    conn = get_connection("data/processed/glip.db")
    report = train_all_models(conn, models_dir="data/models")
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
from pathlib import Path

import joblib

from src.ml.dataset import build_dataset
from src.ml.explain import explain_predictions
from src.ml.targets import (
    completion_target,
    engagement_score_target,
    rating_target,
    replay_target,
)
from src.ml.train import train_target

logger = logging.getLogger(__name__)

TARGET_BUILDERS = {
    "rating": (rating_target, "regression"),
    "completion": (completion_target, "classification"),
    "replay": (replay_target, "classification"),
    "engagement": (engagement_score_target, "regression"),
}


def train_all_models(
    conn: sqlite3.Connection,
    models_dir: str | Path = "data/models",
    explain_top_n: int = 5,
) -> dict:
    df, feature_columns = build_dataset(conn)
    X = df[feature_columns]

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "generated_at": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "n_games_total": len(df),
        "n_games_played": int(df["has_been_played"].sum()),
        "n_games_settled": int(df["has_settled_outcome"].sum()),  # what training actually used
        "feature_columns": feature_columns,
        "targets": {},
    }

    for target_name, (builder, task) in TARGET_BUILDERS.items():
        y = builder(df)
        result = train_target(X, y, task, target_name)

        target_report = {
            "n_eligible": result["n_eligible"],
            "candidates": result["candidates"],
            "best_candidate": result["best_candidate"],
            "holdout_metrics": result["holdout_metrics"],
        }

        if result["model"] is not None:
            model_path = models_dir / f"{target_name}_model.joblib"
            joblib.dump(result["model"], model_path)
            target_report["model_path"] = str(model_path)

            # A small, illustrative explanation sample (not the whole
            # library) — drawn from the same settled population training
            # used, not the broader has_been_played set (which now also
            # includes still-in-progress games training deliberately
            # excludes). Full per-game explanations belong to Sprint 6,
            # where they're attached to actual recommendations.
            sample = X.loc[df["has_settled_outcome"]].head(explain_top_n)
            if len(sample) > 0:
                target_report["example_explanations"] = [
                    {"igdb_id": df.loc[i, "igdb_id"], "title": df.loc[i, "title"], **exp}
                    for i, exp in zip(sample.index, explain_predictions(result["model"], sample))
                ]

        report["targets"][target_name] = target_report

    return report


def save_report(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))
