"""
Composite recommendation scoring, per the spec:

    score = 0.40 * predicted_rating
          + 0.25 * completion_probability
          + 0.20 * replay_probability
          + 0.15 * engagement_score

The spec's own inputs are mixed-scale (rating is 0-10, the other three
are already 0-1), so this normalizes predicted_rating to 0-1 (÷10)
before combining, then multiplies the final score by 100 for a
friendlier 0-100 display number — the relative weighting the spec
specifies is preserved either way.

If a model wasn't trained (missing file — see model_loader.py), its
term is dropped and the remaining weights are renormalized to sum to
1, rather than silently scoring it as 0 (which would understate every
candidate identically and just be noise).
"""
from __future__ import annotations

import pandas as pd

DEFAULT_WEIGHTS = {"rating": 0.40, "completion": 0.25, "replay": 0.20, "engagement": 0.15}

TASK_BY_TARGET = {
    "rating": "regression",
    "completion": "classification",
    "replay": "classification",
    "engagement": "regression",
}


def _predict_column(model, X: pd.DataFrame, task: str) -> pd.Series:
    if task == "classification":
        if hasattr(model, "predict_proba"):
            return pd.Series(model.predict_proba(X)[:, 1], index=X.index)
        return pd.Series(model.predict(X).astype(float), index=X.index)
    return pd.Series(model.predict(X), index=X.index)


def score_candidates(
    candidates: pd.DataFrame,
    feature_columns: list[str],
    models: dict[str, object],
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Returns `candidates` with added columns: predicted_rating,
    predicted_completion_probability, predicted_replay_probability,
    predicted_engagement, recommendation_score — sorted by
    recommendation_score descending.
    """
    weights = dict(weights or DEFAULT_WEIGHTS)
    X = candidates[feature_columns]

    result = candidates.copy()
    normalized_terms: dict[str, pd.Series] = {}

    for target, model in models.items():
        if model is None:
            weights.pop(target, None)
            continue
        preds = _predict_column(model, X, TASK_BY_TARGET[target])
        if target == "rating":
            result["predicted_rating"] = preds.clip(lower=0, upper=10)
            normalized_terms["rating"] = result["predicted_rating"] / 10.0
        elif target == "completion":
            result["predicted_completion_probability"] = preds.clip(lower=0, upper=1)
            normalized_terms["completion"] = result["predicted_completion_probability"]
        elif target == "replay":
            result["predicted_replay_probability"] = preds.clip(lower=0, upper=1)
            normalized_terms["replay"] = result["predicted_replay_probability"]
        elif target == "engagement":
            result["predicted_engagement"] = preds.clip(lower=0, upper=1)
            normalized_terms["engagement"] = result["predicted_engagement"]

    if not weights:
        raise ValueError(
            "No trained models available to score candidates — run scripts/run_ml_training.py first."
        )

    weight_sum = sum(weights.values())
    composite = sum(
        (weights[t] / weight_sum) * normalized_terms[t] for t in weights if t in normalized_terms
    )
    result["recommendation_score"] = (composite * 100).round(2)

    return result.sort_values("recommendation_score", ascending=False).reset_index(drop=True)
