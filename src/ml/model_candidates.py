"""
Model candidates, per the spec: RandomForest, XGBoost, CatBoost,
LightGBM — no neural networks, not enough data for them (this library
has ~400 games; gradient-boosted trees are the right tool at this scale).

Every candidate but RandomForest is an optional dependency. If a
package isn't installed, that candidate is silently skipped from the
comparison (logged once) rather than failing the whole training run —
the same graceful-degradation pattern as embeddings.py in Sprint 3.
"""
from __future__ import annotations

import logging

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def get_candidates(task: str) -> dict[str, object]:
    """task is 'regression' or 'classification'. Returns {name: unfitted estimator}."""
    if task not in ("regression", "classification"):
        raise ValueError("task must be 'regression' or 'classification'")

    candidates: dict[str, object] = {}

    if task == "regression":
        candidates["RandomForest"] = RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        )
    else:
        candidates["RandomForest"] = RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
        )

    try:
        import xgboost as xgb

        candidates["XGBoost"] = (
            xgb.XGBRegressor(n_estimators=300, random_state=RANDOM_STATE)
            if task == "regression"
            else xgb.XGBClassifier(n_estimators=300, random_state=RANDOM_STATE, eval_metric="logloss")
        )
    except ImportError:
        logger.info("xgboost not installed — skipping from candidate comparison.")

    try:
        import catboost as cb

        candidates["CatBoost"] = (
            cb.CatBoostRegressor(iterations=300, random_state=RANDOM_STATE, verbose=False)
            if task == "regression"
            else cb.CatBoostClassifier(iterations=300, random_state=RANDOM_STATE, verbose=False)
        )
    except ImportError:
        logger.info("catboost not installed — skipping from candidate comparison.")

    try:
        import lightgbm as lgb

        candidates["LightGBM"] = (
            lgb.LGBMRegressor(n_estimators=300, random_state=RANDOM_STATE, verbose=-1)
            if task == "regression"
            else lgb.LGBMClassifier(n_estimators=300, random_state=RANDOM_STATE, verbose=-1)
        )
    except ImportError:
        logger.info("lightgbm not installed — skipping from candidate comparison.")

    return candidates
