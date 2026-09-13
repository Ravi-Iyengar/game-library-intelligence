"""
Sprint 5 — model training.

For each of the four targets (rating, completion, replay, engagement):
  1. Restrict to rows with has_been_played (real evidence, not just a
     status label — see dataset.py) and a defined target value.
  2. 80/20 holdout split (stratified for classification).
  3. Within the 80% training portion, 5-fold cross-validation for every
     candidate model (model_candidates.py), to pick the best one — not
     just to report a number.
  4. Refit the winning candidate on the full 80% training portion,
     evaluate once on the untouched 20% holdout, then refit again on
     ALL eligible rows (train+holdout) before saving — the holdout's
     job is model *selection* and an honest final metric, not to be
     permanently withheld from the model that ships.

Missing values in pre-play feature columns (e.g. no other game shares
a given tag, so its affinity is undefined) are median-imputed inside
each model's own pipeline, not in the shared dataset loader — so the
imputation is fit only on that target's training fold, never leaking
statistics from the holdout.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.ml.evaluation import classification_metrics, regression_metrics
from src.ml.model_candidates import get_candidates

logger = logging.getLogger(__name__)

REGRESSION_TARGETS = ("rating", "engagement")
CLASSIFICATION_TARGETS = ("completion", "replay")


def _make_pipeline(estimator) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])


def _cross_validate_candidate(pipeline_template, X, y, task: str) -> dict[str, float]:
    if task == "regression":
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_metrics: list[dict[str, float]] = []
    for train_idx, val_idx in splitter.split(X, y):
        pipeline = clone(pipeline_template)
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        if task == "regression":
            preds = pipeline.predict(X.iloc[val_idx])
            fold_metrics.append(regression_metrics(y.iloc[val_idx], preds))
        else:
            preds = pipeline.predict(X.iloc[val_idx])
            proba = (
                pipeline.predict_proba(X.iloc[val_idx])[:, 1]
                if hasattr(pipeline, "predict_proba")
                else None
            )
            fold_metrics.append(classification_metrics(y.iloc[val_idx], preds, proba))

    keys = fold_metrics[0].keys()
    return {
        k: float(np.mean([m[k] for m in fold_metrics if m[k] is not None]))
        if any(m[k] is not None for m in fold_metrics)
        else None
        for k in keys
    }


def train_target(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    target_name: str,
) -> dict:
    """
    Returns:
        {
            "target": target_name,
            "n_eligible": int,
            "candidates": {name: {"cv_metrics": {...}}},
            "best_candidate": name,
            "holdout_metrics": {...},
            "model": fitted Pipeline (refit on ALL eligible data),
        }
    """
    mask = y.notna()
    X_eligible, y_eligible = X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)
    n = len(y_eligible)

    usable_columns = [c for c in X_eligible.columns if X_eligible[c].notna().any()]
    if not usable_columns:
        logger.warning(
            "No usable feature columns for target '%s' (every pre-play feature is entirely "
            "missing across %d eligible rows). This usually means Sprint 2 (IGDB enrichment) "
            "hasn't run yet, so there are no tags/affinities/IGDB ratings to train on. "
            "Skipping training for this target.",
            target_name, n,
        )
        return {"target": target_name, "n_eligible": n, "candidates": {}, "best_candidate": None,
                "holdout_metrics": {}, "model": None}

    if n < 20:
        logger.warning(
            "Only %d eligible rows for target '%s' — too few for a meaningful "
            "80/20 split and 5-fold CV. Skipping training for this target.",
            n, target_name,
        )
        return {"target": target_name, "n_eligible": n, "candidates": {}, "best_candidate": None,
                "holdout_metrics": {}, "model": None}

    stratify = y_eligible if task == "classification" else None
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_eligible, y_eligible, test_size=0.2, random_state=42, stratify=stratify
    )

    candidates = get_candidates(task)
    cv_results: dict[str, dict] = {}
    for name, estimator in candidates.items():
        pipeline = _make_pipeline(estimator)
        cv_metrics = _cross_validate_candidate(pipeline, X_train, y_train, task)
        cv_results[name] = {"cv_metrics": cv_metrics}
        logger.info("Target '%s' candidate '%s' CV metrics: %s", target_name, name, cv_metrics)

    primary_metric = "rmse" if task == "regression" else "roc_auc"
    lower_is_better = task == "regression"

    def _score(name):
        val = cv_results[name]["cv_metrics"].get(primary_metric)
        if val is None:
            val = cv_results[name]["cv_metrics"].get("f1", 0)
            return val  # higher is better fallback
        return val

    best_name = min(cv_results, key=_score) if lower_is_better else max(cv_results, key=_score)

    best_pipeline = _make_pipeline(clone(candidates[best_name]))
    best_pipeline.fit(X_train, y_train)
    holdout_preds = best_pipeline.predict(X_holdout)
    if task == "regression":
        holdout_metrics = regression_metrics(y_holdout, holdout_preds)
    else:
        holdout_proba = (
            best_pipeline.predict_proba(X_holdout)[:, 1]
            if hasattr(best_pipeline, "predict_proba")
            else None
        )
        holdout_metrics = classification_metrics(y_holdout, holdout_preds, holdout_proba)

    # Final model for deployment: refit the winning candidate on ALL
    # eligible data (train + holdout), now that holdout has done its job
    # of giving an honest, unbiased metric.
    final_pipeline = _make_pipeline(clone(candidates[best_name]))
    final_pipeline.fit(X_eligible, y_eligible)

    return {
        "target": target_name,
        "n_eligible": n,
        "candidates": cv_results,
        "best_candidate": best_name,
        "holdout_metrics": holdout_metrics,
        "model": final_pipeline,
    }
