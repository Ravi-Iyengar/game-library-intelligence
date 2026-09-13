"""Metrics, per the spec: MAE/RMSE/R2 for regression, ROC-AUC/F1 for classification."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def classification_metrics(y_true, y_pred, y_proba=None) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {"f1": f1_score(y_true, y_pred, zero_division=0)}
    if y_proba is not None and len(set(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["roc_auc"] = None  # undefined with a single class present, e.g. a tiny CV fold
    return metrics
