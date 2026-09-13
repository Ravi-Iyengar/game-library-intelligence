"""
Explainability — mandatory per the spec: every trained model gets a
SHAP-based explanation of its predictions.

Degrades gracefully if the `shap` package isn't installed: falls back
to the model's own global feature_importances_ (or coef_ for linear
models, though none are used here) applied identically to every row,
clearly labeled as a fallback. This is strictly worse than real SHAP —
it can't say why *this* game scored the way it did, only which
features matter *overall* — but it's better than no explanation at all
while the dependency is missing.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _shap_available() -> bool:
    try:
        import shap  # noqa: F401

        return True
    except ImportError:
        return False


def explain_predictions(
    pipeline,
    X: pd.DataFrame,
    top_n: int = 5,
) -> list[dict]:
    """
    Returns one entry per row of X:
        {"top_factors": [{"feature": ..., "contribution": +/-0.xx}, ...],
         "method": "shap" | "global_feature_importance"}

    "contribution" is a signed SHAP value (this row, this feature) when
    shap is available, or the model's global importance (unsigned,
    same for every row) as a fallback.
    """
    model = pipeline.named_steps["model"]
    imputer = pipeline.named_steps["imputer"]
    # SimpleImputer silently drops any column that was entirely NaN during
    # fit (e.g. an affinity category with zero observed values in the
    # training set) — its transform() output can be narrower than the
    # input. get_feature_names_out() reports which columns actually
    # survived, so the reconstructed DataFrame's headers line up with the
    # real output width instead of the original input width.
    transformed = imputer.transform(X)
    surviving_columns = imputer.get_feature_names_out(X.columns)
    X_imputed = pd.DataFrame(transformed, columns=surviving_columns, index=X.index)

    if _shap_available():
        import shap

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_imputed)
            # Classifiers can return a list (one array per class); take the positive class.
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]

            results = []
            for row_values in shap_values:
                ranked = sorted(
                    zip(surviving_columns, row_values), key=lambda kv: abs(kv[1]), reverse=True
                )[:top_n]
                results.append(
                    {
                        "method": "shap",
                        "top_factors": [
                            {"feature": f, "contribution": float(v)} for f, v in ranked
                        ],
                    }
                )
            return results
        except Exception as e:  # noqa: BLE001 - SHAP has many model-specific edge cases
            logger.warning("SHAP explanation failed (%s) — falling back to global importance.", e)

    return _global_importance_fallback(model, surviving_columns, len(X), top_n)


def _global_importance_fallback(model, feature_names, n_rows: int, top_n: int) -> list[dict]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        logger.warning(
            "Model has no feature_importances_ and shap is unavailable — "
            "no explanation can be produced. Install shap for real explanations."
        )
        return [{"method": "unavailable", "top_factors": []} for _ in range(n_rows)]

    ranked = sorted(zip(feature_names, importances), key=lambda kv: kv[1], reverse=True)[:top_n]
    entry = {
        "method": "global_feature_importance",
        "top_factors": [{"feature": f, "contribution": float(v)} for f, v in ranked],
    }
    return [entry for _ in range(n_rows)]
