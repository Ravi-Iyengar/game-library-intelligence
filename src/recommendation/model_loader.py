"""Loads the joblib models Sprint 5 saved. Missing files are tolerated
(a target that couldn't be trained yet, e.g. no enrichment data) —
the scorer redistributes weight away from whatever's missing rather
than failing outright."""
from __future__ import annotations

import logging
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

TARGET_NAMES = ("rating", "completion", "replay", "engagement")


def load_models(models_dir: str | Path) -> dict[str, object]:
    models_dir = Path(models_dir)
    models: dict[str, object] = {}
    for name in TARGET_NAMES:
        path = models_dir / f"{name}_model.joblib"
        if path.exists():
            models[name] = joblib.load(path)
        else:
            logger.warning(
                "No saved model for target '%s' at %s — recommendations will be scored "
                "without it (weight redistributed across the remaining models). Run "
                "scripts/run_ml_training.py first to produce it.",
                name, path,
            )
            models[name] = None
    return models
