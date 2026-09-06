"""
Per-recommendation explanations. The spec's own example (Phase 8/9)
explains a recommendation via one blended list of top factors — e.g.
"+ Larian Studios, + CRPG, + Party-based RPG" — attributed to why the
game is predicted to be enjoyed, which is really the rating model's
question. So the explanation for each recommendation comes from
whichever of [rating, completion, replay, engagement] models is
available, in that priority order — rating first, since "why would you
like this" is the most natural read of the spec's example, falling
back to the next available model only if rating wasn't trained.
"""
from __future__ import annotations

import pandas as pd

from src.ml.explain import explain_predictions

EXPLANATION_PRIORITY = ("rating", "completion", "replay", "engagement")


def explain_candidates(
    candidates: pd.DataFrame,
    feature_columns: list[str],
    models: dict[str, object],
    top_n: int = 5,
) -> list[dict]:
    chosen_target = next((t for t in EXPLANATION_PRIORITY if models.get(t) is not None), None)
    if chosen_target is None:
        return [{"method": "unavailable", "source_model": None, "top_factors": []} for _ in range(len(candidates))]

    X = candidates[feature_columns]
    explanations = explain_predictions(models[chosen_target], X, top_n=top_n)
    return [{"source_model": chosen_target, **exp} for exp in explanations]
