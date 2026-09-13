"""
Review embeddings via sentence-transformers (model: all-mpnet-base-v2),
per the project spec.

This degrades gracefully rather than hard-failing the whole feature
store build: if the package isn't installed, or the model can't be
downloaded (no network), get_embedder() returns None once, logs a
single clear warning, and every review_embedding feature is left null.
Re-running build_features.py later (once the dependency/network issue
is fixed) will backfill embeddings for rows that are missing them.
"""
from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

MODEL_NAME = "all-mpnet-base-v2"


@lru_cache(maxsize=1)
def get_embedder():
    """Returns a loaded SentenceTransformer, or None if unavailable. Cached — loads at most once."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning(
            "sentence-transformers is not installed — review_embedding will be null "
            "for all games. Install it (see requirements/requirements.txt) and re-run "
            "the feature store build to backfill embeddings."
        )
        return None

    try:
        return SentenceTransformer(MODEL_NAME)
    except Exception as e:  # noqa: BLE001 - genuinely want to catch anything (network, disk, etc.)
        logger.warning(
            "Could not load sentence-transformers model '%s' (%s) — "
            "review_embedding will be null for all games until this is resolved.",
            MODEL_NAME, e,
        )
        return None


def embed_reviews(reviews: list[str | None]) -> list[list[float] | None]:
    """
    Embed a batch of review texts. None entries (no review) stay None
    in the output, in the same positions, without going through the model.
    """
    embedder = get_embedder()
    if embedder is None:
        return [None] * len(reviews)

    indices = [i for i, r in enumerate(reviews) if r]
    if not indices:
        return [None] * len(reviews)

    texts = [reviews[i] for i in indices]
    vectors = embedder.encode(texts, show_progress_bar=False)

    result: list[list[float] | None] = [None] * len(reviews)
    for i, vector in zip(indices, vectors):
        result[i] = vector.tolist()
    return result
