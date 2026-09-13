"""
Review text features: which review represents a game (it can have
several, one per playthrough), and basic length/existence stats on it.
Embeddings live in embeddings.py — kept separate so the cheap text
stats still work even when the embedding model isn't available.
"""
from __future__ import annotations

from typing import TypedDict


class PlaythroughReview(TypedDict, total=False):
    is_master: int | None
    updated_at: str | None
    review: str | None


def select_representative_review(playthroughs: list[PlaythroughReview]) -> str | None:
    """
    Pick one review to represent the game. Preference order:
      1. The master playthrough's review, if it has one.
      2. Otherwise, the most recently updated playthrough that has one.
      3. Otherwise, None (no review has been written for this game).
    """
    master_reviews = [
        pt["review"] for pt in playthroughs if pt.get("is_master") and pt.get("review")
    ]
    if master_reviews:
        return master_reviews[0]

    reviewed = [pt for pt in playthroughs if pt.get("review")]
    if not reviewed:
        return None

    reviewed.sort(key=lambda pt: pt.get("updated_at") or "", reverse=True)
    return reviewed[0]["review"]


def extract_review_text_features(review: str | None) -> dict[str, int]:
    if not review or not review.strip():
        return {"review_exists": 0, "review_length": 0, "review_word_count": 0}
    return {
        "review_exists": 1,
        "review_length": len(review),
        "review_word_count": len(review.split()),
    }
