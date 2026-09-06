"""
Tag affinity scoring.

For each tag a game carries (a genre, a developer, ...), this computes
how much the user tends to like *other* games that share that tag —
a leave-one-out, hours-weighted average rating. A game's own rating is
excluded from its own affinity score on purpose: this feature is meant
to feed the rating-prediction model later (Sprint 5), and including a
game's own rating in a feature that predicts that same rating would be
target leakage.

Affinity categories: genres, themes, developers, publishers, franchises.
The first four are also one-hot encoded (vocabulary.py); franchises are
not (usually near-unique per game, one-hot would just be noise) but
still get an affinity score, matching the spec's franchise_affinity.
"""
from __future__ import annotations

from collections import defaultdict
from typing import TypedDict

AFFINITY_CATEGORIES = ("genres", "themes", "developers", "publishers", "franchises")

# A rating with no logged hours still counts, but at a nominal weight
# rather than zero — a rated-but-untimed game shouldn't be invisible to
# the affinity calculation.
NOMINAL_WEIGHT_HOURS = 1.0


class GameForAffinity(TypedDict):
    igdb_id: str
    rating: float | None
    total_hours: float | None
    tags: dict[str, list[str]]  # one list per AFFINITY_CATEGORIES key


def _weight(total_hours: float | None) -> float:
    return total_hours if total_hours and total_hours > 0 else NOMINAL_WEIGHT_HOURS


def compute_tag_affinities(
    games: list[GameForAffinity],
) -> dict[str, dict[str, float | None]]:
    """
    Returns {igdb_id: {"genre_affinity": ..., "theme_affinity": ..., ...}}.
    A value is None when the game has no tags in that category, or no
    *other* rated game shares any of them.
    """
    # tag -> list of (igdb_id, rating, weight) contributed by rated games
    contributions: dict[str, dict[str, list[tuple[str, float, float]]]] = {
        cat: defaultdict(list) for cat in AFFINITY_CATEGORIES
    }

    for game in games:
        if game["rating"] is None:
            continue
        weight = _weight(game["total_hours"])
        for cat in AFFINITY_CATEGORIES:
            for tag in game["tags"].get(cat, []):
                contributions[cat][tag].append((game["igdb_id"], game["rating"], weight))

    results: dict[str, dict[str, float | None]] = {}

    for game in games:
        scores: dict[str, float | None] = {}
        for cat in AFFINITY_CATEGORIES:
            tag_scores: list[float] = []
            for tag in game["tags"].get(cat, []):
                others = [
                    (rating, weight)
                    for (igdb_id, rating, weight) in contributions[cat].get(tag, [])
                    if igdb_id != game["igdb_id"]
                ]
                if not others:
                    continue
                total_weight = sum(w for _, w in others)
                tag_scores.append(sum(r * w for r, w in others) / total_weight)

            feature_name = _singular_affinity_name(cat)
            scores[feature_name] = sum(tag_scores) / len(tag_scores) if tag_scores else None

        results[game["igdb_id"]] = scores

    return results


def _singular_affinity_name(category: str) -> str:
    singular = {
        "genres": "genre",
        "themes": "theme",
        "developers": "developer",
        "publishers": "publisher",
        "franchises": "franchise",
    }[category]
    return f"{singular}_affinity"
