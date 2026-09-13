"""
Tag preference tables: for each genre/theme/developer/publisher/
franchise, how much the user rates and plays games carrying that tag.
Per the spec's "Genre Preferences ... Weighted by Hours, Rating" and
the equivalent Developer/Franchise Analysis sections.

This reports on the *whole* library (no leave-one-out) — it's meant for
the dashboard/analytics story, not as an ML feature. Contrast with
feature_engineering/affinity.py, which is leave-one-out on purpose
because it feeds a model that predicts the very rating being averaged.
"""
from __future__ import annotations

from collections import defaultdict

from src.analytics.data_loader import GameRecord

# A rated-but-untimed game still counts, at a nominal weight, rather
# than being invisible to the weighted average — same convention as
# feature_engineering/affinity.py.
NOMINAL_WEIGHT_HOURS = 1.0


def _weight(total_hours: float | None) -> float:
    return total_hours if total_hours and total_hours > 0 else NOMINAL_WEIGHT_HOURS


def tag_preference_table(
    games: list[GameRecord],
    category: str,
    min_games: int = 2,
) -> list[dict]:
    """
    Returns rows sorted by hours-weighted average rating (descending):
        {tag, game_count, total_hours, avg_rating, hours_weighted_rating}

    avg_rating is a plain mean of ratings among rated games with the tag.
    hours_weighted_rating additionally weights each game's rating by its
    hours (or NOMINAL_WEIGHT_HOURS if unlogged). Tags on fewer than
    min_games games are dropped — a preference computed from one data
    point isn't a preference yet.
    """
    game_counts: dict[str, int] = defaultdict(int)
    total_hours_by_tag: dict[str, float] = defaultdict(float)
    ratings_by_tag: dict[str, list[float]] = defaultdict(list)
    weighted_sum_by_tag: dict[str, float] = defaultdict(float)
    weight_sum_by_tag: dict[str, float] = defaultdict(float)

    for g in games:
        for tag in g["tags"].get(category, []):
            game_counts[tag] += 1
            total_hours_by_tag[tag] += g["total_hours"] or 0
            if g["rating"] is not None:
                ratings_by_tag[tag].append(g["rating"])
                w = _weight(g["total_hours"])
                weighted_sum_by_tag[tag] += g["rating"] * w
                weight_sum_by_tag[tag] += w

    rows = []
    for tag, count in game_counts.items():
        if count < min_games:
            continue
        ratings = ratings_by_tag.get(tag, [])
        hours_weighted_rating = (
            weighted_sum_by_tag[tag] / weight_sum_by_tag[tag] if weight_sum_by_tag[tag] else None
        )
        rows.append(
            {
                "tag": tag,
                "game_count": count,
                "total_hours": total_hours_by_tag[tag],
                "avg_rating": (sum(ratings) / len(ratings)) if ratings else None,
                "hours_weighted_rating": hours_weighted_rating,
            }
        )

    rows.sort(key=lambda r: (r["hours_weighted_rating"] is None, -(r["hours_weighted_rating"] or 0)))
    return rows
