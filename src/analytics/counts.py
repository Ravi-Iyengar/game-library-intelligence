"""Library-level counts (Phase.txt Analytics Sprint 1: 'Game Counts')."""
from __future__ import annotations

from collections import Counter

from src.analytics.data_loader import GameRecord


def status_counts(games: list[GameRecord]) -> dict[str, int]:
    """Raw status counts, e.g. {'completed': 199, 'played': 88, ...}."""
    return dict(Counter(g["status"] or "unset" for g in games))


def signal_category_counts(games: list[GameRecord]) -> dict[str, int]:
    """positive/neutral/negative/unknown counts — see feature_engineering/completion.py."""
    return dict(Counter(g["signal_category"] for g in games))


def library_overview(games: list[GameRecord]) -> dict:
    total_hours = sum(g["total_hours"] or 0 for g in games)
    rated = [g["rating"] for g in games if g["rating"] is not None]
    return {
        "total_games": len(games),
        "total_hours_logged": total_hours,
        "rated_games": len(rated),
        "average_rating": (sum(rated) / len(rated)) if rated else None,
        "status_counts": status_counts(games),
        "signal_category_counts": signal_category_counts(games),
    }
