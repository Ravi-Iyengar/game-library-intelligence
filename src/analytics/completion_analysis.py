"""
Completion analysis: "which genres/themes/developers are most likely
to be finished?" per the spec.

completion_rate uses the raw 'completed' status specifically.
positive_rate uses the broader signal_category (which also counts
'played', per the assumption documented in
feature_engineering/completion.py) — reported alongside so you can see
where the two diverge.
"""
from __future__ import annotations

from collections import defaultdict

from src.analytics.data_loader import GameRecord


def completion_rate_by_tag(
    games: list[GameRecord],
    category: str,
    min_games: int = 2,
) -> list[dict]:
    logged_counts: dict[str, int] = defaultdict(int)
    completed_counts: dict[str, int] = defaultdict(int)
    positive_counts: dict[str, int] = defaultdict(int)

    for g in games:
        if g["status"] is None:
            continue  # never logged as played at all — not part of a completion rate
        for tag in g["tags"].get(category, []):
            logged_counts[tag] += 1
            if g["status"] == "completed":
                completed_counts[tag] += 1
            if g["signal_category"] == "positive":
                positive_counts[tag] += 1

    rows = []
    for tag, n in logged_counts.items():
        if n < min_games:
            continue
        rows.append(
            {
                "tag": tag,
                "games_logged": n,
                "completion_rate": completed_counts[tag] / n,
                "positive_rate": positive_counts[tag] / n,
            }
        )
    rows.sort(key=lambda r: r["completion_rate"], reverse=True)
    return rows
