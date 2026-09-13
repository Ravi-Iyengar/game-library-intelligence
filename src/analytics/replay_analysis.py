"""Replay analysis: which tags show up most on replayed games."""
from __future__ import annotations

from collections import defaultdict

from src.analytics.data_loader import GameRecord


def most_replayed(games: list[GameRecord], category: str) -> list[dict]:
    """
    Returns rows sorted by total_replays (descending):
        {tag, total_replays, games_with_replay, games_total}

    total_replays sums each game's replay_count (playthroughs beyond
    the first) across every game carrying the tag. games_with_replay
    counts distinct games (not playthroughs) that were replayed at
    least once, so a single heavily-replayed game can't be mistaken
    for a broad pattern across many games.
    """
    total_replays: dict[str, int] = defaultdict(int)
    games_with_replay: dict[str, int] = defaultdict(int)
    games_total: dict[str, int] = defaultdict(int)

    for g in games:
        for tag in g["tags"].get(category, []):
            games_total[tag] += 1
            total_replays[tag] += g["replay_count"]
            if g["replay_count"] > 0:
                games_with_replay[tag] += 1

    rows = [
        {
            "tag": tag,
            "total_replays": total_replays[tag],
            "games_with_replay": games_with_replay[tag],
            "games_total": games_total[tag],
        }
        for tag in games_total
        if total_replays[tag] > 0
    ]
    rows.sort(key=lambda r: r["total_replays"], reverse=True)
    return rows
