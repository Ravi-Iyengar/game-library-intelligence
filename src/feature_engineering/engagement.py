"""
Engagement metrics: playthrough_count, replay_count, session_count,
average_session_length, days_played, avg_days_between_replays.

Pure functions over already-fetched rows (no DB access here) so this
is easy to unit test — build_features.py does the fetching.
"""
from __future__ import annotations

import datetime as dt
from statistics import mean
from typing import TypedDict


class PlaythroughRow(TypedDict, total=False):
    playthrough_id: int
    is_replay: int | None
    start_date: str | None


class SessionRow(TypedDict, total=False):
    playthrough_id: int
    range_start_date: str | None
    session_start_date: str | None
    hours: float | None
    minutes: float | None


def _parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError:
        return None


def compute_engagement_metrics(
    playthroughs: list[PlaythroughRow],
    sessions: list[SessionRow],
) -> dict[str, float | int | None]:
    playthrough_count = len(playthroughs)
    replay_count = sum(1 for pt in playthroughs if pt.get("is_replay"))
    session_count = len(sessions)

    session_lengths = [
        (s.get("hours") or 0) + (s.get("minutes") or 0) / 60.0 for s in sessions
    ]
    average_session_length_hours = mean(session_lengths) if session_lengths else None

    day_values = {
        s.get("range_start_date") or s.get("session_start_date")
        for s in sessions
        if s.get("range_start_date") or s.get("session_start_date")
    }
    days_played = len(day_values)

    starts = sorted(
        d for d in (_parse_date(pt.get("start_date")) for pt in playthroughs) if d is not None
    )
    if len(starts) >= 2:
        gaps = [(b - a).days for a, b in zip(starts, starts[1:])]
        avg_days_between_replays = sum(gaps) / len(gaps)
    else:
        avg_days_between_replays = None

    return {
        "playthrough_count": playthrough_count,
        "replay_count": replay_count,
        "session_count": session_count,
        "average_session_length_hours": average_session_length_hours,
        "days_played": days_played,
        "avg_days_between_replays": avg_days_between_replays,
    }
