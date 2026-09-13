"""
Target construction for the four models. Every target is only defined
for rows where has_settled_outcome is True (has_been_played AND not
still is_backlog) — see dataset.py for why a game the user hasn't
beaten yet has a real but provisional outcome, not a final training
label.
"""
from __future__ import annotations

import pandas as pd


def rating_target(df: pd.DataFrame) -> pd.Series:
    """Regression target: the user's own rating. NaN where unrated, unplayed, or unsettled."""
    eligible = df["has_settled_outcome"] & df["own_rating"].notna()
    return df["own_rating"].where(eligible)


def completion_target(df: pd.DataFrame) -> pd.Series:
    """Classification target: was this specific playthrough history a completion?"""
    eligible = df["has_settled_outcome"]
    return (df["status_raw"] == "completed").astype("Int64").where(eligible)


def replay_target(df: pd.DataFrame) -> pd.Series:
    """Classification target: was the game replayed at least once?"""
    eligible = df["has_settled_outcome"]
    return (df["replay_count"].fillna(0) > 0).astype("Int64").where(eligible)


def engagement_score_target(df: pd.DataFrame) -> pd.Series:
    """
    Regression target: a composite 0-1 engagement score combining rating,
    hours, and replay behavior.

    The spec explicitly leaves this formula "determined experimentally" —
    this is a documented starting point, not a claimed-optimal weighting:

        engagement_score = 0.4 * rating_component
                          + 0.4 * hours_component
                          + 0.2 * replay_component

    rating_component: own_rating / 10, or 0.5 (neutral) if unrated —
        an unrated-but-played game shouldn't default to "no engagement."
    hours_component: log1p(hours) min-max scaled to [0, 1] across
        eligible rows — log-scaled so a 900-hour outlier (this library
        has one) doesn't squash everything else to near-zero.
    replay_component: min(replay_count, 3) / 3 — caps out at 3 replays
        rather than letting one heavily-replayed game dominate the scale.

    Adjust the weights/components here directly once you have a sense
    of whether this actually tracks how engaged you felt.
    """
    import numpy as np

    eligible = df["has_settled_outcome"]
    hours = df["own_total_hours"].fillna(0).clip(lower=0)
    log_hours = np.log1p(hours)

    eligible_log_hours = log_hours[eligible]
    lo, hi = eligible_log_hours.min(), eligible_log_hours.max()
    span = (hi - lo) if hi > lo else 1.0
    hours_component = (log_hours - lo) / span
    hours_component = hours_component.clip(lower=0, upper=1)

    rating_component = df["own_rating"].apply(lambda r: (r / 10.0) if pd.notna(r) else 0.5)
    replay_component = (df["replay_count"].fillna(0).clip(upper=3)) / 3.0

    score = 0.4 * rating_component + 0.4 * hours_component + 0.2 * replay_component
    return score.where(eligible)
