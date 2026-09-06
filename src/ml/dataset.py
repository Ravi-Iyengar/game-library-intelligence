"""
Sprint 5 dataset construction.

Three things this module has to get right, because getting them wrong
would quietly produce a model that looks fine in testing and is useless
in production:

1. "Has this game actually been played?" The status label alone can't
   be trusted — 42 games in this library are labeled completed/played
   but have zero hours, zero playthroughs, and no real rating. Those
   are bulk-import placeholders, not real outcomes. has_been_played()
   below requires actual evidence (logged hours, a playthrough record,
   or a genuine rating), not just a status string.

2. "Is that outcome actually settled, or still in progress?" Confirmed
   directly with the library's owner: is_backlog=True means "haven't
   beaten/finished this yet" — and 72 games in this library have real
   logged hours (sometimes a lot — Red Dead Redemption 2 at 45 hours)
   while still being flagged is_backlog=True. Their current rating/
   status isn't a final verdict; the user fully intends to keep playing
   and may rate it differently once actually done. has_settled_outcome()
   requires has_been_played() AND is_backlog=False — this is what Sprint
   5 trains on, and its negation (plus wishlist exclusion) is exactly
   Sprint 6's recommendation pool: everything not yet beaten, whether
   untouched or partway through.

3. Every model trained here will eventually score BACKLOG games
   (Sprint 6) that haven't been finished yet. Any feature that only
   exists because a game *was* played — hours, sessions, reviews,
   review embeddings, replay counts — is unavailable or provisional at
   that point. Training on those features would make a model that's
   accurate on settled games and meaningless on the backlog it's
   actually for. PRE_PLAY_FEATURE_COLUMNS is deliberately restricted to
   what's true of any game regardless of play status: its tags, its tag
   affinities (computed from the user's taste on *other* games), and
   IGDB's own aggregate critic rating.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3

import pandas as pd

from src.utils.ratings import effective_rating

ONE_HOT_CATEGORIES = ("genre_one_hot", "theme_one_hot", "developer_one_hot", "publisher_one_hot")
AFFINITY_COLUMNS = (
    "genre_affinity", "theme_affinity", "developer_affinity",
    "publisher_affinity", "franchise_affinity",
)
IGDB_METADATA_COLUMNS = ("aggregated_rating", "aggregated_rating_count", "release_year")

# Populated at import time isn't possible (depends on the one-hot vocab
# actually present in the data), so PRE_PLAY_FEATURE_COLUMNS is computed
# per-dataset by build_dataset() and returned alongside the DataFrame.


def _flatten_feature_dict(feature_dict: dict) -> dict:
    flat = {}
    for key, value in feature_dict.items():
        if key in ONE_HOT_CATEGORIES:
            prefix = key.removesuffix("_one_hot")
            for tag, present in (value or {}).items():
                flat[f"{prefix}__{tag}"] = present
        elif key == "review_embedding":
            continue  # handled separately if/when present — see build_dataset
        else:
            flat[key] = value
    return flat


def _release_year(release_date: str | None) -> float | None:
    if not release_date:
        return None
    try:
        return float(dt.date.fromisoformat(release_date[:10]).year)
    except ValueError:
        return None


def has_been_played(row: pd.Series) -> bool:
    """
    Real evidence of engagement, not just a status label. A game counts
    as "played" if it has logged hours, at least one playthrough record,
    or a genuine (non-zero-default) rating.
    """
    hours = row.get("own_total_hours") or 0
    playthroughs = row.get("playthrough_count") or 0
    rating = row.get("own_rating")
    return bool(hours > 0 or playthroughs > 0 or pd.notna(rating))


def has_settled_outcome(row: pd.Series) -> bool:
    """
    has_been_played() AND not still-in-progress (is_backlog=False).
    A game the user has started but hasn't beaten yet has a real but
    provisional rating/hours — not the final signal Sprint 5 should
    train on. See point 2 in the module docstring.
    """
    is_backlog = bool(row.get("is_backlog") or 0)
    return has_been_played(row) and not is_backlog


def build_dataset(conn: sqlite3.Connection) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns (df, pre_play_feature_columns).

    df has one row per game: every flattened feature-store field, plus
    igdb_id/title, IGDB metadata (aggregated_rating/_count, release_year),
    a has_been_played boolean, and a has_settled_outcome boolean.
    pre_play_feature_columns is the list of columns safe to use as model
    *input* for any of the four targets (see module docstring for why
    this is restricted).
    """
    rows = conn.execute(
        """
        SELECT f.igdb_id, g.title, f.feature_json,
               g.aggregated_rating, g.aggregated_rating_count, g.release_date
        FROM features f
        JOIN games g ON g.igdb_id = f.igdb_id
        """
    ).fetchall()

    records = []
    for row in rows:
        flat = _flatten_feature_dict(json.loads(row["feature_json"]))
        flat["igdb_id"] = row["igdb_id"]
        flat["title"] = row["title"]
        flat["aggregated_rating"] = row["aggregated_rating"]
        flat["aggregated_rating_count"] = row["aggregated_rating_count"]
        flat["release_year"] = _release_year(row["release_date"])
        records.append(flat)

    df = pd.DataFrame.from_records(records)

    # own_rating already passed through effective_rating() in the feature
    # store (Sprint 3), but this guards the dataset against being built
    # from a features table populated before that fix was applied.
    if "own_rating" in df.columns:
        df["own_rating"] = df["own_rating"].apply(effective_rating)

    one_hot_cols = [
        c for c in df.columns
        if any(c.startswith(f"{cat.removesuffix('_one_hot')}__") for cat in ONE_HOT_CATEGORIES)
    ]
    pre_play_feature_columns = one_hot_cols + [
        c for c in AFFINITY_COLUMNS + IGDB_METADATA_COLUMNS if c in df.columns
    ]

    df["has_been_played"] = df.apply(has_been_played, axis=1)
    df["has_settled_outcome"] = df.apply(has_settled_outcome, axis=1)

    # One-hot columns are 0/1 already; affinity/metadata columns can be
    # legitimately missing (no other game shares the tag; not IGDB-rated
    # externally) — leave as NaN here, let each model's pipeline decide
    # how to impute, rather than silently filling in this shared loader.

    return df, pre_play_feature_columns
