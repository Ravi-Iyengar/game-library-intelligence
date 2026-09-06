"""
Rating semantics.

In this export, a rating of 0 is Backloggd's "no rating given" default,
not a genuine 0/10 score — confirmed by checking: every rating=0 game
also has total_hours=0, including 61 games logged as "completed" (nobody
completes 61 games and rates all of them the worst possible score while
also logging zero hours on each). Treating raw 0 as a real rating would
corrupt every rating-weighted calculation downstream (tag affinities,
library averages, and eventually the Sprint 5 rating-prediction target).

If this assumption is wrong for your data — if you genuinely do rate
some games 0/10 — change EFFECTIVE_ZERO_IS_UNRATED to False here; every
caller reads through effective_rating() rather than the raw column.
"""
from __future__ import annotations

EFFECTIVE_ZERO_IS_UNRATED = True


def effective_rating(raw_rating: float | int | None) -> float | None:
    if raw_rating is None:
        return None
    if EFFECTIVE_ZERO_IS_UNRATED and raw_rating == 0:
        return None
    return raw_rating
