"""
The candidate pool for recommendations: games eligible to be ranked —
everything not yet beaten, per the library owner's own definition:
untouched games AND games started but not finished (is_backlog=True
with real logged hours). This is deliberately the negation of Sprint
5's training filter (has_settled_outcome — see dataset.py): a game
either has a settled outcome (safe to learn from, not to recommend
again) or it's still open business (safe to recommend, not yet safe to
learn a final label from). A game can't be both.

Wishlist games (is_wishlist=1) are excluded too, per spec: wishlist is
a "future acquisition pool," not part of recommendation evaluation —
though in this export is_wishlist is False for every game, so this is
a defensive filter for whenever that's no longer true.
"""
from __future__ import annotations

import pandas as pd


def candidate_pool(df: pd.DataFrame) -> pd.DataFrame:
    not_settled = ~df["has_settled_outcome"]
    not_wishlist = df.get("is_wishlist", 0).fillna(0).astype(int) == 0
    return df.loc[not_settled & not_wishlist].reset_index(drop=True)
