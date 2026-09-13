"""
Tag vocabulary for one-hot encoding.

Engineering Risk 3 from the project's own risk log: IGDB returns many
genre/theme/keyword tags, most of which occur on only one or two games
in a ~400-game library and would just be noise columns. Mitigation
(per that risk log): prune to tags occurring in at least
`min_occurrences` games before one-hot encoding.

This module only builds the vocabulary (which tags qualify); it doesn't
do the encoding itself — see affinity.py and build_features.py for that.
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter

DEFAULT_MIN_OCCURRENCES = 5

# Columns in `games` that hold a JSON array of tag names.
TAG_COLUMNS = ("genres", "themes", "developers", "publishers")


def _tag_list(raw_json: str | None) -> list[str]:
    if not raw_json:
        return []
    try:
        return json.loads(raw_json) or []
    except json.JSONDecodeError:
        return []


def build_vocabulary(
    conn: sqlite3.Connection,
    column: str,
    min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
) -> list[str]:
    """
    Return the sorted list of tag values in `column` (one of TAG_COLUMNS)
    that appear on at least `min_occurrences` games. Only enriched games
    (enriched_at IS NOT NULL) are considered, since un-enriched rows have
    no tags yet.
    """
    if column not in TAG_COLUMNS:
        raise ValueError(f"column must be one of {TAG_COLUMNS}, got {column!r}")

    counts: Counter[str] = Counter()
    rows = conn.execute(f"SELECT {column} FROM games WHERE enriched_at IS NOT NULL")
    for (raw_json,) in rows:
        counts.update(_tag_list(raw_json))

    return sorted(tag for tag, n in counts.items() if n >= min_occurrences)


def build_all_vocabularies(
    conn: sqlite3.Connection,
    min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
) -> dict[str, list[str]]:
    """Convenience wrapper: vocabulary for every TAG_COLUMNS entry at once."""
    return {col: build_vocabulary(conn, col, min_occurrences) for col in TAG_COLUMNS}


def one_hot(tags: list[str], vocabulary: list[str]) -> dict[str, int]:
    """Encode a game's own tag list against a fixed vocabulary."""
    tag_set = set(tags)
    return {tag: int(tag in tag_set) for tag in vocabulary}
