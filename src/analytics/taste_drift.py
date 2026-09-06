"""
Genre/theme evolution and taste drift ("Has preference shifted from
exploration towards narrative?" per the spec).

Each playthrough's hours are attributed to the year of its start_date
and to every tag its game carries (a multi-genre game contributes its
full hours to each of its genres — the same one-hot convention used
elsewhere, not hours divided across tags). A year's "preference vector"
is its hours total per tag, over a shared vocabulary; cosine similarity
between two years' vectors measures how alike those two years' taste
was, independent of how many total hours were logged in each.
"""
from __future__ import annotations

import math
from collections import defaultdict

from src.analytics.data_loader import GameRecord, PlaythroughYearRecord


def yearly_tag_hours(
    games: list[GameRecord],
    playthrough_years: list[PlaythroughYearRecord],
    category: str = "genres",
) -> dict[int, dict[str, float]]:
    tags_by_game = {g["igdb_id"]: g["tags"].get(category, []) for g in games}

    result: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for record in playthrough_years:
        for tag in tags_by_game.get(record["igdb_id"], []):
            result[record["year"]][tag] += record["hours"]

    return {year: dict(tags) for year, tags in result.items()}


def top_tags_by_year(
    yearly_hours: dict[int, dict[str, float]],
    top_n: int = 5,
) -> dict[int, list[tuple[str, float]]]:
    return {
        year: sorted(tags.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        for year, tags in yearly_hours.items()
    }


def build_year_vectors(
    yearly_hours: dict[int, dict[str, float]],
    vocabulary: list[str],
) -> dict[int, list[float]]:
    """Project each year's tag-hours onto a fixed, shared vocabulary (missing tag = 0)."""
    return {
        year: [tags.get(tag, 0.0) for tag in vocabulary]
        for year, tags in yearly_hours.items()
    }


def _cosine_similarity(a: list[float], b: list[float]) -> float | None:
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return None  # a year with none of the vocabulary's tags has no defined direction
    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)


def taste_similarity_matrix(year_vectors: dict[int, list[float]]) -> dict[int, dict[int, float | None]]:
    """Full year x year cosine similarity matrix (diagonal is always 1.0 for a non-empty year)."""
    years = sorted(year_vectors)
    return {
        y1: {y2: _cosine_similarity(year_vectors[y1], year_vectors[y2]) for y2 in years}
        for y1 in years
    }
