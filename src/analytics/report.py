"""
Sprint 4 — Analytics.

Assembles every analysis in this package into one report dict, ready
to serialize to JSON for the Sprint 7 dashboard to read. Nothing here
writes to the database — analytics is read-only, downstream of
ingestion/enrichment/feature-engineering.

Usage (see scripts/run_analytics.py for the CLI):

    from src.storage.db import get_connection
    from src.analytics.report import build_analytics_report

    conn = get_connection("data/processed/glip.db")
    report = build_analytics_report(conn)
"""
from __future__ import annotations

import sqlite3

from src.analytics.completion_analysis import completion_rate_by_tag
from src.analytics.counts import library_overview
from src.analytics.data_loader import load_games, load_playthrough_years
from src.analytics.replay_analysis import most_replayed
from src.analytics.tag_preferences import tag_preference_table
from src.analytics.taste_drift import (
    build_year_vectors,
    taste_similarity_matrix,
    top_tags_by_year,
    yearly_tag_hours,
)
from src.feature_engineering.vocabulary import build_vocabulary

# franchises are deliberately excluded from tag_preferences/replay/completion
# breakdowns below by default — most franchises in a personal library have
# only 1-2 entries, so a game_count-based min_games filter already prunes
# almost all of them. They're still reported on their own (see
# franchise_preferences) since spec explicitly calls out Franchise Affinity.
PREFERENCE_CATEGORIES = ("genres", "themes", "developers", "publishers")


def build_analytics_report(
    conn: sqlite3.Connection,
    min_games: int = 2,
    min_tag_occurrences_for_drift: int = 5,
    top_n_per_year: int = 5,
) -> dict:
    games = load_games(conn)
    playthrough_years = load_playthrough_years(conn)

    report: dict = {
        "library_overview": library_overview(games),
        "tag_preferences": {
            cat: tag_preference_table(games, cat, min_games=min_games)
            for cat in PREFERENCE_CATEGORIES
        },
        "franchise_preferences": tag_preference_table(games, "franchises", min_games=min_games),
        "replay_analysis": {
            cat: most_replayed(games, cat) for cat in PREFERENCE_CATEGORIES
        },
        "completion_analysis": {
            cat: completion_rate_by_tag(games, cat, min_games=min_games)
            for cat in PREFERENCE_CATEGORIES
        },
    }

    # Taste drift needs a stable, shared vocabulary across years, or the
    # cosine similarity would be comparing vectors built from different
    # axes. Reuses the same pruning rule as Sprint 3's one-hot vocab.
    genre_vocab = build_vocabulary(conn, "genres", min_occurrences=min_tag_occurrences_for_drift)
    genre_hours_by_year = yearly_tag_hours(games, playthrough_years, category="genres")
    genre_vectors = build_year_vectors(genre_hours_by_year, genre_vocab)

    report["taste_drift"] = {
        "vocabulary": genre_vocab,
        "top_genres_by_year": top_tags_by_year(genre_hours_by_year, top_n=top_n_per_year),
        "similarity_matrix": taste_similarity_matrix(genre_vectors),
    }

    return report
