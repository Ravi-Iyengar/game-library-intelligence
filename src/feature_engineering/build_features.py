"""
Sprint 3 — Feature store.

Builds one row per game in the `features` table, combining:
  * the game's own signal (rating, hours, status flags)
  * one-hot tags for genre/theme/developer/publisher (pruned vocabulary)
  * leave-one-out tag affinities for genre/theme/developer/publisher/franchise
  * engagement metrics from playthroughs/sessions
  * completion/status-signal features (see completion.py for the
    status-mapping assumption that needs your confirmation)
  * review text stats + embedding (embedding is null if
    sentence-transformers isn't available — see embeddings.py)

Every ML model in later sprints reads from this table and only this
table — no model gets custom preprocessing, per the project's own
design principle.

Usage (see scripts/run_feature_engineering.py for the CLI):

    from src.storage.db import get_connection
    from src.feature_engineering.build_features import build_feature_store

    conn = get_connection("data/processed/glip.db")
    summary = build_feature_store(conn)
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
from collections import defaultdict

from src.feature_engineering.affinity import compute_tag_affinities
from src.feature_engineering.completion import compute_completion_features
from src.feature_engineering.embeddings import embed_reviews
from src.feature_engineering.engagement import compute_engagement_metrics
from src.feature_engineering.reviews import (
    extract_review_text_features,
    select_representative_review,
)
from src.feature_engineering.vocabulary import build_all_vocabularies, one_hot
from src.utils.ratings import effective_rating

logger = logging.getLogger(__name__)


def _tag_list(raw_json: str | None) -> list[str]:
    if not raw_json:
        return []
    try:
        return json.loads(raw_json) or []
    except json.JSONDecodeError:
        return []


def _fetch_games(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT g.igdb_id, g.genres, g.themes, g.developers, g.publishers, g.franchises,
               ug.rating, ug.total_hours, ug.status,
               ug.is_backlog, ug.is_playing, ug.is_wishlist, ug.is_liked
        FROM games g
        LEFT JOIN user_games ug ON ug.igdb_id = g.igdb_id
        """
    ).fetchall()


def _fetch_playthroughs_and_sessions(
    conn: sqlite3.Connection,
) -> tuple[dict[str, list[dict]], dict[int, list[dict]]]:
    playthroughs_by_game: dict[str, list[dict]] = defaultdict(list)
    for row in conn.execute(
        "SELECT playthrough_id, igdb_id, is_replay, start_date, is_master, updated_at, review "
        "FROM playthroughs"
    ):
        playthroughs_by_game[row["igdb_id"]].append(dict(row))

    sessions_by_playthrough: dict[int, list[dict]] = defaultdict(list)
    for row in conn.execute(
        "SELECT playthrough_id, range_start_date, session_start_date, hours, minutes FROM sessions"
    ):
        sessions_by_playthrough[row["playthrough_id"]].append(dict(row))

    return playthroughs_by_game, sessions_by_playthrough


def build_feature_store(
    conn: sqlite3.Connection,
    min_occurrences: int = 5,
    compute_embeddings: bool = True,
) -> dict[str, int]:
    games = _fetch_games(conn)
    vocab = build_all_vocabularies(conn, min_occurrences)
    playthroughs_by_game, sessions_by_playthrough = _fetch_playthroughs_and_sessions(conn)

    affinity_inputs = [
        {
            "igdb_id": g["igdb_id"],
            "rating": effective_rating(g["rating"]),
            "total_hours": g["total_hours"],
            "tags": {
                "genres": _tag_list(g["genres"]),
                "themes": _tag_list(g["themes"]),
                "developers": _tag_list(g["developers"]),
                "publishers": _tag_list(g["publishers"]),
                "franchises": _tag_list(g["franchises"]),
            },
        }
        for g in games
    ]
    affinities = compute_tag_affinities(affinity_inputs)

    # Gather representative reviews across all games first, so embeddings
    # (the expensive step) can run as a single batch instead of one call
    # per game.
    representative_reviews: dict[str, str | None] = {}
    engagement_by_game: dict[str, dict] = {}

    for g in games:
        igdb_id = g["igdb_id"]
        game_playthroughs = playthroughs_by_game.get(igdb_id, [])
        game_sessions = [
            s
            for pt in game_playthroughs
            for s in sessions_by_playthrough.get(pt["playthrough_id"], [])
        ]
        engagement_by_game[igdb_id] = compute_engagement_metrics(game_playthroughs, game_sessions)
        representative_reviews[igdb_id] = select_representative_review(game_playthroughs)

    igdb_ids_in_order = [g["igdb_id"] for g in games]
    review_texts_in_order = [representative_reviews[i] for i in igdb_ids_in_order]
    embeddings_in_order = (
        embed_reviews(review_texts_in_order)
        if compute_embeddings
        else [None] * len(review_texts_in_order)
    )
    embedding_by_game = dict(zip(igdb_ids_in_order, embeddings_in_order))

    now = dt.datetime.utcnow().isoformat(timespec="seconds")
    n_written = 0
    n_with_embedding = 0

    for g in games:
        igdb_id = g["igdb_id"]
        engagement = engagement_by_game[igdb_id]
        completion = compute_completion_features(
            status=g["status"],
            is_backlog=g["is_backlog"],
            is_playing=g["is_playing"],
            is_wishlist=g["is_wishlist"],
            is_liked=g["is_liked"],
            playthrough_count=engagement["playthrough_count"],
            replay_count=engagement["replay_count"],
        )
        review_text = representative_reviews[igdb_id]
        review_features = extract_review_text_features(review_text)
        embedding = embedding_by_game.get(igdb_id)
        if embedding is not None:
            n_with_embedding += 1

        feature_dict = {
            "igdb_id": igdb_id,
            "own_rating": effective_rating(g["rating"]),
            "own_total_hours": g["total_hours"],
            **engagement,
            **completion,
            **affinities.get(igdb_id, {}),
            "genre_one_hot": one_hot(_tag_list(g["genres"]), vocab["genres"]),
            "theme_one_hot": one_hot(_tag_list(g["themes"]), vocab["themes"]),
            "developer_one_hot": one_hot(_tag_list(g["developers"]), vocab["developers"]),
            "publisher_one_hot": one_hot(_tag_list(g["publishers"]), vocab["publishers"]),
            **review_features,
            "review_embedding": embedding,
        }

        conn.execute(
            """
            INSERT INTO features (igdb_id, feature_json, computed_at)
            VALUES (?, ?, ?)
            ON CONFLICT(igdb_id) DO UPDATE SET
                feature_json = excluded.feature_json,
                computed_at = excluded.computed_at
            """,
            (igdb_id, json.dumps(feature_dict), now),
        )
        n_written += 1

    conn.commit()

    summary = {
        "games": n_written,
        "with_review_embedding": n_with_embedding,
        "vocab_genres": len(vocab["genres"]),
        "vocab_themes": len(vocab["themes"]),
        "vocab_developers": len(vocab["developers"]),
        "vocab_publishers": len(vocab["publishers"]),
    }
    logger.info("Feature store build complete: %s", summary)
    return summary
