"""
Sprint 2 — IGDB enrichment.

Reads the igdb_ids already sitting in the `games` table (populated by
Sprint 1 ingestion) and fills in the metadata columns from IGDB,
matching strictly on id — never on title, per the project spec.

Usage (see scripts/run_enrichment.py for the CLI):

    from src.storage.db import get_connection
    from src.enrichment.igdb_client import IGDBClient
    from src.enrichment.enrich_games import enrich_games

    conn = get_connection("data/processed/glip.db")
    client = IGDBClient.from_env()
    summary = enrich_games(conn, client)
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
from typing import Any

from src.enrichment.igdb_client import IGDBClient

logger = logging.getLogger(__name__)

BATCH_SIZE = 500  # IGDB's documented max `limit` per request

FIELDS = (
    "fields name, genres.name, themes.name, keywords.name, "
    "franchises.name, collections.name, game_modes.name, "
    "player_perspectives.name, "
    "involved_companies.company.name, involved_companies.developer, "
    "involved_companies.publisher, "
    "first_release_date, aggregated_rating, aggregated_rating_count, "
    "platforms.name;"
)


def _names(field: list[dict] | None) -> list[str]:
    """Pull out .name from a list of {id, name} objects IGDB returns for expanded fields."""
    if not field:
        return []
    return [item["name"] for item in field if "name" in item]


def _split_companies(involved_companies: list[dict] | None) -> tuple[list[str], list[str]]:
    """involved_companies each carry `developer`/`publisher` booleans plus a nested company.name."""
    developers, publishers = [], []
    for ic in involved_companies or []:
        name = (ic.get("company") or {}).get("name")
        if not name:
            continue
        if ic.get("developer"):
            developers.append(name)
        if ic.get("publisher"):
            publishers.append(name)
    return developers, publishers


def parse_igdb_game(raw: dict) -> dict[str, Any]:
    """Map one IGDB /games response object onto our `games` table column names."""
    developers, publishers = _split_companies(raw.get("involved_companies"))

    release_date = None
    if raw.get("first_release_date") is not None:
        release_date = dt.datetime.utcfromtimestamp(raw["first_release_date"]).date().isoformat()

    return {
        "igdb_id": str(raw["id"]),
        "genres": json.dumps(_names(raw.get("genres"))),
        "themes": json.dumps(_names(raw.get("themes"))),
        "keywords": json.dumps(_names(raw.get("keywords"))),
        "franchises": json.dumps(_names(raw.get("franchises"))),
        "collections": json.dumps(_names(raw.get("collections"))),
        "game_modes": json.dumps(_names(raw.get("game_modes"))),
        "player_perspectives": json.dumps(_names(raw.get("player_perspectives"))),
        "developers": json.dumps(developers),
        "publishers": json.dumps(publishers),
        "release_date": release_date,
        "aggregated_rating": raw.get("aggregated_rating"),
        "aggregated_rating_count": raw.get("aggregated_rating_count"),
        "platforms": json.dumps(_names(raw.get("platforms"))),
        "enriched_at": dt.datetime.utcnow().isoformat(timespec="seconds"),
    }


def _chunks(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _pending_igdb_ids(conn: sqlite3.Connection, force: bool) -> list[str]:
    clause = "" if force else "WHERE enriched_at IS NULL"
    rows = conn.execute(f"SELECT igdb_id FROM games {clause}").fetchall()
    return [row["igdb_id"] for row in rows]


def _update_game(conn: sqlite3.Connection, parsed: dict[str, Any]) -> None:
    conn.execute(
        """
        UPDATE games SET
            genres = :genres,
            themes = :themes,
            keywords = :keywords,
            franchises = :franchises,
            collections = :collections,
            game_modes = :game_modes,
            player_perspectives = :player_perspectives,
            developers = :developers,
            publishers = :publishers,
            release_date = :release_date,
            aggregated_rating = :aggregated_rating,
            aggregated_rating_count = :aggregated_rating_count,
            platforms = :platforms,
            enriched_at = :enriched_at
        WHERE igdb_id = :igdb_id
        """,
        parsed,
    )


def enrich_games(
    conn: sqlite3.Connection,
    client: IGDBClient,
    force: bool = False,
) -> dict[str, int]:
    """
    Enrich every game in the `games` table that hasn't been enriched yet
    (or all of them, if force=True). Matches strictly on igdb_id.
    """
    igdb_ids = _pending_igdb_ids(conn, force)
    if not igdb_ids:
        logger.info("Nothing to enrich — all games already have enriched_at set.")
        return {"requested": 0, "matched": 0, "not_found": 0}

    matched = 0
    found_ids: set[str] = set()

    for batch in _chunks(igdb_ids, BATCH_SIZE):
        id_list = ", ".join(batch)
        query = f"{FIELDS} where id = ({id_list}); limit {BATCH_SIZE};"
        results = client.query("games", query)

        for raw in results:
            parsed = parse_igdb_game(raw)
            _update_game(conn, parsed)
            matched += 1
            found_ids.add(parsed["igdb_id"])

        conn.commit()
        logger.info("Enriched %d/%d games in this batch.", len(results), len(batch))

    not_found = [i for i in igdb_ids if i not in found_ids]
    if not_found:
        logger.warning(
            "%d igdb_id(s) had no match on IGDB (delisted/invalid id?): %s",
            len(not_found), not_found,
        )
        # Still stamp enriched_at so these don't get re-queried on every
        # future run — a --force pass will pick them up again if IGDB's
        # catalog changes. Metadata columns are left NULL, which is how
        # "attempted but no match" is distinguished from "never tried".
        stamp = dt.datetime.utcnow().isoformat(timespec="seconds")
        conn.executemany(
            "UPDATE games SET enriched_at = ? WHERE igdb_id = ?",
            [(stamp, igdb_id) for igdb_id in not_found],
        )
        conn.commit()

    return {
        "requested": len(igdb_ids),
        "matched": matched,
        "not_found": len(not_found),
    }
