#!/usr/bin/env python3
"""
Sprint 2 CLI: enrich the games table with IGDB metadata.

Requires IGDB_CLIENT_ID and IGDB_CLIENT_SECRET as environment variables
(never pass credentials as CLI args — they'd land in shell history):

    export IGDB_CLIENT_ID=...
    export IGDB_CLIENT_SECRET=...
    python scripts/run_enrichment.py --db data/processed/glip.db

Only rows with enriched_at IS NULL are processed by default; pass
--force to re-enrich everything (e.g. after IGDB data changes).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage.db import get_connection, init_schema  # noqa: E402
from src.enrichment.igdb_client import IGDBClient, IGDBAuthError  # noqa: E402
from src.enrichment.enrich_games import enrich_games  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich GLIP games with IGDB metadata.")
    parser.add_argument("--db", default="data/processed/glip.db")
    parser.add_argument("--force", action="store_true", help="Re-enrich all games, not just new ones")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        client = IGDBClient.from_env()
    except IGDBAuthError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    conn = get_connection(args.db)
    init_schema(conn)

    summary = enrich_games(conn, client, force=args.force)
    print("Enrichment summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    conn.close()


if __name__ == "__main__":
    main()
