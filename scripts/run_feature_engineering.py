#!/usr/bin/env python3
"""
Sprint 3 CLI: build the feature store (one row per game in `features`).

    python scripts/run_feature_engineering.py --db data/processed/glip.db

Pass --no-embeddings to skip review embeddings entirely (useful if
sentence-transformers isn't installed yet, or you want a fast rebuild
of the rest of the feature vector while iterating).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage.db import get_connection, init_schema  # noqa: E402
from src.feature_engineering.build_features import build_feature_store  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the GLIP feature store.")
    parser.add_argument("--db", default="data/processed/glip.db")
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=5,
        help="Minimum games a genre/theme/developer/publisher tag must appear in to be one-hot encoded (default: 5)",
    )
    parser.add_argument("--no-embeddings", action="store_true", help="Skip review embeddings")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    conn = get_connection(args.db)
    init_schema(conn)

    summary = build_feature_store(
        conn,
        min_occurrences=args.min_occurrences,
        compute_embeddings=not args.no_embeddings,
    )

    print("Feature store summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    conn.close()


if __name__ == "__main__":
    main()
