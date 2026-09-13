#!/usr/bin/env python3
"""
Sprint 6 CLI: generate ranked backlog recommendations.

    python scripts/run_recommendations.py --db data/processed/glip.db

Writes to the `recommendations` table and prints the top N to the
console. Requires at least one trained model (scripts/run_ml_training.py)
— missing ones just drop out of the composite score's weighting rather
than blocking the whole run.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage.db import get_connection, init_schema  # noqa: E402
from src.recommendation.generate import generate_recommendations  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GLIP backlog recommendations.")
    parser.add_argument("--db", default="data/processed/glip.db")
    parser.add_argument("--models-dir", default="data/models")
    parser.add_argument("--top", type=int, default=10, help="How many to print (all are still saved)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    conn = get_connection(args.db)
    init_schema(conn)

    try:
        ranked = generate_recommendations(conn, models_dir=args.models_dir)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Ranked {len(ranked)} backlog games. Top {min(args.top, len(ranked))}:\n")
    for _, row in ranked.head(args.top).iterrows():
        print(f"{row['recommendation_score']:5.1f}  {row['title']}")
        factors = row["explanation"].get("top_factors", [])
        if factors:
            factor_str = ", ".join(
                f"{'+' if f['contribution'] > 0 else '-'} {f['feature']}" for f in factors[:3]
            )
            print(f"       {factor_str}")

    conn.close()


if __name__ == "__main__":
    main()
