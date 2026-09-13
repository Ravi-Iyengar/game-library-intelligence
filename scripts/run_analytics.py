#!/usr/bin/env python3
"""
Sprint 4 CLI: build the analytics report and write it to a JSON file.

    python scripts/run_analytics.py --db data/processed/glip.db --out data/processed/analytics_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage.db import get_connection, init_schema  # noqa: E402
from src.analytics.report import build_analytics_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the GLIP analytics report.")
    parser.add_argument("--db", default="data/processed/glip.db")
    parser.add_argument("--out", default="data/processed/analytics_report.json")
    parser.add_argument("--min-games", type=int, default=2)
    args = parser.parse_args()

    conn = get_connection(args.db)
    init_schema(conn)

    report = build_analytics_report(conn, min_games=args.min_games)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))

    overview = report["library_overview"]
    print(f"Analytics report written to {out_path.resolve()}")
    print(f"  total_games: {overview['total_games']}")
    print(f"  total_hours_logged: {overview['total_hours_logged']}")
    print(f"  status_counts: {overview['status_counts']}")

    conn.close()


if __name__ == "__main__":
    main()
