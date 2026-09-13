#!/usr/bin/env python3
"""
Sprint 1 CLI: build the SQLite database and load a library export into it.

Usage:
    python scripts/run_ingestion.py path/to/export.json [--db data/processed/glip.db]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/run_ingestion.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage.db import get_connection, init_schema  # noqa: E402
from src.ingestion.ingest_json import ingest_export  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a GLIP library export into SQLite.")
    parser.add_argument("export_path", help="Path to the JSON export file")
    parser.add_argument(
        "--db",
        default="data/processed/glip.db",
        help="Path to the SQLite database file (default: data/processed/glip.db)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    conn = get_connection(args.db)
    init_schema(conn)
    summary = ingest_export(conn, args.export_path)

    print("Ingestion summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"Database: {Path(args.db).resolve()}")

    conn.close()


if __name__ == "__main__":
    main()
