#!/usr/bin/env python3
"""
Sprint 5 CLI: train the rating/completion/replay/engagement models.

    python scripts/run_ml_training.py --db data/processed/glip.db

Saves one .joblib model per target to data/models/ and a training
report (metrics, candidate comparison, example explanations) to
data/processed/ml_training_report.json.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage.db import get_connection, init_schema  # noqa: E402
from src.ml.run_training import train_all_models, save_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GLIP ML models.")
    parser.add_argument("--db", default="data/processed/glip.db")
    parser.add_argument("--models-dir", default="data/models")
    parser.add_argument("--report", default="data/processed/ml_training_report.json")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    conn = get_connection(args.db)
    init_schema(conn)

    report = train_all_models(conn, models_dir=args.models_dir)
    save_report(report, args.report)

    print(
        f"{report['n_games_played']}/{report['n_games_total']} games played; "
        f"{report['n_games_settled']} have a settled outcome (trained on)."
    )
    for target_name, result in report["targets"].items():
        print(f"\n{target_name}:")
        print(f"  eligible rows: {result['n_eligible']}")
        print(f"  best candidate: {result['best_candidate']}")
        print(f"  holdout metrics: {result['holdout_metrics']}")
    print(f"\nFull report: {Path(args.report).resolve()}")

    conn.close()


if __name__ == "__main__":
    main()
