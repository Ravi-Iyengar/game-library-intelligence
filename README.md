# Gaming Library Intelligence Platform (GLIP)

A personal gaming analytics and recommendation platform, built around a
user-owned library export (Backloggd-style JSON), enriched with IGDB
metadata, and used to drive preference analytics, explainable ML, and
backlog recommendations.

See `docs/SPECIFICATION.md` and `docs/ENGINEERING_DESIGN.md` for the
full project spec and architecture. **All 7 sprints are implemented** —
ingestion, IGDB enrichment, feature store, analytics, ML training, the
recommendation engine, and the Streamlit dashboard. See `docs/ROADMAP.md`
for the full history of design decisions and real data-quality findings
along the way, including one important caveat about Sprint 7 (the
dashboard's rendering layer couldn't be run in the sandbox this was
built in — see that section before your first `streamlit run`).

## Quickstart — the full pipeline, in order

```bash
pip install -r requirements/requirements.txt

# Sprint 1: ingest your export into a fresh SQLite database
python scripts/run_ingestion.py data/raw/export.json --db data/processed/glip.db

# Sprint 2: enrich games with IGDB metadata (needs Twitch/IGDB credentials —
# see docs/ROADMAP.md; never pass these as CLI args)

python scripts/run_enrichment.py --db data/processed/glip.db

# Sprint 3: build the feature store (one row per game)
python scripts/run_feature_engineering.py --db data/processed/glip.db

# Sprint 4: build the analytics report
python scripts/run_analytics.py --db data/processed/glip.db --out data/processed/analytics_report.json

# Sprint 5: train the rating/completion/replay/engagement models
python scripts/run_ml_training.py --db data/processed/glip.db

# Sprint 6: rank your backlog
python scripts/run_recommendations.py --db data/processed/glip.db

# Sprint 7: launch the dashboard
python -m streamlit run streamlit_app/app.py
# Run the tests
pytest tests/
```

Ingestion creates/updates `games`, `user_games`, `playthroughs`, and
`sessions` tables from the raw export. Enrichment fills in `games`'
metadata columns (genres, themes, developers, ...) from IGDB, matched
strictly by `igdb_id`. Feature engineering writes one row per game into
`features`, combining engagement metrics, tag one-hot/affinity, status
signals, and review text/embedding features. Analytics reads all of the
above into one JSON report (library overview, tag preferences, replay
and completion analysis, taste drift). ML training saves a model per
target to `data/models/`, restricted to features available for any
game (tags, affinities, IGDB rating) so the same models can score your
unplayed/unfinished backlog. The recommendation engine ranks that
backlog by a composite predicted-enjoyment score with a per-game
explanation, writing to the `recommendations` table. The dashboard
reads all of it and puts a UI on top, with a button to regenerate
recommendations live.

## Repository layout

```
gaming-library-intelligence/
├── docs/                 Specification, design, schema, roadmap
├── config/               settings.yaml, logging.yaml
├── data/                 raw / enriched / processed / features / models
├── src/
│   ├── ingestion/        Sprint 1 (implemented)
│   ├── enrichment/       Sprint 2 (IGDB metadata) — not yet implemented
│   ├── feature_engineering/  Sprint 3 — not yet implemented
│   ├── analytics/        Sprint 4 — not yet implemented
│   ├── ml/               Sprint 5 — not yet implemented
│   ├── recommendation/   Sprint 6 — not yet implemented
│   ├── visualization/    Sprint 7 (Streamlit) — not yet implemented
│   ├── storage/          DB connection + schema (implemented)
│   └── utils/
├── tests/
├── notebooks/
├── requirements/
├── scripts/
└── streamlit_app/
```
"# game-library-intelligence" 
