# Roadmap

| Sprint | Deliverable | Status |
|---|---|---|
| 1 | Database + JSON ingestion | **Done** — `src/storage`, `src/ingestion`, `scripts/run_ingestion.py` |
| 2 | IGDB enrichment (fills `games` metadata columns, keyed by `igdb_id`) | **Done** — `src/enrichment`, `scripts/run_enrichment.py` |
| 3 | Feature store (one row per game in `features`, incl. review embeddings) | **Done** — `src/feature_engineering`, `scripts/run_feature_engineering.py` |
| 4 | Analytics (genre/theme/developer/franchise preferences, replay analysis) | **Done** — `src/analytics`, `scripts/run_analytics.py` |
| 5 | ML (rating / completion / replay / engagement prediction) | **Done** — `src/ml`, `scripts/run_ml_training.py` |
| 6 | Recommendation engine (ranked backlog + SHAP explanations) | **Done** — `src/recommendation`, `scripts/run_recommendations.py` |
| 7 | Streamlit dashboard | **Done** — `streamlit_app/`, `streamlit run streamlit_app/app.py` |

## Running Sprint 2

Set your Twitch/IGDB credentials as environment variables (never as CLI
args — those land in shell history, and never commit them to the repo):

```bash
export IGDB_CLIENT_ID=...
export IGDB_CLIENT_SECRET=...
python scripts/run_enrichment.py --db data/processed/glip.db
```

Only games with `enriched_at IS NULL` are processed by default. IDs
IGDB has no match for (delisted/invalid) are stamped as attempted so
they aren't re-queried every run; pass `--force` to re-enrich
everything, e.g. after IGDB's catalog has changed upstream.

Enrichment matches strictly on `igdb_id` (never title), queries in
batches of up to 500 ids (IGDB's per-request limit), and respects
IGDB's documented 4 requests/second rate limit.

## Running Sprint 3

```bash
python scripts/run_feature_engineering.py --db data/processed/glip.db
```

Builds one row per game in `features` (a JSON blob per row — see
`src/feature_engineering/build_features.py` for the full field list).
Genre/theme/developer/publisher one-hot vectors only include tags that
appear on at least `--min-occurrences` games (default 5) — this needs
Sprint 2 to have run first, or every vocabulary will be empty. Tag
*affinity* scores (genre/theme/developer/publisher/franchise) work
even without one-hot vocab and are computed leave-one-out (a game's
own rating never contributes to its own affinity score, to avoid
target leakage into the Sprint 5 models).

**Status mapping — confirmed with you directly, no longer a guess:**
`src/feature_engineering/completion.py` maps your export's actual
status values (`completed / retired / played / abandoned / shelved`)
onto positive/neutral/negative signal categories: `completed` →
positive, `abandoned` → negative, and `shelved` / `played` / `retired`
all → neutral (both `played`, e.g. open-ended games with no defined
ending, and `retired`, "got my fill, not going back" — are neither a
preference-for nor a preference-against signal). On your real 408-game
library this produces `{positive: 199, neutral: 195, negative: 14}`.
This is what defines the training label in Sprint 5.

Review embeddings need `sentence-transformers` installed and network
access to download `all-mpnet-base-v2` the first time — pass
`--no-embeddings` to skip that step (e.g. while iterating on other
features), and re-run without the flag later to backfill embeddings
into rows that don't have one yet.

## Running Sprint 4

```bash
python scripts/run_analytics.py --db data/processed/glip.db --out data/processed/analytics_report.json
```

Writes one JSON report combining: library overview (counts by status/
signal category), genre/theme/developer/publisher/franchise preference
tables (hours-weighted average rating), replay analysis, completion
rate by tag, and taste drift (yearly genre-hours vectors + a
year-by-year cosine similarity matrix). Read-only — nothing here
writes to the database. Needs Sprint 2 (enrichment) to have actually
run for the tag-based breakdowns to be non-empty; the overview/status
counts work regardless.

**Data quality finding, confirmed with you and fixed in both Sprint 3
and Sprint 4:** your export uses `rating: 0` as its "no rating given"
default (Backloggd can't actually assign a 0-star rating) — confirmed
by every `rating=0` game also having `total_hours: 0`, including 61
games logged `completed`. Left unhandled, this would have skewed every
rating-weighted number (average rating, tag affinities, and eventually
the Sprint 5 model's training target) toward "everything is mediocre."
Both sprints now read ratings through
`src/utils/ratings.py::effective_rating()`, which treats a raw 0 as
unrated. On your real data this moved `rated_games` from 408 to a
correct 309 and the average rating from 5.2 to 6.9.

Also worth knowing for anything that displays ratings later: Backloggd
stores ratings as stars × 2 (a 5-star scale in half-star steps, so the
raw 0–10 integer in the export = stars × 2 — e.g. a raw 8 is 4 stars).
No code change was needed for this since every calculation already
treats the raw value as a plain, consistent 0–10 number; it only
matters if you want to show a rating back to yourself as stars.

## Running Sprint 5

```bash
python scripts/run_ml_training.py --db data/processed/glip.db
```

Trains 4 models — rating (regression), completion (classification),
replay (classification), engagement (regression, a composite score —
see `src/ml/targets.py` for the exact formula) — comparing RandomForest,
XGBoost, CatBoost, and LightGBM candidates (whichever are installed;
missing ones are skipped with a log line, not an error) via 80/20
holdout + 5-fold CV, then saves the winning model per target to
`data/models/*.joblib` plus a full report to
`data/processed/ml_training_report.json`.

**Two important design decisions, both driven by real problems found in
your actual data:**

1. **"Has this game been played?" uses real evidence, not the status
   label.** 42 of your 408 games are labeled `completed`/`played` but
   have zero hours, zero playthroughs, and no real rating — bulk-import
   placeholders, not real outcomes. `src/ml/dataset.py::has_been_played()`
   requires actual evidence (logged hours, a playthrough record, or a
   genuine rating) before a game counts as played at all. On your real
   library this correctly identifies 366/408 as actually played.

   **Refined further, confirmed directly with you:** `is_backlog=True`
   means "haven't beaten this yet" — even when real hours are logged
   (72 of your games, including 45 hours into Red Dead Redemption 2,
   are still flagged `is_backlog=True`). Their current rating/status is
   real but not final, so `has_settled_outcome()` requires
   `has_been_played()` **and** `is_backlog=False` before a game counts
   as a training example. On your real library: 294 settled (used for
   training), 72 played-but-still-in-progress, 42 untouched. The 72
   in-progress games move to the Sprint 6 recommendation pool instead —
   see below.

2. **Model inputs are restricted to "pre-play" features only**: tag
   one-hot vectors, tag affinities, and IGDB's own aggregate rating —
   nothing derived from actually having played the game (hours,
   sessions, reviews, review embeddings, replay counts). This is
   because Sprint 6 will use these same models to score **backlog
   games you haven't played yet**, and those games will always have
   zero/null for anything play-derived. A model trained on hours/reviews
   would just learn "0 hours = unplayed" and be worthless for actually
   ranking your backlog. This means review embeddings, despite being in
   the spec's feature store, are *not* used as model inputs here — a
   deliberate deviation from the spec, for a use-case reason the spec
   itself didn't reconcile (its own feature store mixes pre-play and
   post-play signals into one table with no such distinction).

**Needs enrichment data to actually produce models:** in this sandbox
(no network to hit real IGDB), every pre-play feature column is empty,
so all four targets are correctly skipped with a clear warning rather
than crashing — verified end-to-end with synthetic data that the
training pipeline itself works once real tags/affinities exist. Once
you've run Sprint 2 for real, re-run this script and it'll actually
train.

## Running Sprint 6

```bash
python scripts/run_recommendations.py --db data/processed/glip.db
```

Ranks your backlog (unplayed games — see below for the precise
definition) by a composite predicted-enjoyment score and writes the
result to the `recommendations` table, printing the top 10 with a
one-line "why" for each. Needs at least one trained model from
Sprint 5.

**The backlog pool is "not yet beaten," confirmed directly with you** —
including games you've started but not finished, not just untouched
ones. `src/recommendation/candidate_pool.py` defines it as the negation
of Sprint 5's training filter: `~has_settled_outcome` (plus a wishlist
exclusion). A game is either settled (safe to learn a final label
from, not worth recommending again) or still open business (safe to
recommend, not yet safe to treat as a finished outcome) — never both.
On your real library this pool is 114 games: the 42 untouched plus the
72 you've started but haven't beaten yet (e.g. Red Dead Redemption 2 at
45 hours) — exactly the games you'd actually want ranked by "should I
go back and finish this."

**Composite score**, per the spec:
`0.40×rating + 0.25×completion_probability + 0.20×replay_probability
+ 0.15×engagement_score`, normalized to a 0-100 display value. If a
target's model wasn't trained (e.g. no enrichment data yet), its
term is dropped and the remaining weights renormalize to sum to 1,
rather than silently scoring every candidate lower by a fixed amount.

**Explanation source:** each recommendation's SHAP top-factors come
from whichever of [rating, completion, replay, engagement] models is
available, in that priority order — rating first, since "why would I
like this" is the natural read of the spec's own example
(`+ Larian Studios, + CRPG, + Party-based RPG, ...`).

**A real bug this integration caught:** scikit-learn's `SimpleImputer`
silently *drops* any column that was entirely missing across every
training row (e.g. an affinity category nothing in your library has a
value for yet), so its output can be narrower than its input. The
explanation code was naively re-labeling that narrower output with the
original column names — which, past the first dropped column, silently
attributes SHAP values to the wrong feature. Fixed to read the
imputer's actual surviving column names; a regression test locks this
in (`tests/test_ml.py::test_explain_predictions_handles_all_nan_feature_column`).

## Running Sprint 7

```bash
streamlit run streamlit_app/app.py
```

Sidebar navigation across all 8 sections from the spec: Overview,
Genre/Theme/Developer/Franchise Analysis, Replay Analysis, Taste Drift,
Recommendation Center (with a "Regenerate recommendations" button that
re-runs Sprint 6 live), and Explainability (pick any recommended game,
see its top contributing factors as a bar chart).

**Important limitation, stated plainly:** this sandbox has neither
`streamlit` nor `plotly` installed, and no browser — so `app.py` itself
has never actually been run or visually inspected. Everything that
*can* be tested headlessly was: all the DB-querying and pandas logic
lives in `streamlit_app/data_access.py`, which has zero dependency on
streamlit and is covered by `tests/test_dashboard_data.py` (10 tests,
including a full round-trip through the "Regenerate recommendations"
button's underlying function against real trained models). `app.py`
itself is kept deliberately thin — it does almost nothing but call
`data_access.py` and hand the result to `st.dataframe`/`st.bar_chart`/
`st.metric` — and sticks to long-stable streamlit APIs rather than
newer ones I couldn't verify. But the rendering layer is the one part
of this whole project that hasn't been run for real. Please treat your
first `streamlit run` as the actual first test of `app.py` specifically,
and tell me what breaks if anything does.

Charts use streamlit's built-in `st.bar_chart` (Altair under the hood,
ships with streamlit, nothing extra to install) rather than the
`plotly` listed in requirements — a deliberate choice to minimize the
untestable surface area given the constraint above. `plotly` is left
in requirements.txt per the spec's own tech stack if you'd rather swap
the charts for it later; nothing here requires it.
