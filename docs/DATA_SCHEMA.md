# Data Schema (Sprint 1)

Source of truth: `src/storage/schema.sql`. This doc explains the mapping
from the raw export to each table; regenerate from the SQL if they drift.

## Source shape

The export is a JSON array. Each entry:

```json
{
  "id": "119171",
  "name": "Baldur's Gate III",
  "game_log": { "status": "...", "rating": 9, "total_hours": 120, ... },
  "playthroughs": {
    "<playthrough_id>": {
      "id": 5001, "rating": 9, "hours_played": 40, "mins_played": 0,
      "is_replay": false, "start_date": "...", "finish_date": "...",
      "review": "...", "play_dates": [ { "id": 9001, "hours": 3, ... } ]
    }
  }
}
```

`id` is the IGDB id and is used as `igdb_id` everywhere — never re-derived
from `name`.

## Tables

| Table | Grain | Source |
|---|---|---|
| `games` | one row per game | top-level `id` + `name`; enrichment columns (genres, themes, developers, ...) are NULL until Sprint 2 fills them from IGDB |
| `user_games` | one row per game | `game_log` block: status, rating, total_hours (hours+minutes combined), is_backlog/is_playing/is_wishlist/is_liked flags, last_edited_at |
| `playthroughs` | one row per playthrough | each entry in the `playthroughs` dict |
| `sessions` | one row per play-date entry | each entry's `play_dates` list, nested inside a playthrough |
| `features` | one row per game (empty for now) | Sprint 3 |
| `recommendations` | one row per game (empty for now) | Sprint 9 |

## Deliberate raw-load choices

- **Status is stored as-is.** The export uses `completed / retired / played /
  abandoned / shelved`; the spec's status vocabulary (`completed / shelved /
  abandoned / playing / backlog / wishlist`) is a *behavioral interpretation*,
  not a 1:1 match (e.g. `is_backlog` and `is_wishlist` are separate boolean
  flags in the export, not status values). Reconciling the two is an
  analytics/feature-engineering concern, not an ingestion concern — ingestion
  stays a faithful copy of the source so nothing is lost or silently
  reinterpreted before analysis.
- **No session table existed as a literal `sessions` key in the source** —
  the closest analog is each playthrough's `play_dates` list, which is what's
  loaded into `sessions`.
- **Re-running ingestion is idempotent** (`INSERT OR REPLACE` keyed by the
  export's own ids), so re-importing an updated export just refreshes rows.
