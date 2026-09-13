-- Gaming Library Intelligence Platform (GLIP)
-- Sprint 1: Database schema
--
-- Design notes:
--   * igdb_id (TEXT) is the authoritative primary key for a game everywhere,
--     per the project spec. It is never re-derived from title matching.
--   * "games" holds the raw title from the export plus columns reserved for
--     IGDB enrichment (Sprint 2). Enrichment columns are nullable and are
--     left NULL until the enrichment step fills them in.
--   * "user_games" is the single-user status/rating snapshot (the export's
--     game_log block) — one row per game.
--   * "playthroughs" and "sessions" are proper 1:many children, matching the
--     nesting in the source export (playthroughs -> play_dates).
--   * "features" and "recommendations" are created empty here as forward
--     placeholders for Sprint 3 and Sprint 9 — nothing writes to them yet.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS games (
    igdb_id                 TEXT PRIMARY KEY,
    title                   TEXT NOT NULL,

    -- Sprint 2 (IGDB enrichment) fills these in; NULL until then.
    genres                  TEXT,   -- JSON array, stored as text (SQLite has no array type)
    themes                  TEXT,   -- JSON array
    keywords                TEXT,   -- JSON array
    franchises              TEXT,   -- JSON array
    collections              TEXT,   -- JSON array
    game_modes              TEXT,   -- JSON array
    player_perspectives     TEXT,   -- JSON array
    developers              TEXT,   -- JSON array
    publishers              TEXT,   -- JSON array
    release_date            TEXT,
    aggregated_rating        REAL,
    aggregated_rating_count INTEGER,
    platforms                TEXT,   -- JSON array

    enriched_at              TEXT    -- timestamp set by the enrichment step
);

CREATE TABLE IF NOT EXISTS user_games (
    igdb_id             TEXT PRIMARY KEY REFERENCES games(igdb_id),
    status              TEXT,       -- raw status string from the export (completed/retired/played/abandoned/shelved/...)
    rating              INTEGER,
    total_hours         REAL,       -- total_hours + total_minutes/60, from game_log
    is_backlog          INTEGER,    -- 0/1
    is_playing          INTEGER,    -- 0/1
    is_wishlist         INTEGER,    -- 0/1
    is_liked            INTEGER,    -- 0/1 (game_liked)
    last_edited_at      INTEGER     -- unix timestamp, as given
);

CREATE TABLE IF NOT EXISTS playthroughs (
    playthrough_id      INTEGER PRIMARY KEY,
    igdb_id             TEXT NOT NULL REFERENCES games(igdb_id),
    rating              INTEGER,
    hours_played        REAL,       -- hours_played + mins_played/60
    hours_finished      REAL,       -- hours_finished + mins_finished/60
    hours_mastered      REAL,       -- hours_mastered + mins_mastered/60
    is_replay           INTEGER,    -- 0/1
    is_master           INTEGER,    -- 0/1
    start_date          TEXT,
    finish_date         TEXT,
    platform            TEXT,
    played_platform      TEXT,
    review               TEXT,
    review_spoilers      INTEGER,   -- 0/1
    created_at           TEXT,
    updated_at           TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id          INTEGER PRIMARY KEY,
    playthrough_id      INTEGER NOT NULL REFERENCES playthroughs(playthrough_id),
    range_start_date    TEXT,
    range_end_date      TEXT,
    session_start_date  TEXT,       -- the entry's own start_date, if set
    session_finish_date TEXT,       -- the entry's own finish_date, if set
    hours               REAL,
    minutes             REAL,
    note                TEXT
);

-- Forward placeholders — populated in later sprints, not by ingestion.
CREATE TABLE IF NOT EXISTS features (
    igdb_id     TEXT PRIMARY KEY REFERENCES games(igdb_id),
    feature_json TEXT,
    computed_at  TEXT
);

CREATE TABLE IF NOT EXISTS recommendations (
    igdb_id                    TEXT PRIMARY KEY REFERENCES games(igdb_id),
    predicted_rating            REAL,
    predicted_completion_prob   REAL,
    predicted_replay_prob       REAL,
    predicted_engagement        REAL,
    recommendation_score        REAL,
    explanation_json            TEXT,
    generated_at                 TEXT
);

CREATE INDEX IF NOT EXISTS idx_playthroughs_igdb_id ON playthroughs(igdb_id);
CREATE INDEX IF NOT EXISTS idx_sessions_playthrough_id ON sessions(playthrough_id);
