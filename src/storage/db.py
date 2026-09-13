"""
Thin SQLite connection helper for GLIP.

Nothing in here contains business logic — it just opens a connection
with sane defaults and applies schema.sql. Ingestion, enrichment, and
every later sprint import get_connection() from here so there is a
single place that owns "how we talk to the database."
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys enabled and Row access."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False)  # ← FIX
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create all tables/indexes if they don't already exist (idempotent)."""
    sql = SCHEMA_PATH.read_text()
    conn.executescript(sql)
    conn.commit()
