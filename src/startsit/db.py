"""Data-access layer.

SQLite for now; the SQL sticks to what Postgres also accepts (ON CONFLICT ... DO UPDATE)
so a Supabase backend can replace the connection without touching callers.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = "data/startsit.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    player_id   TEXT PRIMARY KEY,  -- nflverse GSIS id
    name        TEXT NOT NULL,
    position    TEXT NOT NULL,
    team        TEXT,
    pfr_id      TEXT
);

CREATE TABLE IF NOT EXISTS player_weeks (
    player_id        TEXT NOT NULL,
    season           INTEGER NOT NULL,
    week             INTEGER NOT NULL,
    team             TEXT,
    opponent         TEXT,
    offense_snaps    INTEGER,
    snap_pct         REAL,
    targets          INTEGER,
    carries          INTEGER,
    receptions       INTEGER,
    target_share     REAL,
    air_yards_share  REAL,
    wopr             REAL,
    fp_std           REAL,
    fp_half          REAL,
    fp_ppr           REAL,
    xfp_std          REAL,
    xfp_half         REAL,
    xfp_ppr          REAL,
    PRIMARY KEY (player_id, season, week)
);

CREATE TABLE IF NOT EXISTS injuries (
    player_id        TEXT NOT NULL,
    season           INTEGER NOT NULL,
    week             INTEGER NOT NULL,
    team             TEXT,
    report_status    TEXT,
    practice_status  TEXT,
    injury           TEXT,
    PRIMARY KEY (player_id, season, week)
);

CREATE TABLE IF NOT EXISTS usage_features (
    player_id            TEXT NOT NULL,
    season               INTEGER NOT NULL,
    through_week         INTEGER NOT NULL,
    games                INTEGER NOT NULL,
    last_game_week       INTEGER NOT NULL,  -- features go stale if this lags through_week
    snap_pct_l3          REAL,
    snap_pct_trend       REAL,
    target_share_l3      REAL,
    opportunities_l3     REAL,
    fp_std_l3            REAL,
    fp_half_l3           REAL,
    fp_ppr_l3            REAL,
    xfp_std_l3           REAL,
    xfp_half_l3          REAL,
    xfp_ppr_l3           REAL,
    xfp_ppr_season       REAL,
    fp_minus_xfp_ppr_l3  REAL,
    PRIMARY KEY (player_id, season, through_week)
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    source       TEXT NOT NULL,
    season       INTEGER NOT NULL,
    max_week     INTEGER,
    row_count    INTEGER NOT NULL,
    finished_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "players": ("player_id",),
    "player_weeks": ("player_id", "season", "week"),
    "injuries": ("player_id", "season", "week"),
    "usage_features": ("player_id", "season", "through_week"),
}


class Database:
    def __init__(self, path: str | Path | None = None):
        path = str(path or os.environ.get("STARTSIT_DB_PATH", DEFAULT_DB_PATH))
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def upsert(self, table: str, rows: Iterable[Mapping[str, Any]]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        keys = PRIMARY_KEYS[table]
        cols = list(rows[0].keys())
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c not in keys)
        sql = (
            f"INSERT INTO {table} ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' for _ in cols)}) "
            f"ON CONFLICT ({', '.join(keys)}) DO UPDATE SET {updates}"
        )
        with self.conn:
            self.conn.executemany(sql, [tuple(r[c] for c in cols) for r in rows])
        return len(rows)

    def record_run(self, source: str, season: int, max_week: int | None, row_count: int) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO ingest_runs (source, season, max_week, row_count) VALUES (?, ?, ?, ?)",
                (source, season, max_week, row_count),
            )

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(sql, params)]

    def close(self) -> None:
        self.conn.close()
