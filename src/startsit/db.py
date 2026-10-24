"""Data-access layer: the only module that contains SQL.

Two backends behind one class: SQLite (a file path, for local work and tests) and
Postgres (a postgres:// URL, i.e. Supabase). SQL is written once with `?` placeholders
and only uses syntax both accept; the Postgres backend translates the placeholders.
Callers use the named query methods below, so they never depend on the backend.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, TypedDict, cast

DEFAULT_DB_PATH = "data/startsit.db"

# Types: TEXT, INTEGER and DOUBLE PRECISION mean the same in both backends (Postgres
# REAL is single precision, so it isn't used).
SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    player_id   TEXT PRIMARY KEY,  -- nflverse GSIS id
    name        TEXT NOT NULL,
    position    TEXT NOT NULL,
    team        TEXT,  -- latest roster team
    pfr_id      TEXT
);

CREATE TABLE IF NOT EXISTS roster_weeks (
    player_id  TEXT NOT NULL,
    season     INTEGER NOT NULL,
    week       INTEGER NOT NULL,
    team       TEXT,
    status     TEXT,  -- ACT, RES (IR/PUP/NFI), DEV (practice squad), CUT, RET, INA, EXE, ...
    PRIMARY KEY (player_id, season, week)
);

CREATE TABLE IF NOT EXISTS player_weeks (
    player_id        TEXT NOT NULL,
    season           INTEGER NOT NULL,
    week             INTEGER NOT NULL,
    team             TEXT,
    opponent         TEXT,
    offense_snaps    INTEGER,
    snap_pct         DOUBLE PRECISION,
    targets          INTEGER,
    carries          INTEGER,
    receptions       INTEGER,
    target_share     DOUBLE PRECISION,
    air_yards_share  DOUBLE PRECISION,
    wopr             DOUBLE PRECISION,
    fp_std           DOUBLE PRECISION,
    fp_half          DOUBLE PRECISION,
    fp_ppr           DOUBLE PRECISION,
    xfp_std          DOUBLE PRECISION,
    xfp_half         DOUBLE PRECISION,
    xfp_ppr          DOUBLE PRECISION,
    PRIMARY KEY (player_id, season, week)
);

-- Overwritten as reports update; recommendations must store the packet they used.
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
    snap_pct_l3          DOUBLE PRECISION,
    snap_pct_trend       DOUBLE PRECISION,
    target_share_l3      DOUBLE PRECISION,
    opportunities_l3     DOUBLE PRECISION,
    fp_std_l3            DOUBLE PRECISION,
    fp_half_l3           DOUBLE PRECISION,
    fp_ppr_l3            DOUBLE PRECISION,
    xfp_std_l3           DOUBLE PRECISION,
    xfp_half_l3          DOUBLE PRECISION,
    xfp_ppr_l3           DOUBLE PRECISION,
    xfp_ppr_season       DOUBLE PRECISION,
    fp_minus_xfp_ppr_l3  DOUBLE PRECISION,
    PRIMARY KEY (player_id, season, through_week)
);

CREATE TABLE IF NOT EXISTS games (
    game_id    TEXT PRIMARY KEY,
    season     INTEGER NOT NULL,
    week       INTEGER NOT NULL,
    gameday    TEXT,
    home_team  TEXT NOT NULL,
    away_team  TEXT NOT NULL
);

-- FantasyPros weekly ECR, one archived copy per scrape (see startsit.ingest.rankings).
CREATE TABLE IF NOT EXISTS rankings (
    season           INTEGER NOT NULL,
    week             INTEGER NOT NULL,
    scrape_date      TEXT NOT NULL,
    fantasypros_id   TEXT NOT NULL,
    player_id        TEXT,  -- GSIS id; null for the few players the id map lacks
    player_name      TEXT NOT NULL,
    position         TEXT NOT NULL,
    team             TEXT,  -- FantasyPros abbreviations (LAR, not nflverse's LA)
    opponent         TEXT,  -- e.g. "vs. DAL", "at LV"
    ecr              DOUBLE PRECISION,  -- average expert rank within position
    ecr_sd           DOUBLE PRECISION,
    ecr_best         DOUBLE PRECISION,
    ecr_worst        DOUBLE PRECISION,
    proj_ppr         DOUBLE PRECISION,  -- FantasyPros rank-to-points, PPR scoring
    pos_rank         TEXT,
    start_sit_grade  TEXT,
    game_status      TEXT,
    source_url       TEXT,
    PRIMARY KEY (season, week, scrape_date, fantasypros_id)
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    source       TEXT NOT NULL,
    season       INTEGER NOT NULL,
    max_week     INTEGER,
    row_count    INTEGER NOT NULL,
    finished_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "players": ("player_id",),
    "roster_weeks": ("player_id", "season", "week"),
    "player_weeks": ("player_id", "season", "week"),
    "injuries": ("player_id", "season", "week"),
    "usage_features": ("player_id", "season", "through_week"),
    "games": ("game_id",),
    "rankings": ("season", "week", "scrape_date", "fantasypros_id"),
}
TABLES = (*PRIMARY_KEYS, "ingest_runs")


class PlayerRow(TypedDict):
    player_id: str
    name: str
    position: str
    team: str | None
    pfr_id: str | None


class RosterRow(TypedDict):
    player_id: str
    season: int
    week: int
    team: str | None
    status: str | None


class InjuryRow(TypedDict):
    player_id: str
    season: int
    week: int
    team: str | None
    report_status: str | None
    practice_status: str | None
    injury: str | None


class UsageRow(TypedDict):
    player_id: str
    season: int
    through_week: int
    games: int
    last_game_week: int
    snap_pct_l3: float | None
    snap_pct_trend: float | None
    target_share_l3: float | None
    opportunities_l3: float | None
    fp_std_l3: float
    fp_half_l3: float
    fp_ppr_l3: float
    xfp_std_l3: float
    xfp_half_l3: float
    xfp_ppr_l3: float
    xfp_ppr_season: float
    fp_minus_xfp_ppr_l3: float


class RankingRow(TypedDict):
    season: int
    week: int
    scrape_date: str
    fantasypros_id: str
    player_id: str | None
    player_name: str
    position: str
    team: str | None
    opponent: str | None
    ecr: float | None
    ecr_sd: float | None
    ecr_best: float | None
    ecr_worst: float | None
    proj_ppr: float | None
    pos_rank: str | None
    start_sit_grade: str | None
    game_status: str | None
    source_url: str | None


def resolve_target(target: str | Path | None = None) -> str:
    """Explicit target, else $DATABASE_URL, else $STARTSIT_DB_PATH, else the default file."""
    return str(target or os.environ.get("DATABASE_URL") or os.environ.get("STARTSIT_DB_PATH") or DEFAULT_DB_PATH)


def is_postgres(target: str) -> bool:
    return target.startswith(("postgres://", "postgresql://"))


class Database:
    def __init__(self, target: str | Path | None = None):
        target = resolve_target(target)
        self.is_postgres = is_postgres(target)
        self.conn: Any
        if self.is_postgres:
            import psycopg
            from psycopg.rows import dict_row

            self.conn = psycopg.connect(target, row_factory=dict_row)
            with self.conn.transaction():
                self.conn.execute(SCHEMA)
        else:
            if target != ":memory:":
                Path(target).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(target)
            self.conn.row_factory = sqlite3.Row
            self.conn.executescript(SCHEMA)

    def __enter__(self) -> Database:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    # --- generic ---------------------------------------------------------------

    def _sql(self, sql: str) -> str:
        # Our SQL never contains a literal ? or %, so this translation is safe.
        return sql.replace("?", "%s") if self.is_postgres else sql

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        cur = self.conn.execute(self._sql(sql), tuple(params))
        rows = [dict(r) for r in cur.fetchall()]
        self.conn.commit()  # end the implicit transaction (Postgres would otherwise hold it open)
        return rows

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        self.conn.execute(self._sql(sql), tuple(params))
        self.conn.commit()

    def _one(self, sql: str, params: Sequence[Any]) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def _scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = self._one(sql, params)
        return next(iter(row.values())) if row else None

    def upsert(self, table: str, rows: Iterable[Mapping[str, Any]]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        keys = PRIMARY_KEYS[table]
        cols = list(rows[0].keys())
        non_keys = [c for c in cols if c not in keys]
        action = "DO UPDATE SET " + ", ".join(f"{c} = excluded.{c}" for c in non_keys) if non_keys else "DO NOTHING"
        sql = self._sql(
            f"INSERT INTO {table} ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' for _ in cols)}) "
            f"ON CONFLICT ({', '.join(keys)}) {action}"
        )
        values = [tuple(r[c] for c in cols) for r in rows]
        if self.is_postgres:
            with self.conn.transaction(), self.conn.cursor() as cur:
                cur.executemany(sql, values)
        else:
            with self.conn:
                self.conn.executemany(sql, values)
        return len(rows)

    def record_run(self, source: str, season: int, max_week: int | None, row_count: int) -> None:
        self.execute(
            "INSERT INTO ingest_runs (source, season, max_week, row_count) VALUES (?, ?, ?, ?)",
            (source, season, max_week, row_count),
        )

    def recent_runs(self, limit: int) -> list[dict[str, Any]]:
        return self.query(
            "SELECT source, season, max_week, row_count, finished_at FROM ingest_runs "
            "ORDER BY finished_at DESC, source LIMIT ?",
            (limit,),
        )

    # --- players ---------------------------------------------------------------

    def player(self, player_id: str) -> PlayerRow | None:
        return cast("PlayerRow | None", self._one("SELECT * FROM players WHERE player_id = ?", (player_id,)))

    def players_named(self, name: str, team: str | None = None) -> list[PlayerRow]:
        """Exact (case-insensitive) name matches, else substring matches; optionally on one team."""
        team_filter, team_params = (" AND upper(team) = upper(?)", (team,)) if team else ("", ())
        for pattern, op in ((name, "="), (f"%{name}%", "LIKE")):
            rows = self.query(
                f"SELECT * FROM players WHERE lower(name) {op} lower(?){team_filter} ORDER BY name, team",
                (pattern, *team_params),
            )
            if rows:
                return cast("list[PlayerRow]", rows)
        return []

    def roster_status(self, player_id: str, season: int, week: int) -> RosterRow | None:
        """The player's roster entry for `week`, else the latest one before it."""
        return cast(
            "RosterRow | None",
            self._one(
                "SELECT * FROM roster_weeks WHERE player_id = ? AND season = ? AND week <= ? "
                "ORDER BY week DESC LIMIT 1",
                (player_id, season, week),
            ),
        )

    def injury(self, player_id: str, season: int, week: int) -> InjuryRow | None:
        return cast(
            "InjuryRow | None",
            self._one(
                "SELECT * FROM injuries WHERE player_id = ? AND season = ? AND week = ?", (player_id, season, week)
            ),
        )

    def usage_before(self, player_id: str, season: int, week: int) -> UsageRow | None:
        """Usage features as of the latest week before `week` (no peeking at that week's game)."""
        return cast(
            "UsageRow | None",
            self._one(
                "SELECT * FROM usage_features WHERE player_id = ? AND season = ? AND through_week < ? "
                "ORDER BY through_week DESC LIMIT 1",
                (player_id, season, week),
            ),
        )

    # --- rankings --------------------------------------------------------------

    def latest_ranking(self, player_id: str, season: int, week: int) -> RankingRow | None:
        return cast(
            "RankingRow | None",
            self._one(
                "SELECT * FROM rankings WHERE player_id = ? AND season = ? AND week = ? "
                "ORDER BY scrape_date DESC LIMIT 1",
                (player_id, season, week),
            ),
        )

    def latest_rankings_week(self, season: int) -> int | None:
        return cast("int | None", self._scalar("SELECT max(week) AS w FROM rankings WHERE season = ?", (season,)))

    # --- schedule and stats ------------------------------------------------------

    def latest_season(self) -> int | None:
        return cast("int | None", self._scalar("SELECT max(season) AS s FROM player_weeks"))

    def latest_played_week(self, season: int) -> int | None:
        return cast("int | None", self._scalar("SELECT max(week) AS w FROM player_weeks WHERE season = ?", (season,)))

    def week_has_games(self, season: int, week: int) -> bool:
        return self._one("SELECT 1 AS x FROM games WHERE season = ? AND week = ? LIMIT 1", (season, week)) is not None

    def team_games(self, season: int, team: str, after_week: int, through_week: int) -> int:
        """Games `team` played (or plays) in weeks after_week+1 .. through_week."""
        return int(
            self._scalar(
                "SELECT count(*) AS n FROM games WHERE season = ? AND week > ? AND week <= ? "
                "AND (home_team = ? OR away_team = ?)",
                (season, after_week, through_week, team, team),
            )
        )

    def missed_games(self, player_id: str, season: int, after_week: int, through_week: int) -> int | None:
        """Games the player's team (per that week's roster) played in weeks after_week+1 .. through_week.

        None if there are no roster rows for those weeks, so the caller can fall back.
        """
        row = self._one(
            "SELECT count(*) AS weeks, count(g.game_id) AS games FROM roster_weeks r "
            "LEFT JOIN games g ON g.season = r.season AND g.week = r.week "
            "AND (g.home_team = r.team OR g.away_team = r.team) "
            "WHERE r.player_id = ? AND r.season = ? AND r.week > ? AND r.week <= ?",
            (player_id, season, after_week, through_week),
        )
        if row is None or row["weeks"] == 0:
            return None
        return int(row["games"])
