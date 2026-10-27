"""Data-access layer: the only module that contains SQL.

Two backends behind one class: SQLite (a file path, for local work and tests) and
Postgres (a postgres:// URL, i.e. Supabase). SQL is written once with `?` placeholders
and only uses syntax both accept; the Postgres backend translates the placeholders.
Callers use the named query methods below, so they never depend on the backend.

Each table is defined once, as a TypedDict of its columns plus its primary key (`Table`).
The CREATE TABLE statement, the polars schema used to build frames for it, and the
typed rows the query methods return all come from that one definition.

polars is imported only when a polars schema is asked for (by the ingest jobs), so the web
app runs without it.
"""

from __future__ import annotations

import os
import sqlite3
import types
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, TypedDict, cast, get_args, get_type_hints

if TYPE_CHECKING:
    import polars as pl

DEFAULT_DB_PATH = "data/startsit.db"


# --- table definitions -------------------------------------------------------------


class PlayerRow(TypedDict):
    player_id: str  # nflverse GSIS id
    name: str
    position: str
    team: str | None  # latest roster team
    pfr_id: str | None


class RosterRow(TypedDict):
    player_id: str
    season: int
    week: int
    team: str | None
    status: str | None  # ACT, RES (IR/PUP/NFI), DEV (practice squad), CUT, RET, INA, EXE, ...


class PlayerWeekRow(TypedDict):
    player_id: str
    season: int
    week: int
    team: str | None
    opponent: str | None
    offense_snaps: int | None
    snap_pct: float | None
    targets: int | None
    carries: int | None
    receptions: int | None
    target_share: float | None
    air_yards_share: float | None
    wopr: float | None
    fp_ppr: float | None
    fp_half: float | None
    fp_std: float | None
    xfp_ppr: float | None  # expected points from usage (FF Opportunity)
    xfp_half: float | None
    xfp_std: float | None


class InjuryRow(TypedDict):
    """Overwritten as reports update; recommendations must store the packet they used."""

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
    last_game_week: int  # features go stale if this lags through_week
    snap_pct_l3: float | None
    snap_pct_trend: float | None
    target_share_l3: float | None
    opportunities_l3: float | None
    fp_ppr_l3: float
    fp_half_l3: float
    fp_std_l3: float
    xfp_ppr_l3: float
    xfp_half_l3: float
    xfp_std_l3: float
    xfp_ppr_season: float
    fp_minus_xfp_ppr_l3: float


class GameRow(TypedDict):
    game_id: str
    season: int
    week: int
    gameday: str | None  # YYYY-MM-DD, US Eastern
    gametime: str | None  # kickoff, US Eastern "HH:MM"
    home_team: str
    away_team: str
    spread_line: float | None  # betting spread, home team's view: positive = home favored by that many
    total_line: float | None  # betting over/under for the game's combined points


class RankingRow(TypedDict):
    """FantasyPros weekly ECR, one archived copy per scrape (see startsit.ingest.rankings)."""

    season: int
    week: int
    scrape_date: str
    fantasypros_id: str
    player_id: str | None  # GSIS id; null for the few players the id map lacks
    player_name: str
    position: str
    team: str | None  # FantasyPros abbreviations (LAR, not nflverse's LA)
    opponent: str | None  # e.g. "vs. DAL", "at LV"
    ecr: float | None  # average expert rank within position
    ecr_sd: float | None
    ecr_best: float | None
    ecr_worst: float | None
    proj_ppr: float | None  # FantasyPros rank-to-points, PPR scoring
    pos_rank: str | None
    start_sit_grade: str | None
    game_status: str | None
    source_url: str | None


def utc_iso(dt: datetime) -> str:
    """The timestamp format news rows use: ISO 8601 in UTC to the second, so string order is time order."""
    return dt.astimezone(UTC).isoformat(timespec="seconds")


def utc_timestamp(dt: datetime) -> str:
    """The format of the TIMESTAMP log columns (llm_calls.called_at): 'YYYY-MM-DD HH:MM:SS' in UTC."""
    return dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


class NewsItemRow(TypedDict):
    """One headline from an RSS feed: headline, summary and link only (CLAUDE.md: link out, don't copy articles)."""

    item_id: str  # "<source>:<feed guid>", else the link (some feeds reuse one link per player)
    url: str
    source: str
    title: str
    summary: str | None
    published_at: str  # ISO 8601, UTC
    fetched_at: str


class NewsExtractionRow(TypedDict):
    """The L2 cache: one row per news item the LLM has processed, so no item is sent twice."""

    item_id: str
    model: str
    extracted_at: str
    fact_count: int


class NewsFactRow(TypedDict):
    """A structured, player-level fact extracted from one news item."""

    item_id: str
    player_id: str
    url: str  # the item's link, for citing the source
    source: str
    published_at: str
    event: str  # see startsit.ingest.news_extract.EVENTS
    severity: str  # high | medium | low
    summary: str  # restates only the item's own text
    game_status: str | None  # out | doubtful | questionable | active, when the item states one


class ProjectionRow(TypedDict):
    """FantasyPros per-format projections, one archived copy per fetch date (see startsit.ingest.projections)."""

    season: int
    week: int
    fetched_date: str
    fantasypros_id: str
    player_id: str | None
    points_ppr: float | None
    points_half: float | None
    points_std: float | None
    receptions: float | None  # projected catches


class DecisionCacheRow(TypedDict):
    """A validated LLM decision, keyed by a hash of the model, prompt version and exact packet text."""

    cache_key: str
    created_at: str
    model: str
    decision_json: str


class RecommendationRow(TypedDict):
    """Every answer shown to a user, with the exact evidence behind it, for the weekly evaluation.

    Injury reports and roster statuses are overwritten as they update, so the stored packet is
    the only record of what the user and the LLM saw.
    """

    rec_id: str
    created_at: str  # ISO 8601, UTC
    season: int
    week: int
    scoring: str
    player_ids: str  # JSON list, in the order asked
    pick: str
    baseline_pick: str
    confidence: str
    source: str  # llm | cache | baseline
    model: str | None
    decision_json: str
    packet_text: str  # exactly what the LLM saw (or would have)
    packet_json: str  # the packet's structure: per-player scores and evidence items with ids and links
    visitor: str | None  # salted hash of the IP; None outside the web app


class EvaluationRow(TypedDict):
    """A recommendation scored against what happened (step 6a). Hits are 1/0, or None for a tie."""

    rec_id: str
    scored_at: str  # ISO 8601, UTC
    season: int
    week: int
    points_json: str  # {player_id: actual points in the question's scoring}
    top_scorer: str | None  # None when the top score is tied
    pick_hit: int | None  # the pick shown (AI, or the baseline when the AI wasn't used)
    baseline_hit: int | None
    consensus_pick: str | None  # raw FantasyPros consensus; None when it can't choose
    consensus_hit: int | None


# Postgres REAL is single precision, so floats are DOUBLE PRECISION (same meaning in SQLite).
_SQL_TYPES: dict[type, str] = {str: "TEXT", int: "INTEGER", float: "DOUBLE PRECISION"}


def _column_types(row: type) -> dict[str, tuple[type, bool]]:
    """Column name -> (base type, nullable), from a TypedDict's annotations."""
    out = {}
    for name, hint in get_type_hints(row).items():
        args = get_args(hint)
        nullable = isinstance(hint, types.UnionType) and type(None) in args
        out[name] = (next(a for a in args if a is not type(None)) if nullable else hint, nullable)
    return out


@dataclass(frozen=True)
class Table:
    name: str
    row: type  # the TypedDict describing one row
    key: tuple[str, ...]

    @property
    def columns(self) -> list[str]:
        return list(get_type_hints(self.row))

    def ddl(self) -> str:
        cols = [
            f"    {name} {_SQL_TYPES[base]}{'' if nullable else ' NOT NULL'}"
            for name, (base, nullable) in _column_types(self.row).items()
        ]
        cols.append(f"    PRIMARY KEY ({', '.join(self.key)})")
        return f"CREATE TABLE IF NOT EXISTS {self.name} (\n" + ",\n".join(cols) + "\n);"

    def column_ddl(self, name: str) -> str:
        """Type clause for one column, for ALTER TABLE ... ADD COLUMN (always nullable)."""
        base, _ = _column_types(self.row)[name]
        return f"{name} {_SQL_TYPES[base]}"

    def polars_schema(self) -> dict[str, type[pl.DataType]]:
        import polars as pl

        types: dict[type, type[pl.DataType]] = {str: pl.String, int: pl.Int64, float: pl.Float64}
        return {name: types[base] for name, (base, _) in _column_types(self.row).items()}


TABLES: dict[str, Table] = {
    t.name: t
    for t in (
        Table("players", PlayerRow, ("player_id",)),
        Table("roster_weeks", RosterRow, ("player_id", "season", "week")),
        Table("player_weeks", PlayerWeekRow, ("player_id", "season", "week")),
        Table("injuries", InjuryRow, ("player_id", "season", "week")),
        Table("usage_features", UsageRow, ("player_id", "season", "through_week")),
        Table("games", GameRow, ("game_id",)),
        Table("rankings", RankingRow, ("season", "week", "scrape_date", "fantasypros_id")),
        Table("news_items", NewsItemRow, ("item_id",)),
        Table("news_extractions", NewsExtractionRow, ("item_id",)),
        Table("news_facts", NewsFactRow, ("item_id", "player_id")),
        Table("decision_cache", DecisionCacheRow, ("cache_key",)),
        Table("projections", ProjectionRow, ("season", "week", "fetched_date", "fantasypros_id")),
        Table("recommendations", RecommendationRow, ("rec_id",)),
        Table("evaluations", EvaluationRow, ("rec_id",)),
    )
}

# An append-only log with a database-side default, so it's written by hand.
INGEST_RUNS_DDL = """CREATE TABLE IF NOT EXISTS ingest_runs (
    source       TEXT NOT NULL,
    season       INTEGER NOT NULL,
    max_week     INTEGER,
    row_count    INTEGER NOT NULL,
    finished_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);"""

# Every LLM request, so free-tier limits (per day, per minute, per visitor) can be enforced.
# called_at is written by the caller (UTC), so limits use one clock.
LLM_CALLS_DDL = """CREATE TABLE IF NOT EXISTS llm_calls (
    purpose    TEXT NOT NULL,
    model      TEXT NOT NULL,
    ok         INTEGER NOT NULL,
    called_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    visitor    TEXT
);
CREATE INDEX IF NOT EXISTS llm_calls_called_at ON llm_calls (called_at);"""

# Columns added to the hand-written tables after they first shipped: (table, column, SQL type).
# Like TABLES columns, they're added to existing databases on connect and must be nullable.
LATER_COLUMNS = (("llm_calls", "visitor", "TEXT"),)

SCHEMA = "\n\n".join([*(t.ddl() for t in TABLES.values()), INGEST_RUNS_DDL, LLM_CALLS_DDL])
ALL_TABLE_NAMES = (*TABLES, "ingest_runs", "llm_calls")


# A games row (alias g) involving a team; the placeholder is the team expression.
_PLAYS = "(g.home_team = {0} OR g.away_team = {0})"


def resolve_target(target: str | Path | None = None) -> str:
    """Explicit target, else $DATABASE_URL, else $STARTSIT_DB_PATH, else the default file."""
    return str(target or os.environ.get("DATABASE_URL") or os.environ.get("STARTSIT_DB_PATH") or DEFAULT_DB_PATH)


def is_postgres(target: str) -> bool:
    return target.startswith(("postgres://", "postgresql://"))


class Database:
    """A connection. `setup` (default) creates missing tables and columns first; the web app does
    that once per process and skips it on later connections."""

    def __init__(self, target: str | Path | None = None, setup: bool = True):
        target = resolve_target(target)
        self.is_postgres = is_postgres(target)
        self.conn: Any
        if self.is_postgres:
            import psycopg
            from psycopg.rows import dict_row

            # prepare_threshold=None: no server-side prepared statements, which Supabase's
            # transaction pooler (used by the serverless web app) doesn't support.
            self.conn = psycopg.connect(target, row_factory=dict_row, prepare_threshold=None)
            if setup:
                self._setup_postgres()
        else:
            if target != ":memory:":
                Path(target).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(target)
            self.conn.row_factory = sqlite3.Row
            if setup or target == ":memory:":
                self.conn.executescript(SCHEMA)
        if setup:
            self._add_missing_columns()

    def _setup_postgres(self) -> None:
        with self.conn.transaction():
            self.conn.execute(SCHEMA)
            # Supabase also serves the public schema through its REST API (anon/authenticated
            # roles). Row-level security with no policies closes that door; the app connects as
            # the tables' owner, which RLS doesn't restrict.
            for table in ALL_TABLE_NAMES:
                self.conn.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")

    def _existing_columns(self) -> dict[str, set[str]]:
        """Columns of every table that exists, by table name."""
        if self.is_postgres:
            rows = self.query(
                "SELECT table_name AS t, column_name AS c FROM information_schema.columns "
                "WHERE table_schema = current_schema()"
            )
            out: dict[str, set[str]] = {}
            for r in rows:
                out.setdefault(r["t"], set()).add(r["c"])
            return out
        return {t: {r["name"] for r in self.query(f"PRAGMA table_info({t})")} for t in ALL_TABLE_NAMES}

    def _add_missing_columns(self) -> None:
        """Schema migration for added columns: CREATE TABLE IF NOT EXISTS leaves existing tables alone.

        Columns added this way must be nullable in the row type, since existing rows have no value.
        """
        existing = self._existing_columns()
        for table in TABLES.values():
            missing = [c for c in table.columns if c not in existing.get(table.name, set())]
            for column in missing:
                if not _column_types(table.row)[column][1]:
                    raise RuntimeError(
                        f"new column {table.name}.{column} must be nullable to be added to existing rows"
                    )
                self.execute(f"ALTER TABLE {table.name} ADD COLUMN {table.column_ddl(column)}")
        for name, column, sql_type in LATER_COLUMNS:
            if column not in existing.get(name, set()):
                self.execute(f"ALTER TABLE {name} ADD COLUMN {column} {sql_type}")

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
        keys = TABLES[table].key
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

    def record_llm_call(
        self, purpose: str, model: str, ok: bool, called_at: datetime, visitor: str | None = None
    ) -> None:
        self.execute(
            "INSERT INTO llm_calls (purpose, model, ok, called_at, visitor) VALUES (?, ?, ?, ?, ?)",
            (purpose, model, int(ok), utc_timestamp(called_at), visitor),
        )

    def llm_calls_since(self, since: datetime, purpose: str | None = None, visitor: str | None = None) -> int:
        """LLM requests made at or after `since`, optionally only for one purpose and/or one visitor."""
        sql, params = "SELECT count(*) AS n FROM llm_calls WHERE called_at >= ?", [utc_timestamp(since)]
        for column, value in (("purpose", purpose), ("visitor", visitor)):
            if value is not None:
                sql += f" AND {column} = ?"
                params.append(value)
        return int(self._scalar(sql, params))

    # --- news --------------------------------------------------------------------

    def unextracted_news(self, since: str, limit: int) -> list[NewsItemRow]:
        """News items published at or after `since` that the LLM hasn't processed yet, newest first."""
        return cast(
            "list[NewsItemRow]",
            self.query(
                "SELECT i.* FROM news_items i LEFT JOIN news_extractions e ON e.item_id = i.item_id "
                "WHERE e.item_id IS NULL AND i.published_at >= ? ORDER BY i.published_at DESC LIMIT ?",
                (since, limit),
            ),
        )

    def news_for_player(self, player_id: str, since: str, until: str, limit: int) -> list[NewsFactRow]:
        """The player's news facts published in [since, until], newest first."""
        return cast(
            "list[NewsFactRow]",
            self.query(
                "SELECT * FROM news_facts WHERE player_id = ? AND published_at >= ? AND published_at <= ? "
                "ORDER BY published_at DESC LIMIT ?",
                (player_id, since, until, limit),
            ),
        )

    def latest_news_status(self, player_id: str, since: str, until: str) -> NewsFactRow | None:
        """The newest news fact in [since, until] that states the player's game status."""
        return cast(
            "NewsFactRow | None",
            self._one(
                "SELECT * FROM news_facts WHERE player_id = ? AND published_at >= ? AND published_at <= ? "
                "AND game_status IS NOT NULL ORDER BY published_at DESC LIMIT 1",
                (player_id, since, until),
            ),
        )

    # --- players ---------------------------------------------------------------

    def all_players(self) -> list[PlayerRow]:
        return cast("list[PlayerRow]", self.query("SELECT * FROM players"))

    def player(self, player_id: str) -> PlayerRow | None:
        return cast("PlayerRow | None", self._one("SELECT * FROM players WHERE player_id = ?", (player_id,)))

    def _player_row(
        self, table: str, player_id: str, season: int, condition: str, value: int, order_by: str | None = None
    ) -> dict[str, Any] | None:
        """One row of `table` for the player and season matching `condition` (newest first by `order_by`)."""
        order = f" ORDER BY {order_by} DESC" if order_by else ""
        return self._one(
            f"SELECT * FROM {table} WHERE player_id = ? AND season = ? AND {condition}{order} LIMIT 1",
            (player_id, season, value),
        )

    def roster_status(self, player_id: str, season: int, week: int) -> RosterRow | None:
        """The player's roster entry for `week`, else the latest one before it."""
        return cast("RosterRow | None", self._player_row("roster_weeks", player_id, season, "week <= ?", week, "week"))

    def injury(self, player_id: str, season: int, week: int) -> InjuryRow | None:
        return cast("InjuryRow | None", self._player_row("injuries", player_id, season, "week = ?", week))

    def usage_before(self, player_id: str, season: int, week: int) -> UsageRow | None:
        """Usage features as of the latest week before `week` (no peeking at that week's game)."""
        return cast(
            "UsageRow | None",
            self._player_row("usage_features", player_id, season, "through_week < ?", week, "through_week"),
        )

    # --- rankings --------------------------------------------------------------

    # `through_day` (YYYY-MM-DD, UTC) limits these to archived copies from that day or earlier, so a
    # question asked in the past (the benchmark backfill) sees only what existed then.

    def _latest_copy(
        self, table: str, date_column: str, player_id: str, season: int, week: int, through_day: str | None
    ) -> dict[str, Any] | None:
        return self._one(
            f"SELECT * FROM {table} WHERE player_id = ? AND season = ? AND week = ? AND {date_column} <= ? "
            f"ORDER BY {date_column} DESC LIMIT 1",
            (player_id, season, week, through_day or "9999-12-31"),
        )

    def latest_ranking(
        self, player_id: str, season: int, week: int, through_day: str | None = None
    ) -> RankingRow | None:
        return cast(
            "RankingRow | None", self._latest_copy("rankings", "scrape_date", player_id, season, week, through_day)
        )

    def rankings_for_week(self, season: int, week: int, through_day: str | None = None) -> list[RankingRow]:
        """Every player's ranking from the latest scrape of the week (on or before `through_day`)."""
        day = through_day or "9999-12-31"
        return cast(
            "list[RankingRow]",
            self.query(
                "SELECT * FROM rankings WHERE season = ? AND week = ? AND scrape_date = "
                "(SELECT max(scrape_date) FROM rankings WHERE season = ? AND week = ? AND scrape_date <= ?)",
                (season, week, season, week, day),
            ),
        )

    def latest_projection(
        self, player_id: str, season: int, week: int, through_day: str | None = None
    ) -> ProjectionRow | None:
        return cast(
            "ProjectionRow | None",
            self._latest_copy("projections", "fetched_date", player_id, season, week, through_day),
        )

    def latest_rankings_week(self, season: int) -> int | None:
        return cast("int | None", self._scalar("SELECT max(week) AS w FROM rankings WHERE season = ?", (season,)))

    # --- schedule and stats ------------------------------------------------------

    def games(self, season: int) -> list[GameRow]:
        return cast("list[GameRow]", self.query("SELECT * FROM games WHERE season = ? ORDER BY game_id", (season,)))

    def latest_season(self) -> int | None:
        return cast("int | None", self._scalar("SELECT max(season) AS s FROM player_weeks"))

    def latest_played_week(self, season: int) -> int | None:
        return cast("int | None", self._scalar("SELECT max(week) AS w FROM player_weeks WHERE season = ?", (season,)))

    def team_game(self, season: int, week: int, team: str) -> GameRow | None:
        return cast(
            "GameRow | None",
            self._one(
                f"SELECT * FROM games g WHERE season = ? AND week = ? AND {_PLAYS.format('?')}",
                (season, week, team, team),
            ),
        )

    def team_last_game(self, season: int, team: str, before_week: int) -> GameRow | None:
        """The team's latest game before `before_week`."""
        return cast(
            "GameRow | None",
            self._one(
                f"SELECT * FROM games g WHERE season = ? AND week < ? AND {_PLAYS.format('?')} "
                "ORDER BY week DESC LIMIT 1",
                (season, before_week, team, team),
            ),
        )

    def week_has_games(self, season: int, week: int) -> bool:
        return self._one("SELECT 1 AS x FROM games WHERE season = ? AND week = ? LIMIT 1", (season, week)) is not None

    def team_games(self, season: int, team: str, after_week: int, through_week: int) -> int:
        """Games `team` played (or plays) in weeks after_week+1 .. through_week."""
        return int(
            self._scalar(
                "SELECT count(*) AS n FROM games g WHERE season = ? AND week > ? AND week <= ? "
                f"AND {_PLAYS.format('?')}",
                (season, after_week, through_week, team, team),
            )
        )

    def missed_games(self, player_id: str, season: int, after_week: int, through_week: int) -> int | None:
        """Games the player's team (per that week's roster) played in weeks after_week+1 .. through_week.

        None if there are no roster rows for those weeks, so the caller can fall back.
        """
        row = self._one(
            "SELECT count(*) AS weeks, count(g.game_id) AS games FROM roster_weeks r "
            f"LEFT JOIN games g ON g.season = r.season AND g.week = r.week AND {_PLAYS.format('r.team')} "
            "WHERE r.player_id = ? AND r.season = ? AND r.week > ? AND r.week <= ?",
            (player_id, season, after_week, through_week),
        )
        if row is None or row["weeks"] == 0:
            return None
        return int(row["games"])

    # --- decisions -----------------------------------------------------------------

    def questions_asked(self, season: int, week: int, visitors: Sequence[str]) -> list[dict[str, Any]]:
        """The player ids (JSON) and scoring of each recommendation logged for these visitors that week."""
        marks = ", ".join("?" for _ in visitors)
        return self.query(
            f"SELECT player_ids, scoring FROM recommendations WHERE season = ? AND week = ? AND visitor IN ({marks})",
            (season, week, *visitors),
        )

    def has_run(self, sources: Sequence[str], season: int, week: int) -> bool:
        """Whether `ingest_runs` records any of these sources for that week."""
        marks = ", ".join("?" for _ in sources)
        return (
            self._one(
                f"SELECT 1 AS x FROM ingest_runs WHERE source IN ({marks}) AND season = ? AND max_week = ? LIMIT 1",
                (*sources, season, week),
            )
            is not None
        )

    def cached_decision(self, cache_key: str) -> DecisionCacheRow | None:
        return cast(
            "DecisionCacheRow | None", self._one("SELECT * FROM decision_cache WHERE cache_key = ?", (cache_key,))
        )

    # --- evaluation ---------------------------------------------------------------------

    def week_games(self, season: int, week: int) -> list[GameRow]:
        return cast("list[GameRow]", self.query("SELECT * FROM games WHERE season = ? AND week = ?", (season, week)))

    def player_week(self, player_id: str, season: int, week: int) -> PlayerWeekRow | None:
        return cast("PlayerWeekRow | None", self._player_row("player_weeks", player_id, season, "week = ?", week))

    def team_has_stats(self, season: int, week: int, team: str) -> bool:
        """Whether any player's stats for `team` in `week` are loaded (so its game is in the data)."""
        return (
            self._one(
                "SELECT 1 AS x FROM player_weeks WHERE season = ? AND week = ? AND team = ? LIMIT 1",
                (season, week, team),
            )
            is not None
        )

    def unscored_recommendations(self) -> list[RecommendationRow]:
        return cast(
            "list[RecommendationRow]",
            self.query(
                "SELECT r.* FROM recommendations r LEFT JOIN evaluations e ON e.rec_id = r.rec_id "
                "WHERE e.rec_id IS NULL ORDER BY r.created_at"
            ),
        )

    def scored_recommendations(self, season: int) -> list[dict[str, Any]]:
        """Every scored recommendation of the season: the evaluation joined with what was shown."""
        return self.query(
            "SELECT e.*, r.scoring, r.pick, r.baseline_pick, r.confidence, r.source, r.visitor, r.packet_json "
            "FROM evaluations e JOIN recommendations r ON r.rec_id = e.rec_id "
            "WHERE e.season = ? ORDER BY e.week, r.created_at",
            (season,),
        )

    # --- web app ---------------------------------------------------------------------

    def latest_scrape_date(self, season: int, week: int) -> str | None:
        """When the week's rankings were scraped (YYYY-MM-DD)."""
        return cast(
            "str | None",
            self._scalar("SELECT max(scrape_date) AS d FROM rankings WHERE season = ? AND week = ?", (season, week)),
        )

    def week_gamedays(self, season: int, week: int) -> tuple[str, str] | None:
        """The week's first and last game days (YYYY-MM-DD, US Eastern), e.g. Thursday and Monday."""
        row = self._one(
            "SELECT min(gameday) AS first, max(gameday) AS last FROM games WHERE season = ? AND week = ?",
            (season, week),
        )
        return (row["first"], row["last"]) if row and row["first"] else None

    def latest_news_fetch(self) -> str | None:
        """When news feeds were last fetched (ISO 8601, UTC)."""
        return cast("str | None", self._scalar("SELECT max(fetched_at) AS t FROM news_items"))

    def latest_stats_run(self) -> str | None:
        """When nflverse stats were last ingested ('YYYY-MM-DD HH:MM:SS', UTC)."""
        value = self._scalar("SELECT max(finished_at) AS t FROM ingest_runs WHERE source LIKE ?", ("nflverse:%",))
        return str(value)[:19] if value is not None else None
