"""nflverse ingest: weekly usage, fantasy points, expected points, rosters, injury reports, schedule.

Fetching is kept separate from the transforms so the transforms can be tested with
small hand-built frames and no network.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from startsit.db import TABLES
from startsit.scoring import SCORINGS, column, from_ppr

SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]
# nflverse's defensive positions (player stats and snap counts label them differently).
DEFENSE_POSITIONS = ["DE", "DT", "NT", "DL", "EDGE", "LB", "ILB", "MLB", "OLB", "CB", "DB", "S", "SAF", "FS", "SS"]
# Season stats kept per defender: player_stats column -> defender_weeks column.
DEFENSE_STATS = {
    "def_sacks": "sacks",
    "def_qb_hits": "qb_hits",
    "def_interceptions": "interceptions",
    "def_pass_defended": "pass_defended",
    "def_tackles_for_loss": "tackles_for_loss",
}
PLAYER_WEEK = ["player_id", "season", "week"]

COUNT_COLS = ["targets", "carries", "receptions"]
SHARE_COLS = ["target_share", "air_yards_share", "wopr"]


def _regular_season_skill(
    frame: pl.DataFrame, type_col: str = "game_type", id_col: str | None = "gsis_id"
) -> pl.DataFrame:
    """Regular-season rows for QB/RB/WR/TE (with a player id, when `id_col` is given)."""
    cond = (pl.col(type_col) == "REG") & pl.col("position").is_in(SKILL_POSITIONS)
    if id_col:
        cond &= pl.col(id_col).is_not_null()
    return frame.filter(cond)


def _season_week() -> list[pl.Expr]:
    """season and week as Int64 (sources disagree: Int32, Float64, ...)."""
    return [pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64)]


@dataclass
class RawNflverse:
    player_stats: pl.DataFrame
    snap_counts: pl.DataFrame
    ff_opportunity: pl.DataFrame
    injuries: pl.DataFrame
    players: pl.DataFrame
    rosters: pl.DataFrame
    schedules: pl.DataFrame


def fetch(season: int) -> RawNflverse:
    import nflreadpy as nfl

    return RawNflverse(
        player_stats=nfl.load_player_stats([season]),
        snap_counts=nfl.load_snap_counts([season]),
        ff_opportunity=nfl.load_ff_opportunity([season]),
        injuries=nfl.load_injuries([season]),
        players=nfl.load_players(),
        rosters=nfl.load_rosters_weekly([season]),
        schedules=nfl.load_schedules([season]),
    )


def build_player_weeks(raw: RawNflverse) -> pl.DataFrame:
    """One row per (player, season, week) for regular-season skill-position players.

    Snap counts are keyed by PFR id, so they are mapped to GSIS ids via the players
    table. A player who took snaps but recorded no stat still gets a row (0 points),
    because that zero is a real usage signal.
    """
    id_map = raw.players.select("gsis_id", "pfr_id").drop_nulls().unique("pfr_id")

    snaps = (
        _regular_season_skill(raw.snap_counts, id_col="pfr_player_id")
        .join(id_map, left_on="pfr_player_id", right_on="pfr_id", how="inner")
        .select(
            pl.col("gsis_id").alias("player_id"),
            *_season_week(),
            pl.col("team").alias("snap_team"),
            pl.col("opponent").alias("snap_opponent"),
            pl.col("offense_snaps").cast(pl.Int64),
            pl.col("offense_pct").alias("snap_pct"),
        )
    )

    stats = _regular_season_skill(raw.player_stats, type_col="season_type", id_col="player_id").select(
        "player_id",
        *_season_week(),
        "team",
        pl.col("opponent_team").alias("opponent"),
        *COUNT_COLS,
        *SHARE_COLS,
        pl.col("fantasy_points").alias("fp_std"),
        pl.col("fantasy_points_ppr").alias("fp_ppr"),
    )

    # FF Opportunity's expected points are PPR-scored; other formats remove expected receptions.
    total_exp, receptions_exp = pl.col("total_fantasy_points_exp"), pl.col("receptions_exp")
    xfp = raw.ff_opportunity.filter(pl.col("player_id").is_not_null()).select(
        "player_id",
        *_season_week(),
        *[from_ppr(total_exp, receptions_exp, s).alias(column("xfp", s)) for s in SCORINGS],
    )

    weeks = (
        stats.join(snaps, on=PLAYER_WEEK, how="full", coalesce=True)
        .join(xfp, on=PLAYER_WEEK, how="left")
        .with_columns(
            pl.coalesce("team", "snap_team").alias("team"),
            pl.coalesce("opponent", "snap_opponent").alias("opponent"),
            pl.col("fp_std").fill_null(0.0),
            pl.col("fp_ppr").fill_null(0.0),
        )
        .with_columns(
            *[pl.col(c).fill_null(0) for c in COUNT_COLS],
            *[pl.col(c).fill_null(0.0) for c in SHARE_COLS],
            *[pl.col(column("xfp", s)).fill_null(0.0) for s in SCORINGS],
        )
        .with_columns(from_ppr(pl.col("fp_ppr"), pl.col("receptions"), "half").alias("fp_half"))
    )
    return weeks.select(TABLES["player_weeks"].columns).sort(PLAYER_WEEK)


def _skill_or(frame: pl.DataFrame, defender_ids: pl.Series | None) -> pl.DataFrame:
    """Regular-season rows (with a GSIS id) for QB/RB/WR/TE and, if given, these defenders."""
    keep = pl.col("position").is_in(SKILL_POSITIONS)
    if defender_ids is not None:
        keep |= pl.col("gsis_id").is_in(defender_ids.implode())
    return frame.filter((pl.col("game_type") == "REG") & pl.col("gsis_id").is_not_null() & keep)


def build_defender_weeks(raw: RawNflverse) -> pl.DataFrame:
    """One row per (defender, season, week) he was on the field: defensive snaps and playmaking stats.

    Snap counts (by PFR id, mapped to GSIS ids via the players table) say who starts; player stats
    say who makes plays. A defender with stats but no mapped snap count gets 0 snaps.
    """
    id_map = raw.players.select("gsis_id", "pfr_id").drop_nulls().unique("pfr_id")
    snaps = (
        raw.snap_counts.filter((pl.col("game_type") == "REG") & pl.col("position").is_in(DEFENSE_POSITIONS))
        .join(id_map, left_on="pfr_player_id", right_on="pfr_id", how="inner")
        .select(
            pl.col("gsis_id").alias("player_id"),
            *_season_week(),
            pl.col("team").alias("snap_team"),
            pl.col("defense_snaps").cast(pl.Int64),
            pl.col("defense_pct").cast(pl.Float64),
        )
        .filter(pl.col("defense_snaps") > 0)
    )
    stats = raw.player_stats.filter(
        (pl.col("season_type") == "REG")
        & pl.col("position").is_in(DEFENSE_POSITIONS)
        & pl.col("player_id").is_not_null()
    ).select("player_id", *_season_week(), "team", *[pl.col(c).alias(n) for c, n in DEFENSE_STATS.items()])
    return (
        snaps.join(stats, on=PLAYER_WEEK, how="full", coalesce=True)
        .with_columns(
            pl.coalesce("snap_team", "team").alias("team"),
            pl.col("defense_snaps").fill_null(0),
            pl.col("defense_pct").fill_null(0.0),
            pl.col("sacks").cast(pl.Float64).fill_null(0.0),
            *[pl.col(c).cast(pl.Int64).fill_null(0) for c in list(DEFENSE_STATS.values())[1:]],
        )
        .select(TABLES["defender_weeks"].columns)
        .sort(PLAYER_WEEK)
    )


def build_defenders(raw: RawNflverse, defender_weeks: pl.DataFrame) -> pl.DataFrame:
    """The defenders in `defender_weeks`: name and position from the players table, team of the latest game."""
    latest = (
        defender_weeks.sort("season", "week")
        .group_by("player_id")
        .agg(pl.col("team").drop_nulls().last().alias("team"))
    )
    master = raw.players.select(
        pl.col("gsis_id").alias("player_id"), pl.col("display_name").alias("name"), "position"
    ).unique("player_id")
    return (
        latest.join(master, on="player_id", how="inner")
        .filter(pl.col("name").is_not_null() & pl.col("position").is_not_null())
        .select(TABLES["defenders"].columns)
        .sort("player_id")
    )


def build_roster_weeks(raw: RawNflverse, defender_ids: pl.Series | None = None) -> pl.DataFrame:
    """Each player's team and roster status per week (ACT, RES, DEV, CUT, RET, ...), for skill players
    and the given defenders."""
    return (
        _skill_or(raw.rosters, defender_ids)
        .select(
            pl.col("gsis_id").alias("player_id"),
            *_season_week(),
            "team",
            "status",
        )
        .unique(PLAYER_WEEK, keep="last")
        .sort(PLAYER_WEEK)
    )


def build_players(raw: RawNflverse, player_weeks: pl.DataFrame) -> pl.DataFrame:
    """Skill-position players on a roster this season or with a game this season.

    Rosters are included so players who haven't played yet (rookies, returns from
    injury) can still be asked about. The team is the latest roster team, which follows
    trades and releases; the team of the last game is only a fallback.
    """
    from_roster = (
        _regular_season_skill(raw.rosters)
        .sort("week")
        .group_by("gsis_id")
        .agg(
            pl.col("full_name").last().alias("roster_name"),
            pl.col("position").last().alias("roster_position"),
            pl.col("team").drop_nulls().last().alias("roster_team"),
            pl.col("pfr_id").drop_nulls().last().alias("roster_pfr_id"),
        )
        .rename({"gsis_id": "player_id"})
    )
    game_team = (
        player_weeks.sort("week").group_by("player_id").agg(pl.col("team").drop_nulls().last().alias("game_team"))
    )
    master = raw.players.select(
        pl.col("gsis_id").alias("player_id"), "display_name", pl.col("position").alias("master_position"), "pfr_id"
    ).unique("player_id")
    ids = pl.concat([from_roster.select("player_id"), player_weeks.select("player_id")]).unique()
    # Position: the master table's (it can list e.g. FB for a back who plays RB), else the roster's.
    return (
        ids.join(from_roster, on="player_id", how="left")
        .join(master, on="player_id", how="left")
        .join(game_team, on="player_id", how="left")
        .select(
            "player_id",
            pl.coalesce("display_name", "roster_name").alias("name"),
            pl.coalesce("master_position", "roster_position", pl.lit("UNK")).alias("position"),
            pl.coalesce("roster_team", "game_team").alias("team"),
            pl.coalesce("pfr_id", "roster_pfr_id").alias("pfr_id"),
        )
        .filter(pl.col("name").is_not_null())
        .sort("player_id")
    )


def build_injuries(raw: RawNflverse, defender_ids: pl.Series | None = None) -> pl.DataFrame:
    """Weekly injury reports for skill players and the given defenders."""
    return (
        _skill_or(raw.injuries, defender_ids)
        .select(
            pl.col("gsis_id").alias("player_id"),
            *_season_week(),
            "team",
            "report_status",
            "practice_status",
            pl.coalesce("report_primary_injury", "practice_primary_injury").alias("injury"),
        )
        .unique(PLAYER_WEEK, keep="last")
        .sort(PLAYER_WEEK)
    )


def build_games(raw: RawNflverse) -> pl.DataFrame:
    """Regular-season games with betting lines; also how byes are detected (a team with no game that week)."""
    return (
        raw.schedules.filter(pl.col("game_type") == "REG")
        .select(
            "game_id",
            *_season_week(),
            pl.col("gameday").cast(pl.String),
            pl.col("gametime").cast(pl.String),
            "home_team",
            "away_team",
            pl.col("spread_line").cast(pl.Float64),
            pl.col("total_line").cast(pl.Float64),
        )
        .sort("game_id")
    )
