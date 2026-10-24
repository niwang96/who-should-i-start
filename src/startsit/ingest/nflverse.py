"""nflverse ingest: weekly usage, fantasy points, expected points, injury reports.

Fetching is kept separate from the transforms so the transforms can be tested with
small hand-built frames and no network.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]

COUNT_COLS = ["targets", "carries", "receptions"]
SHARE_COLS = ["target_share", "air_yards_share", "wopr"]
POINT_COLS = ["fp_std", "fp_half", "fp_ppr", "xfp_std", "xfp_half", "xfp_ppr"]


@dataclass
class RawNflverse:
    player_stats: pl.DataFrame
    snap_counts: pl.DataFrame
    ff_opportunity: pl.DataFrame
    injuries: pl.DataFrame
    players: pl.DataFrame


def fetch(season: int) -> RawNflverse:
    import nflreadpy as nfl

    return RawNflverse(
        player_stats=nfl.load_player_stats([season]),
        snap_counts=nfl.load_snap_counts([season]),
        ff_opportunity=nfl.load_ff_opportunity([season]),
        injuries=nfl.load_injuries([season]),
        players=nfl.load_players(),
    )


def build_player_weeks(raw: RawNflverse) -> pl.DataFrame:
    """One row per (player, season, week) for regular-season skill-position players.

    Snap counts are keyed by PFR id, so they are mapped to GSIS ids via the players
    table. A player who took snaps but recorded no stat still gets a row (0 points),
    because that zero is a real usage signal.
    """
    id_map = raw.players.select("gsis_id", "pfr_id").drop_nulls().unique("pfr_id")

    snaps = (
        raw.snap_counts.filter(
            (pl.col("game_type") == "REG") & pl.col("position").is_in(SKILL_POSITIONS)
        )
        .join(id_map, left_on="pfr_player_id", right_on="pfr_id", how="inner")
        .select(
            pl.col("gsis_id").alias("player_id"),
            "season",
            "week",
            pl.col("team").alias("snap_team"),
            pl.col("opponent").alias("snap_opponent"),
            pl.col("offense_snaps").cast(pl.Int64),
            pl.col("offense_pct").alias("snap_pct"),
        )
    )

    stats = raw.player_stats.filter(
        (pl.col("season_type") == "REG") & pl.col("position").is_in(SKILL_POSITIONS)
    ).select(
        "player_id",
        "season",
        "week",
        "team",
        pl.col("opponent_team").alias("opponent"),
        *COUNT_COLS,
        *SHARE_COLS,
        pl.col("fantasy_points").alias("fp_std"),
        pl.col("fantasy_points_ppr").alias("fp_ppr"),
    )

    # FF Opportunity's expected points are PPR-scored; derive the other formats by
    # removing expected receptions.
    xfp = raw.ff_opportunity.filter(pl.col("player_id").is_not_null()).select(
        "player_id",
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
        pl.col("total_fantasy_points_exp").alias("xfp_ppr"),
        (pl.col("total_fantasy_points_exp") - pl.col("receptions_exp")).alias("xfp_std"),
        (pl.col("total_fantasy_points_exp") - 0.5 * pl.col("receptions_exp")).alias("xfp_half"),
    )

    keys = ["player_id", "season", "week"]
    weeks = (
        stats.with_columns(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64))
        .join(
            snaps.with_columns(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64)),
            on=keys,
            how="full",
            coalesce=True,
        )
        .join(xfp, on=keys, how="left")
        .with_columns(
            pl.coalesce("team", "snap_team").alias("team"),
            pl.coalesce("opponent", "snap_opponent").alias("opponent"),
            pl.col("fp_std").fill_null(0.0),
            pl.col("fp_ppr").fill_null(0.0),
        )
        .with_columns(
            ((pl.col("fp_std") + pl.col("fp_ppr")) / 2).alias("fp_half"),
            *[pl.col(c).fill_null(0) for c in COUNT_COLS],
            *[pl.col(c).fill_null(0.0) for c in SHARE_COLS],
            *[pl.col(c).fill_null(0.0) for c in ("xfp_std", "xfp_half", "xfp_ppr")],
        )
        .drop("snap_team", "snap_opponent")
    )
    return weeks.select(
        *keys,
        "team",
        "opponent",
        "offense_snaps",
        "snap_pct",
        *COUNT_COLS,
        *SHARE_COLS,
        *POINT_COLS,
    ).sort(keys)


def build_players(raw: RawNflverse, player_weeks: pl.DataFrame) -> pl.DataFrame:
    """Players who appear in this season's weekly data, with their latest team."""
    latest_team = (
        player_weeks.sort("week")
        .group_by("player_id")
        .agg(pl.col("team").drop_nulls().last().alias("current_team"))
    )
    # No position filter here: player_weeks is already limited to skill positions, and
    # the players table can list a different one (e.g. FB for a back who plays RB).
    return (
        raw.players.select(
            pl.col("gsis_id").alias("player_id"),
            pl.col("display_name").alias("name"),
            pl.col("position").fill_null("UNK"),
            pl.col("latest_team").alias("team"),
            "pfr_id",
        )
        .unique("player_id")
        .join(latest_team, on="player_id", how="inner")
        .with_columns(pl.coalesce("current_team", "team").alias("team"))
        .drop("current_team")
        .sort("player_id")
    )


def build_injuries(raw: RawNflverse) -> pl.DataFrame:
    return (
        raw.injuries.filter(
            (pl.col("game_type") == "REG")
            & pl.col("position").is_in(SKILL_POSITIONS)
            & pl.col("gsis_id").is_not_null()
        )
        .select(
            pl.col("gsis_id").alias("player_id"),
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            "team",
            "report_status",
            "practice_status",
            pl.coalesce("report_primary_injury", "practice_primary_injury").alias("injury"),
        )
        .unique(["player_id", "season", "week"], keep="last")
        .sort(["player_id", "season", "week"])
    )
