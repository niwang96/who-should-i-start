"""Usage features computed from player_weeks.

Features are computed as of every week ("through_week"), not just the latest, so the
weekly evaluation can score a recommendation against the data that existed when it
was made.
"""

from __future__ import annotations

import polars as pl

WINDOW = 3


def _features_through(weeks: pl.DataFrame, through_week: int) -> pl.DataFrame:
    hist = weeks.filter(pl.col("week") <= through_week).sort("week")
    recent = hist.group_by("player_id", "season").tail(WINDOW)
    prior = (
        hist.join(recent.select("player_id", "season", "week"), on=["player_id", "season", "week"], how="anti")
        .group_by("player_id", "season")
        .tail(WINDOW)
    )

    l3 = recent.group_by("player_id", "season").agg(
        pl.col("snap_pct").mean().alias("snap_pct_l3"),
        pl.col("target_share").mean().alias("target_share_l3"),
        (pl.col("targets") + pl.col("carries")).mean().alias("opportunities_l3"),
        *[
            pl.col(c).mean().alias(f"{c}_l3")
            for c in ("fp_std", "fp_half", "fp_ppr", "xfp_std", "xfp_half", "xfp_ppr")
        ],
    )
    season = hist.group_by("player_id", "season").agg(
        pl.len().alias("games"),
        pl.col("week").max().cast(pl.Int64).alias("last_game_week"),
        pl.col("xfp_ppr").mean().alias("xfp_ppr_season"),
    )
    prior_snaps = prior.group_by("player_id", "season").agg(
        pl.col("snap_pct").mean().alias("snap_pct_prior")
    )

    return (
        season.join(l3, on=["player_id", "season"])
        .join(prior_snaps, on=["player_id", "season"], how="left")
        .with_columns(
            pl.lit(through_week, dtype=pl.Int64).alias("through_week"),
            pl.col("games").cast(pl.Int64),
            (pl.col("snap_pct_l3") - pl.col("snap_pct_prior")).alias("snap_pct_trend"),
            (pl.col("fp_ppr_l3") - pl.col("xfp_ppr_l3")).alias("fp_minus_xfp_ppr_l3"),
        )
        .drop("snap_pct_prior")
    )


COLUMNS = [
    "player_id",
    "season",
    "through_week",
    "games",
    "last_game_week",
    "snap_pct_l3",
    "snap_pct_trend",
    "target_share_l3",
    "opportunities_l3",
    "fp_std_l3",
    "fp_half_l3",
    "fp_ppr_l3",
    "xfp_std_l3",
    "xfp_half_l3",
    "xfp_ppr_l3",
    "xfp_ppr_season",
    "fp_minus_xfp_ppr_l3",
]


def compute_usage_features(weeks: pl.DataFrame) -> pl.DataFrame:
    if weeks.is_empty():
        return pl.DataFrame(schema={c: pl.Float64 for c in COLUMNS})
    frames = [_features_through(weeks, w) for w in sorted(weeks["week"].unique().to_list())]
    return pl.concat(frames).select(COLUMNS).sort("player_id", "season", "through_week")
