"""Usage features computed from player_weeks.

Features are computed as of every week ("through_week"), not just the latest, so the
weekly evaluation can score a recommendation against the data that existed when it
was made.
"""

from __future__ import annotations

import polars as pl

from startsit.db import TABLES
from startsit.scoring import POINT_COLUMNS

WINDOW = 3
GROUP = ["player_id", "season"]
USAGE_TABLE = TABLES["usage_features"]


def _features_through(weeks: pl.DataFrame, through_week: int) -> pl.DataFrame:
    hist = weeks.filter(pl.col("week") <= through_week).sort("week")
    recent = hist.group_by(GROUP).tail(WINDOW)
    prior = hist.join(recent.select(*GROUP, "week"), on=[*GROUP, "week"], how="anti").group_by(GROUP).tail(WINDOW)

    l3 = recent.group_by(GROUP).agg(
        pl.col("snap_pct").mean().alias("snap_pct_l3"),
        pl.col("target_share").mean().alias("target_share_l3"),
        (pl.col("targets") + pl.col("carries")).mean().alias("opportunities_l3"),
        *[pl.col(c).mean().alias(f"{c}_l3") for c in POINT_COLUMNS],
    )
    season = hist.group_by(GROUP).agg(
        pl.len().alias("games"),
        pl.col("week").max().cast(pl.Int64).alias("last_game_week"),
        pl.col("xfp_ppr").mean().alias("xfp_ppr_season"),
    )
    prior_snaps = prior.group_by(GROUP).agg(pl.col("snap_pct").mean().alias("snap_pct_prior"))

    return (
        season.join(l3, on=GROUP)
        .join(prior_snaps, on=GROUP, how="left")
        .with_columns(
            pl.lit(through_week, dtype=pl.Int64).alias("through_week"),
            pl.col("games").cast(pl.Int64),
            (pl.col("snap_pct_l3") - pl.col("snap_pct_prior")).alias("snap_pct_trend"),
            (pl.col("fp_ppr_l3") - pl.col("xfp_ppr_l3")).alias("fp_minus_xfp_ppr_l3"),
        )
        .drop("snap_pct_prior")
    )


def compute_usage_features(weeks: pl.DataFrame) -> pl.DataFrame:
    if weeks.is_empty():
        return pl.DataFrame(schema=USAGE_TABLE.polars_schema())
    frames = [_features_through(weeks, w) for w in sorted(weeks["week"].unique().to_list())]
    return pl.concat(frames).select(USAGE_TABLE.columns).sort(list(USAGE_TABLE.key))
