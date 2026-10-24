"""Weekly consensus rankings: FantasyPros ECR as published by DynastyProcess (ffverse).

`fp_latest_weekly.csv` is FantasyPros' full weekly expert consensus for every ranked
player, overwritten in place on each scrape. Each ingest archives the scrape it sees
under its scrape_date so evaluation can later use the rankings that existed at the time.

Skill positions are ranked on PPR pages only; other formats are derived downstream
(see startsit.baseline). FantasyPros must be cited as the source wherever this is shown.

Files are read from raw.githubusercontent.com rather than through nflreadpy, whose
github.com/.../raw URLs some proxies (including the Claude Code cloud env) block.
"""

from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass

import polars as pl

BASE_URL = "https://raw.githubusercontent.com/dynastyprocess/data/master/files"
RANKINGS_URL = f"{BASE_URL}/fp_latest_weekly.csv"
PLAYER_IDS_URL = f"{BASE_URL}/db_playerids.csv"
USER_AGENT = "who-should-i-start/0.1 (+https://github.com/niwang96/who-should-i-start)"

SKILL_PAGES = ["qb", "ppr-rb", "ppr-wr", "ppr-te"]
FLOAT_COLS = {"ecr": "ecr", "sd": "ecr_sd", "best": "ecr_best", "worst": "ecr_worst", "r2p_pts": "proj_ppr"}

RANKINGS_SCHEMA = {
    "season": pl.Int64, "week": pl.Int64, "scrape_date": pl.String, "fantasypros_id": pl.String,
    "player_id": pl.String, "player_name": pl.String, "position": pl.String, "team": pl.String,
    "opponent": pl.String, "ecr": pl.Float64, "ecr_sd": pl.Float64, "ecr_best": pl.Float64,
    "ecr_worst": pl.Float64, "proj_ppr": pl.Float64, "pos_rank": pl.String,
    "start_sit_grade": pl.String, "game_status": pl.String, "source_url": pl.String,
}


@dataclass
class RawRankings:
    rankings: pl.DataFrame
    player_ids: pl.DataFrame


def _read_csv(url: str) -> pl.DataFrame:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
    # Every column as text: the files use the literal "NA" for missing values.
    return pl.read_csv(io.BytesIO(body), infer_schema_length=0, null_values=["NA", ""])


def fetch() -> RawRankings:
    return RawRankings(rankings=_read_csv(RANKINGS_URL), player_ids=_read_csv(PLAYER_IDS_URL))


def infer_week(kickoff_ts: pl.Series, games: pl.DataFrame) -> int | None:
    """The week whose games fall on the most kickoff dates (US Eastern), or None.

    The file carries no week column, and an early-week scrape still lists last
    Thursday's closed game, so match every listed kickoff and take the most common week.
    """
    dates = (
        pl.DataFrame({"ts": kickoff_ts.cast(pl.Int64, strict=False)})
        .drop_nulls()
        .select(
            pl.from_epoch("ts", time_unit="s")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("America/New_York")
            .dt.date()
            .cast(pl.String)
            .alias("gameday")
        )
    )
    weeks = dates.join(games.select("gameday", "week").unique(), on="gameday", how="inner")
    if weeks.is_empty():
        return None
    return int(weeks["week"].mode().sort()[-1])


def build_rankings(raw: RawRankings, games: pl.DataFrame, season: int) -> pl.DataFrame:
    """One row per ranked QB/RB/WR/TE for the scrape's week, keyed to GSIS ids.

    Returns an empty frame if the scrape's games aren't in `season`'s schedule, which
    keeps a backfill of an old season from filing this week's rankings under it.
    """
    skill = raw.rankings.filter(pl.col("page").is_in(SKILL_PAGES))
    week = infer_week(skill["player_game_kickoff_ts"], games.filter(pl.col("season") == season))
    if week is None or skill.is_empty():
        return pl.DataFrame(schema=RANKINGS_SCHEMA)

    id_map = (
        raw.player_ids.select("fantasypros_id", pl.col("gsis_id").alias("player_id"))
        .drop_nulls("fantasypros_id")
        .unique("fantasypros_id", keep="first")
    )
    return (
        skill.join(id_map, on="fantasypros_id", how="left")
        .select(
            pl.lit(season, dtype=pl.Int64).alias("season"),
            pl.lit(week, dtype=pl.Int64).alias("week"),
            "scrape_date",
            "fantasypros_id",
            "player_id",
            "player_name",
            pl.col("page_pos").alias("position"),
            "team",
            pl.col("player_opponent").alias("opponent"),
            *[pl.col(src).cast(pl.Float64, strict=False).alias(dst) for src, dst in FLOAT_COLS.items()],
            "pos_rank",
            "start_sit_grade",
            pl.col("player_game_status").alias("game_status"),
            pl.col("player_page_url").alias("source_url"),
        )
        .unique(["season", "week", "scrape_date", "fantasypros_id"], keep="first")
        .sort("position", "ecr")
    )

