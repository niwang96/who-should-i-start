"""Stage 1 ingest job: python -m startsit.ingest [--season 2026] [--db path-or-url]

Re-ingests the whole season each run (a few thousand rows), so runs are idempotent and
a missed cron run heals itself on the next one. Rankings are the exception: the source
file only holds the latest scrape, so each run archives the one it sees.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

import polars as pl

from startsit.db import Database
from startsit.features import compute_usage_features
from startsit.ingest import nflverse, rankings

log = logging.getLogger("startsit.ingest")

# The source normally refreshes daily; older than this and "current week" answers go stale.
STALE_RANKINGS_DAYS = 3


def run(db: Database, raw: nflverse.RawNflverse, season: int) -> dict[str, int]:
    weeks = nflverse.build_player_weeks(raw).filter(pl.col("season") == season)
    roster_weeks = nflverse.build_roster_weeks(raw).filter(pl.col("season") == season)
    players = nflverse.build_players(raw, weeks)
    injuries = nflverse.build_injuries(raw).filter(pl.col("season") == season)
    features = compute_usage_features(weeks)
    games = nflverse.build_games(raw).filter(pl.col("season") == season)
    max_week = int(weeks.select(pl.max("week")).item()) if not weeks.is_empty() else None

    counts = {
        "players": db.upsert("players", players.to_dicts()),
        "roster_weeks": db.upsert("roster_weeks", roster_weeks.to_dicts()),
        "player_weeks": db.upsert("player_weeks", weeks.to_dicts()),
        "injuries": db.upsert("injuries", injuries.to_dicts()),
        "usage_features": db.upsert("usage_features", features.to_dicts()),
        "games": db.upsert("games", games.to_dicts()),
    }
    for source, n in counts.items():
        db.record_run(f"nflverse:{source}", season, max_week, n)
    log.info("season %s through week %s: %s", season, max_week, counts)
    return counts


def run_rankings(
    db: Database, raw: rankings.RawRankings, games: pl.DataFrame, season: int, today: date | None = None
) -> int:
    ranked = rankings.build_rankings(raw, games, season)
    if ranked.is_empty():
        log.warning("rankings scrape doesn't match any %s week; skipped", season)
        return 0
    week = int(ranked["week"][0])
    scrape_date = str(ranked["scrape_date"][0])
    age = ((today or date.today()) - date.fromisoformat(scrape_date)).days
    if age > STALE_RANKINGS_DAYS:
        log.warning("rankings scrape is %s days old (%s); the source may have stopped updating", age, scrape_date)
    n = db.upsert("rankings", ranked.to_dicts())
    db.record_run("ffverse:rankings", season, week, n)
    log.info(
        "rankings week %s scraped %s: %s rows (%s without a GSIS id)",
        week,
        scrape_date,
        n,
        ranked["player_id"].null_count(),
    )
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=None, help="defaults to the current NFL season")
    parser.add_argument(
        "--db",
        default=None,
        help="SQLite path or postgres:// URL (default: $DATABASE_URL, $STARTSIT_DB_PATH, or data/startsit.db)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.season is None:
        import nflreadpy as nfl

        args.season = nfl.get_current_season()

    raw = nflverse.fetch(args.season)
    with Database(args.db) as db:
        run(db, raw, args.season)
        run_rankings(db, rankings.fetch(), nflverse.build_games(raw), args.season)


if __name__ == "__main__":
    main()
