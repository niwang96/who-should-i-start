"""Stage 1 ingest job: python -m startsit.ingest [--season 2026] [--db path]

Re-ingests the whole season each run (a few thousand rows), so runs are idempotent and
a missed cron run heals itself on the next one.
"""

from __future__ import annotations

import argparse
import logging

import polars as pl

from startsit.db import Database
from startsit.features import compute_usage_features
from startsit.ingest import nflverse

log = logging.getLogger("startsit.ingest")


def run(db: Database, raw: nflverse.RawNflverse, season: int) -> dict[str, int]:
    weeks = nflverse.build_player_weeks(raw).filter(pl.col("season") == season)
    players = nflverse.build_players(raw, weeks)
    injuries = nflverse.build_injuries(raw).filter(pl.col("season") == season)
    features = compute_usage_features(weeks)
    max_week = int(weeks["week"].max()) if not weeks.is_empty() else None

    counts = {
        "players": db.upsert("players", players.to_dicts()),
        "player_weeks": db.upsert("player_weeks", weeks.to_dicts()),
        "injuries": db.upsert("injuries", injuries.to_dicts()),
        "usage_features": db.upsert("usage_features", features.to_dicts()),
    }
    for source, n in counts.items():
        db.record_run(f"nflverse:{source}", season, max_week, n)
    log.info("season %s through week %s: %s", season, max_week, counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=None, help="defaults to the current NFL season")
    parser.add_argument("--db", default=None, help="SQLite path (default: $STARTSIT_DB_PATH or data/startsit.db)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.season is None:
        import nflreadpy as nfl

        args.season = nfl.get_current_season()

    db = Database(args.db)
    try:
        run(db, nflverse.fetch(args.season), args.season)
    finally:
        db.close()


if __name__ == "__main__":
    main()
