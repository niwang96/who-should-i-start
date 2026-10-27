"""Stage 1 ingest job: python -m startsit.ingest [--only stats|news|news-fetch] [--season 2026] [--db path-or-url]

stats: re-ingests the whole season each run (a few thousand rows), so runs are idempotent
and a missed cron run heals itself on the next one. Rankings are the exception: the source
file only holds the latest scrape, so each run archives the one it sees. With
FANTASYPROS_API_KEY set it also archives FantasyPros projections (~26 API calls).

After each stats refresh, recommendations whose games are all in the data are scored against
actual points (step 6a, startsit.evaluate).

news: stores new RSS items, extracts player facts from items not yet processed (batched
LLM calls within a daily budget; skipped without GEMINI_API_KEY), and re-checks the
rankings file so each update is archived within the hour. It also runs the daily stats
refresh when that's due (none since the latest 10:00 UTC), so the daily data arrives even
when GitHub drops the scheduled run meant for it; every scheduled run uses this mode. Once
a week, a few hours before the first Sunday kickoff, it asks the AI the weekly benchmark
matchups (startsit.benchmark).

news-fetch: stores new RSS items only (no LLM), for frequent polling of feeds that only
list their latest few items, by hand; the next news run extracts them.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta

import polars as pl

from startsit import benchmark, evaluate
from startsit.db import TABLES, Database
from startsit.evidence import current_week
from startsit.features import compute_usage_features
from startsit.ingest import news, news_extract, nflverse, projections, rankings
from startsit.llm import BudgetedLLM, budgeted_gemini

log = logging.getLogger("startsit.ingest")

# The source normally refreshes daily; older than this and "current week" answers go stale.
STALE_RANKINGS_DAYS = 3
# News gets a slice of the 500/day Gemini free tier; the rest is for start/sit answers.
# Hourly runs plus the Sunday-morning and pre-night-game runs make at most ~40 calls a day.
NEWS_DAILY_LLM_BUDGET = 60
# The daily stats refresh is due once per day after this time (UTC): nflverse's overnight
# updates are in by then (6am US Eastern).
DAILY_REFRESH_AFTER = time(10, 0)


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


def run_projections(db: Database, client: projections.FantasyProsClient, season: int, today: date | None = None) -> int:
    week = db.latest_rankings_week(season)
    if week is None:
        log.warning("no rankings for %s yet; projections skipped", season)
        return 0
    ranked = db.rankings_for_week(season, week)
    raw = projections.fetch(client, season, week, projections.targets(ranked))
    rows = projections.build(raw, ranked, season, week, (today or date.today()).isoformat())
    n = db.upsert("projections", rows)
    db.record_run("fantasypros:projections", season, week, n)
    log.info("projections week %s: %s players", week, n)
    return n


def run_news(
    db: Database,
    bodies: dict[news.Feed, bytes],
    llm: BudgetedLLM | None,
    season: int,
    now: datetime | None = None,
    extract: bool = True,
) -> dict[str, int]:
    now = now or datetime.now(UTC)
    items = news.build_news_items(bodies, now)
    counts = {"news_items": db.upsert("news_items", items)}
    if extract and llm is None:
        log.warning("no LLM configured; news items stored but not extracted")
    elif extract and llm is not None:
        result = news_extract.extract_news(db, llm, now)
        counts |= {"news_extracted": result.items_processed, "news_facts": result.facts, "llm_calls": result.calls}
    for source, n in counts.items():
        db.record_run(f"news:{source}", season, None, n)
    log.info("news: %s", counts)
    return counts


def stats_due(last_run: str | None, now: datetime) -> bool:
    """Whether the daily stats refresh is due: none since the most recent DAILY_REFRESH_AFTER (UTC)."""
    if last_run is None:
        return True
    slot = datetime.combine(now.astimezone(UTC).date(), DAILY_REFRESH_AFTER, UTC)
    if now < slot:
        slot -= timedelta(days=1)
    return datetime.fromisoformat(last_run[:19]).replace(tzinfo=UTC) < slot


def run_stats(db: Database, season: int) -> None:
    """The daily refresh: nflverse, the rankings file and (with a key) FantasyPros projections."""
    raw = nflverse.fetch(season)
    run(db, raw, season)
    run_rankings(db, rankings.fetch(), nflverse.build_games(raw), season)
    if client := projections.FantasyProsClient.from_env():
        _optional("projections", lambda: run_projections(db, client, season))
    else:
        log.warning("FANTASYPROS_API_KEY not set; projections skipped")
    _optional("evaluation", lambda: run_evaluation(db))


def run_evaluation(db: Database) -> int:
    n = evaluate.score_pending(db)
    log.info("evaluation: %s recommendations scored", n)
    return n


def benchmark_llm(db: Database) -> BudgetedLLM | None:
    return budgeted_gemini(db, "benchmark", benchmark.BENCHMARK_DAILY_LLM_BUDGET, "benchmark skipped")


def games_frame(db: Database, season: int) -> pl.DataFrame:
    """The season's schedule from the database, as the frame the rankings ingest matches against."""
    return pl.DataFrame(db.games(season), schema=TABLES["games"].polars_schema())


def run_benchmark(db: Database, season: int, now: datetime | None = None) -> int:
    """The weekly benchmark matchups, when they're due and the AI is available."""
    week = current_week(db, season)
    if not benchmark.due(db, season, week, now or datetime.now(UTC)):
        return 0
    llm = benchmark_llm(db)
    return benchmark.run(db, llm, season, week) if llm else 0


def run_backfill(db: Database, season: int, now: datetime | None = None) -> int:
    """Ask one past week's benchmark set as of its Sunday morning, when today's budget allows.

    Not on Sundays (UTC, the day the budget resets on: Saturday 8pm to Sunday 8pm Eastern),
    whose budget is the live set's, and only with room for the rest of the week's set, so in
    practice one week a day, newest first; an interrupted set resumes on a later run. The
    rankings scrape from before that Sunday is read from the file's git history and archived,
    so the packets hold what was known then.
    """
    now = now or datetime.now(UTC)
    if now.astimezone(UTC).weekday() == 6:
        return 0
    target = benchmark.backfill_week(db, season, current_week(db, season), now)
    if target is None:
        return 0
    week, as_of = target
    llm = benchmark_llm(db)
    if llm is None or llm.remaining(now) < benchmark.calls_needed(db, season, week):
        return 0
    sha = rankings.snapshot_before(as_of)
    ranked = rankings.build_rankings(rankings.fetch_snapshot(sha), games_frame(db, season), season) if sha else None
    if ranked is None or ranked.is_empty() or int(ranked["week"][0]) != week:
        log.warning("backfill week %s: no rankings scrape for that week before %s", week, as_of)
        return 0
    db.upsert("rankings", ranked.to_dicts())
    return benchmark.run(db, llm, season, week, as_of=as_of, tag=evaluate.BACKFILL)


def _optional(step: str, fn: Callable[[], object]) -> None:
    """Run a supplementary step; its failure is logged, not allowed to stop the steps after it."""
    try:
        fn()
    except Exception:
        log.exception("%s failed; continuing", step)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=None, help="defaults to the current NFL season")
    parser.add_argument(
        "--db",
        default=None,
        help="SQLite path or postgres:// URL (default: $DATABASE_URL, $STARTSIT_DB_PATH, or data/startsit.db)",
    )
    parser.add_argument(
        "--only", choices=["stats", "news", "news-fetch"], default=None, help="run one part (default: stats + news)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.season is None:
        import nflreadpy as nfl

        args.season = nfl.get_current_season()

    with Database(args.db) as db:
        if args.only in (None, "stats"):
            run_stats(db, args.season)
        if args.only == "news-fetch":
            run_news(db, news.fetch_all(), None, args.season, extract=False)
        if args.only == "news":
            if stats_due(db.latest_stats_run(), datetime.now(UTC)):
                log.info("daily stats refresh is due; running it first")
                _optional("daily stats refresh", lambda: run_stats(db, args.season))
            else:
                games = games_frame(db, args.season)
                _optional("rankings re-check", lambda: run_rankings(db, rankings.fetch(), games, args.season))
        if args.only in (None, "news"):
            llm = budgeted_gemini(db, "news", NEWS_DAILY_LLM_BUDGET, "news extraction disabled")
            run_news(db, news.fetch_all(), llm, args.season)
        if args.only == "news":
            _optional("benchmark", lambda: run_benchmark(db, args.season))
            _optional("benchmark backfill", lambda: run_backfill(db, args.season))


if __name__ == "__main__":
    main()
