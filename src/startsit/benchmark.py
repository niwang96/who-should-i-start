"""Step 6a: the weekly benchmark set, so the evaluation has enough matchups to mean something.

Real questions are too few for meaningful hit rates, so once a week the ingest job asks the AI
about 30 close matchups: same position, neighbors in the consensus ranks, from the range
where start/sit decisions are hard. They go through the normal decide path (validation,
cache, budget) and are logged as recommendations with visitor = "benchmark", which every
report keeps apart from real users.

Timing: three hours before the Sunday main slate (the first Sunday kickoff at noon Eastern or
later), when most news is in, like a user setting a lineup on Sunday morning. Players whose
game has already kicked off (Thursday, or a morning international game) are left out.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from startsit.db import Database, GameRow
from startsit.decide import decide, record_recommendation
from startsit.evaluate import BACKFILL, BENCHMARK
from startsit.evidence import EASTERN, build_packet, kickoff
from startsit.llm import BudgetedLLM
from startsit.scoring import SCORINGS, Scoring

log = logging.getLogger("startsit.benchmark")

# Gemini's free tier allows 500 requests/day: decide 390 + news 60 + benchmark 45 (30 matchups + retries).
BENCHMARK_DAILY_LLM_BUDGET = 45
# Matchups per position, and the position rank where pairs start (1-based): borderline starters.
PAIRS = {"QB": 4, "RB": 9, "WR": 12, "TE": 5}
WINDOW_START = {"QB": 10, "RB": 15, "WR": 20, "TE": 6}
LEAD_TIME = timedelta(hours=3)  # before the Sunday main slate: 10am Eastern for the 1pm games
MAIN_SLATE_HOUR = 12  # Eastern; earlier Sunday kickoffs are international morning games
# The backfill's first week: 2026 week 1 rankings scrapes have no kickoff times, so their week
# can't be told (startsit.ingest.rankings.build_rankings).
BACKFILL_FROM_WEEK = 2
# Seconds between AI calls: at most 4 of the shared 12/minute, leaving the rest for real users
# on Sunday mornings, their busiest time (~8 minutes for a set).
CALL_SPACING = 15.0

Matchup = tuple[str, str, Scoring]


def benchmark_time(games: list[GameRow]) -> datetime | None:
    """When the week's benchmark is due: LEAD_TIME before the Sunday main slate (else the first kickoff)."""
    kicks = [k for g in games if (k := kickoff(g)) is not None]
    sundays = [k for k in kicks if (local := k.astimezone(EASTERN)).weekday() == 6 and local.hour >= MAIN_SLATE_HOUR]
    pool = sundays or kicks
    return min(pool) - LEAD_TIME if pool else None


def done(db: Database, season: int, week: int) -> bool:
    """Whether the week has a benchmark set, live or backfilled."""
    return any(db.has_recommendations(season, week, tag) for tag in (BENCHMARK, BACKFILL))


def due(db: Database, season: int, week: int, now: datetime) -> bool:
    """Whether to ask the week's set now: in the LEAD_TIME window before the main slate, if not done.

    A window that passed unused (dropped runs) is left to the backfill, which asks as of its start.
    """
    if done(db, season, week):
        return False
    when = benchmark_time(db.week_games(season, week))
    return when is not None and when <= now < when + LEAD_TIME


def backfill_week(db: Database, season: int, through_week: int, now: datetime) -> tuple[int, datetime] | None:
    """The newest week up to `through_week` whose benchmark window has passed without a set, and its as-of time."""
    for week in range(through_week, BACKFILL_FROM_WEEK - 1, -1):
        when = benchmark_time(db.week_games(season, week))
        if when is not None and now >= when + LEAD_TIME and not done(db, season, week):
            return week, when
    return None


def not_started(db: Database, player_id: str, season: int, week: int, now: datetime) -> bool:
    """Whether the player has a game this week that hasn't kicked off (by that week's roster team)."""
    roster = db.roster_status(player_id, season, week)
    player = db.player(player_id)
    team = (roster and roster["team"]) or (player and player["team"])
    game = db.team_game(season, week, team) if team else None
    start = kickoff(game) if game else None
    return start is not None and start > now


def matchups(db: Database, season: int, week: int, now: datetime) -> list[Matchup]:
    """Neighbors in the consensus ranks, per position, cycling through the scoring formats."""
    ranked = [
        r
        for r in db.rankings_for_week(season, week, through_day=now.astimezone(UTC).date().isoformat())
        if r["player_id"] is not None and r["ecr"] is not None and r["proj_ppr"] is not None
    ]
    out: list[Matchup] = []
    for position, count in PAIRS.items():
        rows = sorted((r for r in ranked if r["position"] == position), key=lambda r: r["ecr"] or 0.0)
        ids = [
            pid
            for r in rows[WINDOW_START[position] - 1 :]
            if not_started(db, pid := str(r["player_id"]), season, week, now)
        ]
        for a, b in list(zip(ids[0::2], ids[1::2], strict=False))[:count]:
            n = len(out)
            # Alternate who's listed first, so a bias toward the first player can't look like skill.
            first, second = (a, b) if n % 2 == 0 else (b, a)
            out.append((first, second, SCORINGS[n % len(SCORINGS)]))
    return out


def run(
    db: Database,
    llm: BudgetedLLM,
    season: int,
    week: int,
    as_of: datetime | None = None,
    tag: str = BENCHMARK,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Ask the AI the week's benchmark matchups and log the answers; returns how many were logged.

    `as_of` (default now) is when the questions count as asked: the backfill passes a past Sunday
    morning, so the packets hold only what was known then. Budget checks use the real clock.
    """
    asked = as_of or clock()
    logged = 0
    for a, b, scoring in matchups(db, season, week, asked):
        if llm.remaining(clock()) <= 0:
            log.warning("benchmark budget spent after %s matchups", logged)
            break
        packet = build_packet(db, [a, b], scoring, season, week, as_of=asked)
        decision = decide(db, packet, llm)
        record_recommendation(db, packet, decision, tag, created_at=asked)
        logged += 1
        if decision.source == "llm":
            sleep(CALL_SPACING)
    log.info("%s week %s: %s matchups", tag, week, logged)
    return logged
