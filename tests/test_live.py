"""Live integration tests: the real external services, end to end through our code.

They catch what the faked unit tests can't: a changed API response, a schema Gemini now
rejects, a moved feed, renamed nflverse columns. Run them with `pytest -m live` (CI runs
them only on merges to main that touch a source adapter, and when started by hand).

Quota per run, kept as small as possible:
- Gemini: 3 calls (one decision, one news extraction, one Reddit vote extraction), at most 5
  with validation retries;
  `LIVE_GEMINI_BUDGET` enforces the ceiling. Gemini's free tier allows 500/day.
- FantasyPros: 1 call (one batch of projections). The free key allows ~75/day; the daily
  ingest uses ~26.
- Everything else (feeds, the rankings file, nflverse, Reddit) is free; each is fetched once.

Each test skips when its key isn't set, so forks and local runs without keys still pass.
"""

from __future__ import annotations

import os
import time
import urllib.error
from datetime import UTC, datetime
from functools import cache

import polars as pl
import pytest

from startsit.baseline import BaselineScore
from startsit.db import Database
from startsit.decide import decide
from startsit.evidence import EvidenceItem, EvidencePacket, PlayerEvidence, Question
from startsit.http import get_bytes
from startsit.ingest import news, news_extract, nflverse, projections, rankings
from startsit.llm import BudgetedLLM, GeminiProvider

pytestmark = pytest.mark.live

LIVE_GEMINI_BUDGET = 5  # hard ceiling on Gemini calls per run, across these tests


def needs(var: str) -> None:
    if not os.environ.get(var):
        pytest.skip(f"{var} not set")


@pytest.fixture(scope="module")
def live_db():
    # A throwaway database: live calls never count against (or show up in) production's llm_calls.
    with Database(":memory:") as db:
        yield db


@pytest.fixture(scope="module")
def gemini(live_db):
    needs("GEMINI_API_KEY")
    return BudgetedLLM(GeminiProvider(), live_db, "live-test", LIVE_GEMINI_BUDGET)


@cache
def season() -> int:
    import nflreadpy as nfl

    return int(nfl.get_current_season())


@cache
def schedule() -> pl.DataFrame:
    import nflreadpy as nfl

    raw = nflverse.RawNflverse(*([pl.DataFrame()] * 6), schedules=nfl.load_schedules([season()]))
    return nflverse.build_games(raw)


@cache
def rankings_raw() -> rankings.RawRankings:
    return rankings.fetch()


def test_gemini_returns_a_valid_grounded_decision(gemini, live_db):
    a = PlayerEvidence(
        "A",
        "Alpha Back",
        "RB",
        "PHI",
        BaselineScore("A", 0.0, "ruled out (news)"),
        [
            EvidenceItem("E1", "A", "consensus", "FantasyPros consensus RB5 (avg rank 5.1)"),
            EvidenceItem(
                "E2", "A", "news", "2026-10-24 RotoWire: Alpha Back ruled out for Sunday. (injury, high impact)"
            ),
        ],
    )
    b = PlayerEvidence(
        "B",
        "Bravo Back",
        "RB",
        "DAL",
        BaselineScore("B", 11.0),
        [EvidenceItem("E3", "B", "consensus", "FantasyPros consensus RB22 (avg rank 21.8)")],
    )
    packet = EvidencePacket(Question(2026, 7, "ppr", datetime(2026, 10, 25, tzinfo=UTC)), [a, b], "B", 11.0)
    decision = decide(live_db, packet, gemini)
    assert decision.source == "llm", decision.note  # validated: real pick, cited ids exist
    assert decision.pick == "B"  # the only playable player
    assert 2 <= len(decision.factors) <= 3


def test_gemini_extracts_news_facts_with_the_real_schema(gemini, live_db):
    live_db.upsert("players", [dict(player_id="A", name="Alpha Back", position="RB", team="PHI", pfr_id=None)])
    now = datetime.now(UTC)
    item = dict(
        item_id="live:1",
        url="https://example.com/1",
        source="Test",
        title="Eagles rule out RB Alpha Back (ankle) for Sunday",
        summary="Alpha Back will miss Sunday's game against Dallas with an ankle sprain.",
        published_at=now.isoformat(timespec="seconds"),
        fetched_at=now.isoformat(timespec="seconds"),
    )
    live_db.upsert("news_items", [item])
    result = news_extract.extract_news(live_db, gemini, now)
    assert result.stopped_early is None and result.calls == 1
    facts = live_db.news_for_player("A", "2000-01-01", "2100-01-01", 5)
    assert [(f["event"], f["game_status"]) for f in facts] == [("injury", "out")]


def test_fantasypros_projections_with_one_call():
    needs("FANTASYPROS_API_KEY")
    ranked = rankings.build_rankings(rankings_raw(), schedule(), season())
    if ranked.is_empty():
        pytest.skip("no current-week rankings (off-season)")
    week = int(ranked["week"][0])
    ids = ranked.filter(pl.col("position") == "WR").sort("ecr")["fantasypros_id"].head(3).to_list()
    client = projections.FantasyProsClient.from_env()
    assert client is not None
    try:  # one call (plus one retry if throttled)
        data = client.get(f"{season()}/projections", {"position": "WR", "week": week, "players": ":".join(ids)})
    except projections.FantasyProsQuotaExceeded:
        pytest.skip("FantasyPros quota reached; projections can't be checked until it resets")
    rows = projections.build(data.get("players") or [], [], season(), week, "live")
    assert len(rows) >= 2, "the top WRs should all have projections"
    assert all(r["points_ppr"] is not None and r["receptions"] is not None for r in rows)


def test_every_news_feed_returns_parseable_items():
    bodies = news.fetch_all()
    items = news.build_news_items(bodies, datetime.now(UTC))
    missing = {f.source for f in news.FEEDS} - {i["source"] for i in items}
    assert not missing, f"feeds with no items: {missing}"


def test_rankings_file_has_the_columns_we_read():
    raw = rankings_raw()
    used = {"page", "page_pos", "scrape_date", "fantasypros_id", "player_name", "team", "player_opponent",
            "pos_rank", "start_sit_grade", "player_game_status", "player_page_url", "player_game_kickoff_ts",
            *rankings.FLOAT_COLS}  # fmt: skip
    assert used <= set(raw.rankings.columns)
    assert {"fantasypros_id", "gsis_id"} <= set(raw.player_ids.columns)
    rankings.build_rankings(raw, schedule(), season())  # maps without error (empty off-season)


@cache
def nflverse_raw() -> nflverse.RawNflverse:
    return nflverse.fetch(season())


def test_nflverse_season_builds_every_table():
    raw = nflverse_raw()
    weeks = nflverse.build_player_weeks(raw)
    assert not weeks.is_empty()
    for frame in (
        nflverse.build_roster_weeks(raw),
        nflverse.build_players(raw, weeks),
        nflverse.build_injuries(raw),
        nflverse.build_games(raw),
    ):
        assert not frame.is_empty()


REDDIT_ROUTES = (
    "https://www.reddit.com/r/fantasyfootball/new.json?limit=5",
    "https://old.reddit.com/r/fantasyfootball/new.json?limit=5",
    "https://www.reddit.com/r/fantasyfootball/new/.rss?limit=5",
)


def reddit_status(url: str) -> str:
    try:
        body = get_bytes(url)
    except urllib.error.HTTPError as e:
        return f"{e.code} {e.reason}"
    except OSError as e:
        return f"{type(e).__name__}: {e}"
    return f"200, {len(body)} bytes, starts {body[:40]!r}"


def test_reddit_is_readable_from_this_runner():
    # Step 6 would read r/fantasyfootball from the ingest jobs, which run here on GitHub Actions.
    # Reddit often treats datacenter IPs differently from a laptop, so this checks each free route.
    results = {}
    for url in REDDIT_ROUTES:
        results[url] = reddit_status(url)
        time.sleep(2)  # be polite
    report = "\n".join(f"{status}  <- {url}" for url, status in results.items())
    print(report)
    assert any(status.startswith("200") for status in results.values()), f"every route blocked:\n{report}"


def test_reddit_wdis_comments_become_crowd_votes(gemini, live_db):
    # Step 6b end to end on live data: find today's WDIS threads, read one thread's comments with
    # the real fetcher (60 s after the search, waiting out a 429 once), and extract votes in one
    # Gemini call. Skips if Reddit is throttling this runner: that's Reddit's limit, not our bug.
    from startsit.ingest import reddit, reddit_extract

    body = reddit.reddit_get(reddit.SEARCH_URL)
    if body is None:
        pytest.skip("Reddit refused the search feed (rate limit on a shared runner IP)")
    threads = reddit.latest_threads(body)
    print("threads:", [t.title for t in threads])
    assert threads, "no WDIS threads in the search feed"
    time.sleep(reddit.SPACING)
    feed = reddit.reddit_get(reddit.comments_url(threads[0]))
    if feed is None:
        pytest.skip("Reddit refused the comments feed (rate limit on a shared runner IP)")
    now = datetime.now(UTC)
    comments = reddit.comments(feed, threads[0], now)
    print(f"{threads[0].title}: {len(comments)} comments")
    assert comments, "the thread's feed had no usable comments"

    raw = nflverse_raw()
    live_db.upsert("players", nflverse.build_players(raw, nflverse.build_player_weeks(raw)).to_dicts())
    live_db.upsert("reddit_comments", comments[:50])
    result = reddit_extract.extract_votes(live_db, gemini, now)
    print(f"extracted: {result}")
    for v in live_db.query(
        "SELECT c.text, v.player_id, v.verdict, v.reason FROM crowd_votes v "
        "JOIN reddit_comments c ON c.comment_id = v.comment_id"
    )[:15]:
        print(f"  {v['verdict']:5} {v['player_id']}  <- {v['text'][:90]!r}  ({v['reason']})")
    assert result.calls == 1 and result.stopped_early is None
    assert result.votes > 0, "no votes from a busy WDIS thread: check the prompt"
