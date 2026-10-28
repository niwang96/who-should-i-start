"""Live integration tests: the real external services, end to end through our code.

They catch what the faked unit tests can't: a changed API response, a schema Gemini now
rejects, a moved feed, renamed nflverse columns. Run them with `pytest -m live` (CI runs
them only on merges to main that touch a source adapter, and when started by hand).

Quota per run, kept as small as possible:
- Gemini: 2 calls (one decision, one news extraction), at most 4 with validation retries;
  `LIVE_GEMINI_BUDGET` enforces the ceiling. Gemini's free tier allows 500/day.
- FantasyPros: 1 call (one batch of projections). The free key allows ~75/day; the daily
  ingest uses ~26.
- Everything else (feeds, the rankings file, nflverse, Reddit) is free; each is fetched once.

Each test skips when its key isn't set, so forks and local runs without keys still pass.
"""

from __future__ import annotations

import os
import re
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

LIVE_GEMINI_BUDGET = 4  # hard ceiling on Gemini calls per run, across these tests


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


def test_nflverse_season_builds_every_table():
    raw = nflverse.fetch(season())
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
    "https://www.reddit.com/r/fantasyfootball/comments/.rss?limit=5",  # newest comments, sub-wide
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


ATOM = "{http://www.w3.org/2005/Atom}"
# r/fantasyfootball's daily "Who Do I Start" threads are titled like "Official: [WDIS RB] - Wed 10/28/2026".
WDIS_SEARCH = "https://www.reddit.com/r/fantasyfootball/search.rss?q=title%3AWDIS&restrict_sr=1&sort=new&limit=10"
REDDIT_SPACING = 12  # seconds between Reddit requests: unauthenticated RSS is tightly rate-limited


def reddit_entries(url: str) -> tuple[str, list[dict[str, str]]]:
    """(status, Atom entries as {title, link, content}) for one feed; never raises."""
    from defusedxml import ElementTree

    try:
        root = ElementTree.fromstring(get_bytes(url))
    except urllib.error.HTTPError as e:
        return f"{e.code} {e.reason}", []
    except (OSError, ElementTree.ParseError) as e:
        return f"{type(e).__name__}: {e}", []
    entries = [
        {
            "title": e.findtext(f"{ATOM}title", ""),
            "link": (e.find(f"{ATOM}link").get("href", "") if e.find(f"{ATOM}link") is not None else ""),
            "content": e.findtext(f"{ATOM}content", ""),
        }
        for e in root.iter(f"{ATOM}entry")
    ]
    return "200", entries


def test_reddit_threads_and_comments_via_rss():
    # Step 6b's path: find the newest WDIS thread with the search feed, then read its comments feed.
    status, threads = reddit_entries(WDIS_SEARCH)
    print(f"search: {status}, {len(threads)} threads")
    for t in threads[:5]:
        print(f"  {t['title']}  <- {t['link']}")
    assert status == "200" and threads, f"search feed: {status}"
    thread = threads[0]
    assert re.search(r"/comments/\w+/", thread["link"]), thread["link"]
    time.sleep(REDDIT_SPACING)
    status, comments = reddit_entries(thread["link"].rstrip("/") + "/.rss?limit=50")
    print(f"comments of {thread['title']!r}: {status}, {len(comments)} entries")
    for c in comments[1:4]:  # the first entry is the post itself
        text = re.sub(r"<[^>]+>", " ", c["content"])
        print("  -", " ".join(text.split())[:160])
    assert status == "200" and len(comments) > 1, f"comments feed: {status}"
