import io
import urllib.error
from datetime import UTC, datetime, timedelta

import pytest
from test_evidence import player

from startsit.db import Database
from startsit.ingest import reddit, reddit_extract
from startsit.ingest.__main__ import run_reddit
from startsit.llm import BudgetedLLM, FakeProvider, LLMError

NOW = datetime(2026, 10, 28, 15, 0, tzinfo=UTC)
THREAD_URL = "https://www.reddit.com/r/fantasyfootball/comments/1xgpg57/official_wdis_wr_wed_10282026/"


def feed(*entries: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">'
        + "".join(entries)
        + "</feed>"
    ).encode()


def entry(id_, title, link, author="/u/fan", content="", published="2026-10-28T14:00:00+00:00"):
    return (
        f'<entry><author><name>{author}</name></author><content type="html">{content}</content>'
        f'<id>{id_}</id><link href="{link}"/><published>{published}</published><title>{title}</title></entry>'
    )


SEARCH = feed(
    entry("t3_new1", "Official: [WDIS WR] - Wed 10/28/2026", THREAD_URL),
    entry("t3_new2", "Official: [WDIS RB] - Wed 10/28/2026", THREAD_URL.replace("1xgpg57", "1xgpg3d")),
    entry("t3_new1", "Official: [WDIS WR] - Wed 10/28/2026", THREAD_URL),  # listed twice
    entry("t3_old", "Official: [WDIS WR] - Tue 10/27/2026", "https://www.reddit.com/r/x/comments/old/"),
    entry("t3_other", "Rant: my league", "https://www.reddit.com/r/x/comments/rant/"),
)
COMMENTS = feed(
    entry("t3_new1", "Official: [WDIS WR] - Wed 10/28/2026", THREAD_URL, content="the post itself"),
    entry("t1_a", "/u/fan on WDIS", THREAD_URL + "a/", content="&lt;p&gt;AJB and Moore&lt;/p&gt;"),
    entry("t1_b", "/u/bot", THREAD_URL + "b/", author="/u/AutoModerator", content="Rules..."),
    entry("t1_c", "/u/fan2", THREAD_URL + "c/", author="/u/fan2", content="PPR pick 1: Cooper or Odunze?"),
    entry("t1_d", "/u/gone", THREAD_URL + "d/", author="[deleted]", content="removed"),
    entry("t1_e", "/u/empty", THREAD_URL + "e/", content=" "),
    entry("t1_f", "/u/future", THREAD_URL + "f/", content="Coker", published="2026-10-29T00:00:00+00:00"),
)


def test_latest_threads_keeps_the_newest_days_wdis_threads_once():
    threads = reddit.latest_threads(SEARCH)
    assert [(t.thread_id, t.day) for t in threads] == [("t3_new1", "10/28/2026"), ("t3_new2", "10/28/2026")]
    assert reddit.latest_threads(b"not xml") == [] and reddit.latest_threads(feed()) == []


def test_comments_skip_the_post_bots_deleted_and_empty_and_cap_future_dates():
    thread = reddit.latest_threads(SEARCH)[0]
    rows = reddit.comments(COMMENTS, thread, NOW)
    assert [(r["comment_id"], r["text"]) for r in rows] == [
        ("t1_a", "AJB and Moore"),
        ("t1_c", "PPR pick 1: Cooper or Odunze?"),
        ("t1_f", "Coker"),
    ]
    assert rows[0]["author"] == "fan" and rows[0]["thread_title"] == thread.title
    assert rows[2]["published_at"] == "2026-10-28T15:00:00+00:00"  # capped at fetch time
    assert reddit.comments_url(thread) == THREAD_URL.rstrip("/") + "/.rss?limit=100"


def http_error(code, headers=None):
    return urllib.error.HTTPError("u", code, "x", headers or {}, io.BytesIO(b""))


def test_reddit_get_waits_out_one_rate_limit_and_gives_up_quietly(monkeypatch):
    replies = [http_error(429, {"x-ratelimit-reset": "18"}), b"ok"]

    def get(url):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(reddit, "get_bytes", get)
    slept: list[float] = []
    assert reddit.reddit_get("u", slept.append) == b"ok" and slept == [20]

    for failure in ([http_error(429, {"x-ratelimit-reset": "18"})] * 2, [http_error(429, {})], [http_error(403)]):
        replies[:] = failure
        assert reddit.reddit_get("u", lambda s: None) is None  # twice limited, reset unknown, or blocked
    replies[:] = [ConnectionResetError("reset")]
    assert reddit.reddit_get("u", lambda s: None) is None


def test_fetch_comments_reads_each_thread_a_minute_apart_and_stops_when_refused(db):
    pages = {reddit.SEARCH_URL: SEARCH, reddit.comments_url(reddit.latest_threads(SEARCH)[0]): COMMENTS}
    slept: list[float] = []
    assert reddit.fetch_comments(db, 2026, NOW, get=pages.get, sleep=slept.append) == 3
    assert slept == [reddit.SPACING] * 2  # before each thread; the second's feed was refused, so it stopped
    assert db.last_run_at(reddit.RUN_SOURCE) is not None
    assert reddit.fetch_comments(db, 2026, NOW, get=lambda url: None, sleep=slept.append) == 0  # search refused


def test_a_fetch_is_due_every_six_hours(db):
    assert reddit.due(db, NOW)
    db.record_run(reddit.RUN_SOURCE, 2026, None, 1)
    now = datetime.fromisoformat(db.last_run_at(reddit.RUN_SOURCE)).replace(tzinfo=UTC)
    assert not reddit.due(db, now + timedelta(hours=5))
    assert reddit.due(db, now + timedelta(hours=6))


def test_a_fresh_fetch_on_sunday_morning_before_the_benchmark(db):
    sunday_7am_et = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)  # EST from Nov 1
    db.execute(
        "INSERT INTO ingest_runs (source, season, max_week, row_count, finished_at) VALUES (?, ?, ?, ?, ?)",
        (reddit.RUN_SOURCE, 2026, None, 1, "2026-11-01 11:00:00"),  # 6am ET
    )
    assert not reddit.due(db, sunday_7am_et)  # within 6 hours, before the Sunday window
    assert reddit.due(db, sunday_7am_et + timedelta(hours=1, minutes=22))  # 8:22am ET: none since 8am yet
    assert not reddit.due(db, datetime(2026, 11, 1, 15, 30, tzinfo=UTC))  # 10:30am: window over


# --- extraction ------------------------------------------------------------------------


def seed_comments(db):
    db.upsert(
        "players",
        [
            player("AJB", "A.J. Brown", "WR", "PHI"),
            player("MOORE1", "D.J. Moore", "WR", "CHI"),
            player("COOP", "Omar Cooper Jr.", "WR", "NYJ"),
            player("SMITH1", "Jaylon Smith", "WR", "LV"),
            player("SMITH2", "Jaylon Smith", "WR", "LAC"),
        ],
    )
    thread = reddit.latest_threads(SEARCH)[0]
    db.upsert("reddit_comments", reddit.comments(COMMENTS, thread, NOW))


VOTES = {
    "votes": [
        {"comment_id": "C2", "player": "A.J. Brown", "team": "", "verdict": "start", "reason": "safe floor"},
        {"comment_id": "C2", "player": "D.J. Moore", "team": "CHI", "verdict": "start", "reason": ""},
        {"comment_id": "C2", "player": "A.J. Brown", "team": "", "verdict": "start", "reason": "again"},  # dup
        {"comment_id": "C3", "player": "Moore", "team": "", "verdict": "start", "reason": ""},  # can't match
        {"comment_id": "C3", "player": "Jaylon Smith", "team": "", "verdict": "sit", "reason": ""},  # ambiguous
        {"comment_id": "C9", "player": "A.J. Brown", "team": "", "verdict": "start", "reason": ""},  # no such id
        {"comment_id": "C2", "player": "A.J. Brown", "team": "", "verdict": "maybe", "reason": ""},  # bad verdict
        "garbage",
    ]
}


def test_extraction_turns_replies_into_matched_votes_and_caches_every_comment(db):
    seed_comments(db)
    provider = FakeProvider(responses=[VOTES])
    result = reddit_extract.extract_votes(db, BudgetedLLM(provider, db, "reddit", 5), NOW)
    assert (result.comments_processed, result.votes, result.calls) == (3, 2, 1)
    # Newest first: the reply posted after fetch time (capped to it), then the two from 14:00.
    assert provider.prompts[0].startswith("Comments:\n[C1] Coker\n[C2] AJB and Moore\n[C3] PPR pick 1")
    rows = db.query("SELECT player_id, verdict, reason, url FROM crowd_votes ORDER BY player_id")
    assert rows == [
        {"player_id": "AJB", "verdict": "start", "reason": "again", "url": THREAD_URL + "a/"},
        {"player_id": "MOORE1", "verdict": "start", "reason": None, "url": THREAD_URL + "a/"},
    ]
    counts = {r["comment_id"]: r["vote_count"] for r in db.query("SELECT * FROM reddit_extractions")}
    assert counts == {"t1_a": 2, "t1_c": 0, "t1_f": 0}
    again = reddit_extract.extract_votes(db, BudgetedLLM(FakeProvider(), db, "reddit", 5), NOW)
    assert again.calls == 0  # every comment is cached
    assert db.crowd_votes("AJB", "2026-10-28T00:00:00+00:00", "2026-10-29T00:00:00+00:00")[0]["verdict"] == "start"


@pytest.mark.parametrize(
    "budget,responses,stopped",
    [(0, (), "daily reddit budget of 0 LLM calls is spent"), (5, (LLMError("boom"),), "boom")],
)
def test_extraction_stops_and_leaves_comments_for_the_next_run(db, budget, responses, stopped):
    seed_comments(db)
    provider = FakeProvider(responses=list(responses))  # a fresh list per backend
    result = reddit_extract.extract_votes(db, BudgetedLLM(provider, db, "reddit", budget), NOW)
    assert result.stopped_early == stopped and result.comments_processed == 0
    assert len(db.unextracted_reddit("2026-10-01T00:00:00+00:00", 10)) == 3


def test_the_ingest_step_fetches_when_due_and_extracts_when_the_ai_is_available(tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO")
    path = str(tmp_path / "r.db")
    fetched = []
    monkeypatch.setattr(reddit, "fetch_comments", lambda db, season, now: fetched.append(now) or 7)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with Database(path) as db:
        assert run_reddit(db, 2026, NOW) == {"reddit_comments": 7}
        assert "reddit extraction disabled: GEMINI_API_KEY is not set" in caplog.text
        db.record_run(reddit.RUN_SOURCE, 2026, None, 7)  # what fetch_comments records
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(
            reddit_extract, "extract_votes", lambda db, llm, now: reddit_extract.ExtractionResult(4, 2, 1)
        )
        counts = run_reddit(db, 2026, datetime.now(UTC))
    assert counts == {"reddit_comments": 0, "reddit_extracted": 4, "crowd_votes": 2, "llm_calls": 1}
    assert len(fetched) == 1


def test_a_one_word_name_matches_a_unique_ranked_player(db):
    from test_cli import _week
    from test_evidence import ranking

    seed_comments(db)
    db.upsert("players", [player("KICK", "Jake Moore", "K", "DAL")])
    db.upsert("player_weeks", [_week("COOP", 7)])  # sets the latest season
    db.upsert(
        "rankings",
        [ranking("COOP", 9.0), ranking("SMITH1", 8.0), ranking("KICK", 7.0, position="K")],
    )
    response = {
        "votes": [
            {"comment_id": "C2", "player": "Cooper", "team": "", "verdict": "start", "reason": ""},
            {"comment_id": "C2", "player": "Omar", "team": "", "verdict": "sit", "reason": ""},  # same comment
            {"comment_id": "C3", "player": "Smith", "team": "", "verdict": "start", "reason": ""},  # one ranked
            {"comment_id": "C3", "player": "Moore", "team": "", "verdict": "start", "reason": ""},  # no ranked WR
        ]
    }
    result = reddit_extract.extract_votes(db, BudgetedLLM(FakeProvider(responses=[response]), db, "reddit", 5), NOW)
    assert result.votes == 2
    rows = db.query("SELECT player_id, verdict FROM crowd_votes ORDER BY player_id")
    assert rows == [{"player_id": "COOP", "verdict": "sit"}, {"player_id": "SMITH1", "verdict": "start"}]


def test_no_ranked_pool_without_data(db):
    assert reddit_extract.ranked_players(db, []) == []
