"""The command-line entry points and network fetchers, with every external service faked."""

import sys
import urllib.error

import pytest
from test_ingest import make_raw, snap, stat_row
from test_news import RSS
from test_rankings import rank_row
from test_rankings import raw as rankings_raw

from startsit import decide, http
from startsit.db import Database
from startsit.ingest import __main__ as ingest_main
from startsit.ingest import news, rankings
from startsit.llm import FakeProvider


def seed(path):
    with Database(path) as db:
        db.upsert(
            "players",
            [
                dict(player_id="A", name="Alpha Back", position="RB", team="PHI", pfr_id=None),
                dict(player_id="B", name="Bravo Back", position="RB", team="DAL", pfr_id=None),
            ],
        )
        db.upsert(
            "games",
            [
                dict(
                    game_id="g7",
                    season=2026,
                    week=7,
                    gameday="2026-10-25",
                    gametime="13:00",
                    home_team="PHI",
                    away_team="DAL",
                    spread_line=3.0,
                    total_line=45.0,
                )
            ],
        )
        db.upsert("player_weeks", [_week("A", 6)])  # sets the latest season
        db.upsert(
            "rankings",
            [_ranking(pid, proj) for pid, proj in (("A", 12.0), ("B", 9.0))],
        )
    return str(path)


def _week(player_id, week):
    return dict(
        player_id=player_id,
        season=2026,
        week=week,
        team="PHI",
        opponent="NYG",
        offense_snaps=50,
        snap_pct=0.8,
        targets=3,
        carries=15,
        receptions=2,
        target_share=0.1,
        air_yards_share=0.0,
        wopr=0.1,
        fp_ppr=14.0,
        fp_half=13.0,
        fp_std=12.0,
        xfp_ppr=13.0,
        xfp_half=12.0,
        xfp_std=11.0,
    )


def _ranking(player_id, proj):
    return dict(
        season=2026,
        week=7,
        scrape_date="2026-10-24",
        fantasypros_id=f"fp{player_id}",
        player_id=player_id,
        player_name=player_id,
        position="RB",
        team=None,
        opponent=None,
        ecr=10.0,
        ecr_sd=1.0,
        ecr_best=5.0,
        ecr_worst=15.0,
        proj_ppr=proj,
        pos_rank="RB10",
        start_sit_grade=None,
        game_status=None,
        source_url=None,
    )


def run_cli(monkeypatch, main, *args):
    monkeypatch.setattr(sys, "argv", ["prog", *args])
    main()


def test_decide_cli_prints_the_packet(tmp_path, monkeypatch, capsys):
    path = seed(tmp_path / "s.db")
    run_cli(monkeypatch, decide.main, "Alpha Back", "bravo back", "--db", path, "--no-llm")
    out = capsys.readouterr().out
    assert out.startswith("Week 7 2026, PPR scoring. Baseline (100% FantasyPros consensus) favors Alpha Back")
    assert "kickoff Sun 1:00 PM ET" in out


def test_cli_exits_with_a_message_for_a_bad_question(tmp_path, monkeypatch, capsys):
    path = seed(tmp_path / "s.db")
    with pytest.raises(SystemExit) as exited:
        run_cli(monkeypatch, decide.main, "Alpha Back", "Nobody", "--db", path)
    assert exited.value.code == 1 and "no player matches 'Nobody'" in capsys.readouterr().err


def test_decide_cli_with_and_without_the_ai(tmp_path, monkeypatch, capsys):
    path = seed(tmp_path / "s.db")
    run_cli(monkeypatch, decide.main, "Alpha Back", "Bravo Back", "--db", path, "--no-llm")
    assert "Start Alpha Back (medium confidence; baseline)" in capsys.readouterr().out

    answer = {
        "pick": "P1",
        "confidence": "medium",
        "factors": [
            {"text": "Higher projection.", "evidence_ids": ["E2"]},
            {"text": "Favored.", "evidence_ids": ["E3"]},
        ],
        "baseline_disagreement": "",
    }
    monkeypatch.setattr(decide, "GeminiProvider", lambda: FakeProvider(responses=[answer]))
    run_cli(monkeypatch, decide.main, "Alpha Back", "Bravo Back", "--db", path)
    out = capsys.readouterr().out
    assert "Start Alpha Back (medium confidence; llm)\n  - Higher projection. [E2]" in out


def test_decide_llm_is_none_without_a_key_and_carries_visitor_limits(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with Database(tmp_path / "s.db") as db:
        assert decide.decide_llm(db) is None
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        llm = decide.decide_llm(db, visitor="v1")
        assert llm is not None and llm.visitor == "v1" and llm.visitor_limits == decide.VISITOR_LIMITS


def test_ingest_cli_runs_each_mode_with_faked_sources(tmp_path, monkeypatch, caplog):
    path = str(tmp_path / "i.db")
    raw = make_raw(stats=[stat_row("G1", 1)], snaps=[snap("P1", 1, 0.8)])
    monkeypatch.setattr(ingest_main.nflverse, "fetch", lambda season: raw)
    monkeypatch.setattr(ingest_main.rankings, "fetch", lambda: rankings_raw([rank_row("19222", "Wide Out")]))
    monkeypatch.setattr(ingest_main.news, "fetch_all", lambda: {news.Feed("RotoWire", "https://x"): RSS})
    monkeypatch.delenv("FANTASYPROS_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    run_cli(monkeypatch, ingest_main.main, "--season", "2026", "--db", path, "--only", "stats")
    assert "FANTASYPROS_API_KEY not set; projections skipped" in caplog.text
    run_cli(monkeypatch, ingest_main.main, "--season", "2026", "--db", path, "--only", "news-fetch")
    run_cli(monkeypatch, ingest_main.main, "--season", "2026", "--db", path, "--only", "news")
    assert "news extraction disabled: GEMINI_API_KEY is not set" in caplog.text
    with Database(path) as db:
        assert db.query("select count(*) as n from player_weeks")[0]["n"] == 1
        assert db.query("select count(*) as n from news_items")[0]["n"] == 1


def test_news_fetchers_skip_failing_and_empty_feeds(monkeypatch, caplog):
    bodies = {"https://ok": b"<rss/>", "https://empty": b"  ", "https://down": None}

    def fake_get(url, headers=None, timeout=30):
        if bodies[url] is None:
            raise urllib.error.URLError("down")
        return bodies[url]

    monkeypatch.setattr(news, "get_bytes", fake_get)
    feeds = [news.Feed(name, f"https://{name}") for name in ("ok", "empty", "down")]
    assert list(news.fetch_all(feeds).values()) == [b"<rss/>"]
    assert "news feed empty failed: empty response" in caplog.text and "news feed down failed" in caplog.text


def test_rankings_fetch_reads_both_files_as_text_with_na_as_null(monkeypatch):
    csv = {rankings.RANKINGS_URL: b"page,ecr\nppr-wr,NA\n", rankings.PLAYER_IDS_URL: b"fantasypros_id,gsis_id\n1,G1\n"}
    monkeypatch.setattr(rankings, "get_bytes", lambda url, timeout: csv[url])
    raw = rankings.fetch()
    assert raw.rankings.to_dicts() == [{"page": "ppr-wr", "ecr": None}]
    assert raw.player_ids.to_dicts() == [{"fantasypros_id": "1", "gsis_id": "G1"}]


def test_get_bytes_sends_the_user_agent(monkeypatch):
    sent = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"body"

    def fake_urlopen(req, timeout):
        sent.update(req.headers)
        return Response()

    monkeypatch.setattr(http.urllib.request, "urlopen", fake_urlopen)
    assert http.get_bytes("https://x", {"X-Api-Key": "k"}) == b"body"
    assert sent["User-agent"] == http.USER_AGENT and sent["X-api-key"] == "k"
