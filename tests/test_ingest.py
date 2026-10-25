import sqlite3
from datetime import UTC, datetime
from typing import TypedDict

import polars as pl
import pytest

from startsit.db import TABLES, Database, Table
from startsit.features import compute_usage_features
from startsit.ingest import news, nflverse
from startsit.ingest.__main__ import run, run_news
from startsit.llm import BudgetedLLM, FakeProvider


def stat_row(player_id, week, position="WR", season_type="REG", **kw):
    row = dict(
        player_id=player_id,
        season=2026,
        week=week,
        season_type=season_type,
        position=position,
        team="DET",
        opponent_team="GB",
        targets=5,
        carries=0,
        receptions=4,
        target_share=0.2,
        air_yards_share=0.25,
        wopr=0.3,
        fantasy_points=6.0,
        fantasy_points_ppr=10.0,
    )
    return row | kw


def make_raw(stats=(), snaps=(), opp=(), injuries=(), games=(), rosters=()):
    players = pl.DataFrame(
        dict(
            gsis_id=["G1", "G2", "G3"],
            pfr_id=["P1", "P2", None],
            display_name=["Wide Out", "Snap Only", "Lineman"],
            position=["WR", "FB", "OL"],
            latest_team=["DET", "GB", "DET"],
        )
    )
    snap_schema = dict(
        season=pl.Int64,
        week=pl.Int64,
        game_type=pl.String,
        pfr_player_id=pl.String,
        position=pl.String,
        team=pl.String,
        opponent=pl.String,
        offense_snaps=pl.Float64,
        offense_pct=pl.Float64,
    )
    opp_schema = dict(
        player_id=pl.String,
        season=pl.Int64,
        week=pl.Float64,
        total_fantasy_points_exp=pl.Float64,
        receptions_exp=pl.Float64,
    )
    inj_schema = dict(
        gsis_id=pl.String,
        season=pl.Int64,
        week=pl.Int64,
        game_type=pl.String,
        position=pl.String,
        team=pl.String,
        report_status=pl.String,
        practice_status=pl.String,
        report_primary_injury=pl.String,
        practice_primary_injury=pl.String,
    )
    roster_schema = dict(
        gsis_id=pl.String,
        season=pl.Int64,
        week=pl.Int64,
        game_type=pl.String,
        position=pl.String,
        full_name=pl.String,
        team=pl.String,
        status=pl.String,
        pfr_id=pl.String,
    )
    games_schema = dict(
        game_id=pl.String,
        season=pl.Int64,
        game_type=pl.String,
        week=pl.Int64,
        gameday=pl.String,
        gametime=pl.String,
        home_team=pl.String,
        away_team=pl.String,
        spread_line=pl.Float64,
        total_line=pl.Float64,
    )
    return nflverse.RawNflverse(
        player_stats=pl.DataFrame(list(stats)) if stats else pl.DataFrame([stat_row("G1", 1)]).clear(),
        snap_counts=pl.DataFrame(list(snaps), schema=snap_schema),
        ff_opportunity=pl.DataFrame(list(opp), schema=opp_schema),
        injuries=pl.DataFrame(list(injuries), schema=inj_schema),
        players=players,
        rosters=pl.DataFrame(list(rosters), schema=roster_schema),
        schedules=pl.DataFrame(list(games), schema=games_schema),
    )


def game(week, home, away, gameday=None, game_type="REG"):
    return dict(
        game_id=f"2026_{week:02d}_{away}_{home}",
        season=2026,
        game_type=game_type,
        week=week,
        gameday=gameday or f"2026-09-{week + 9:02d}",
        gametime="13:00",
        home_team=home,
        away_team=away,
        spread_line=2.5,
        total_line=47.5,
    )


def roster(gsis, week, team, status="ACT", position="WR", name=None):
    return dict(
        gsis_id=gsis,
        season=2026,
        week=week,
        game_type="REG",
        position=position,
        full_name=name or gsis,
        team=team,
        status=status,
        pfr_id=None,
    )


def snap(pfr, week, pct, position="WR", game_type="REG"):
    return dict(
        season=2026,
        week=week,
        game_type=game_type,
        pfr_player_id=pfr,
        position=position,
        team="DET",
        opponent="GB",
        offense_snaps=pct * 60,
        offense_pct=pct,
    )


def test_player_weeks_joins_snaps_and_expected_points():
    raw = make_raw(
        stats=[stat_row("G1", 1), stat_row("G1", 20, season_type="POST")],
        snaps=[snap("P1", 1, 0.8), snap("P2", 1, 0.3, position="RB")],
        opp=[dict(player_id="G1", season=2026, week=1.0, total_fantasy_points_exp=12.0, receptions_exp=4.0)],
    )
    weeks = nflverse.build_player_weeks(raw)
    g1 = weeks.filter(pl.col("player_id") == "G1").row(0, named=True)
    assert weeks.height == 2  # postseason row dropped
    assert g1["snap_pct"] == 0.8
    assert g1["fp_half"] == 8.0
    assert (g1["xfp_ppr"], g1["xfp_half"], g1["xfp_std"]) == (12.0, 10.0, 8.0)

    # Snaps but no box-score line: kept as a zero-point game, not dropped.
    g2 = weeks.filter(pl.col("player_id") == "G2").row(0, named=True)
    assert g2["snap_pct"] == 0.3
    assert g2["fp_ppr"] == 0.0 and g2["targets"] == 0 and g2["xfp_ppr"] == 0.0
    assert g2["team"] == "DET"


def test_players_fall_back_to_last_game_team_and_listed_position():
    raw = make_raw(
        stats=[stat_row("G1", 1, team="DET"), stat_row("G1", 2, team="CHI")], snaps=[snap("P2", 1, 0.3, position="RB")]
    )
    players = nflverse.build_players(raw, nflverse.build_player_weeks(raw))
    assert players["player_id"].to_list() == ["G1", "G2"]
    assert players.filter(pl.col("player_id") == "G1")["team"].item() == "CHI"
    assert players.filter(pl.col("player_id") == "G2")["position"].item() == "FB"


def test_players_prefer_roster_team_and_include_players_who_havent_played():
    # G1 played for DET, then was traded to ATL; R1 is on a roster but has no games yet.
    raw = make_raw(
        stats=[stat_row("G1", 1, team="DET")],
        rosters=[
            roster("G1", 1, "DET"),
            roster("G1", 2, "ATL"),
            roster("R1", 2, "GB", status="EXE", position="RB", name="Returning Back"),
            roster("OL1", 2, "GB", position="OL"),
        ],
    )
    players = nflverse.build_players(raw, nflverse.build_player_weeks(raw))
    assert players.select("player_id", "name", "position", "team").rows() == [
        ("G1", "Wide Out", "WR", "ATL"),
        ("R1", "Returning Back", "RB", "GB"),
    ]
    roster_weeks = nflverse.build_roster_weeks(raw)
    assert roster_weeks.filter(pl.col("player_id") == "R1").select("week", "team", "status").rows() == [
        (2, "GB", "EXE")
    ]


def weeks_frame(rows):
    base = dict(
        season=2026,
        targets=5,
        carries=1,
        target_share=0.2,
        fp_std=5.0,
        fp_half=7.0,
        fp_ppr=9.0,
        xfp_std=6.0,
        xfp_half=8.0,
        xfp_ppr=10.0,
    )
    return pl.DataFrame([base | r for r in rows])


def test_features_use_last_three_games_and_trend():
    weeks = weeks_frame(
        [
            dict(player_id="A", week=w, snap_pct=p, fp_ppr=fp)
            for w, p, fp in [(1, 0.4, 1.0), (2, 0.5, 2.0), (3, 0.6, 3.0), (5, 0.8, 4.0), (6, 0.8, 5.0), (7, 0.8, 6.0)]
        ]
    )
    f = compute_usage_features(weeks)
    latest = f.filter(pl.col("through_week") == 7).row(0, named=True)
    assert latest["games"] == 6 and latest["last_game_week"] == 7
    assert latest["snap_pct_l3"] == pytest.approx(0.8)
    assert latest["snap_pct_trend"] == pytest.approx(0.3)
    assert latest["fp_ppr_l3"] == pytest.approx(5.0)
    assert latest["fp_minus_xfp_ppr_l3"] == pytest.approx(-5.0)
    assert latest["opportunities_l3"] == pytest.approx(6.0)

    # Point-in-time: week-3 features must not see later games.
    wk3 = f.filter(pl.col("through_week") == 3).row(0, named=True)
    assert wk3["games"] == 3 and wk3["fp_ppr_l3"] == pytest.approx(2.0)
    assert wk3["snap_pct_trend"] is None  # no games before the window


def test_features_mark_stale_players():
    weeks = weeks_frame(
        [
            dict(player_id="A", week=1, snap_pct=0.9),
            dict(player_id="B", week=1, snap_pct=0.5),
            dict(player_id="B", week=4, snap_pct=0.5),
        ]
    )
    a = compute_usage_features(weeks).filter((pl.col("player_id") == "A") & (pl.col("through_week") == 4))
    assert a["last_game_week"].item() == 1


def test_run_is_idempotent(db):
    raw = make_raw(
        stats=[stat_row("G1", 1), stat_row("G1", 2)],
        snaps=[snap("P1", 1, 0.8), snap("P1", 2, 0.9)],
        injuries=[
            dict(
                gsis_id="G1",
                season=2026,
                week=2,
                game_type="REG",
                position="WR",
                team="DET",
                report_status="Questionable",
                practice_status="Limited Participation in Practice",
                report_primary_injury="Hamstring",
                practice_primary_injury="Hamstring",
            )
        ],
        games=[game(1, "DET", "GB"), game(2, "CHI", "DET"), dict(game(22, "DET", "KC"), game_type="POST")],
        rosters=[roster("G1", 1, "DET"), roster("G1", 2, "DET")],
    )
    first = run(db, raw, 2026)
    second = run(db, raw, 2026)
    assert (
        first
        == second
        == {
            "players": 1,
            "roster_weeks": 2,
            "player_weeks": 2,
            "injuries": 1,
            "usage_features": 2,
            "games": 2,
        }
    )
    assert db.query("select count(*) as n from player_weeks")[0]["n"] == 2
    assert db.query("select injury from injuries")[0]["injury"] == "Hamstring"
    assert db.query("select count(*) as n from ingest_runs")[0]["n"] == 12
    assert [r["source"] for r in db.recent_runs(2)] and len(db.recent_runs(100)) == 12


def test_upsert_updates_existing_rows(db):
    db.upsert("players", [dict(player_id="G1", name="A", position="WR", team="DET", pfr_id=None)])
    db.upsert("players", [dict(player_id="G1", name="A", position="WR", team="CHI", pfr_id=None)])
    assert db.query("select team from players") == [{"team": "CHI"}]


class KeysOnlyRow(TypedDict):
    k: str


def test_upsert_of_key_only_rows_ignores_duplicates(db, monkeypatch):
    # A table whose columns are all key columns gets DO NOTHING (DO UPDATE SET would be invalid SQL).
    db.execute("CREATE TABLE keys_only (k TEXT PRIMARY KEY)")
    monkeypatch.setitem(TABLES, "keys_only", Table("keys_only", KeysOnlyRow, ("k",)))
    assert db.upsert("keys_only", [dict(k="a"), dict(k="a")]) == 2
    assert db.query("select k from keys_only") == [{"k": "a"}]


def test_table_ddl_is_generated_from_the_row_type():
    ddl = TABLES["games"].ddl()
    assert "game_id TEXT NOT NULL" in ddl and "gameday TEXT," in ddl and "PRIMARY KEY (game_id)" in ddl
    assert TABLES["usage_features"].polars_schema()["snap_pct_l3"] == pl.Float64


RSS = b"""<rss><channel><item><title>Puka Nacua questionable</title><link>https://example.com/1</link>
<pubDate>Sat, 24 Oct 2026 10:00:00 GMT</pubDate></item></channel></rss>"""


def test_run_news_stores_items_and_extracts_facts(db):
    db.upsert("players", [dict(player_id="NACUA", name="Puka Nacua", position="WR", team="LA", pfr_id=None)])
    now = datetime(2026, 10, 24, 20, 0, tzinfo=UTC)
    provider = FakeProvider(
        responses=[
            {
                "facts": [
                    dict(item_id="N1", player="Puka Nacua", team="LAR", event="injury", severity="medium", summary="Q.")
                ]
            }
        ]
    )
    bodies = {news.Feed("RotoWire", "https://example.com/rss"): RSS}
    counts = run_news(db, bodies, BudgetedLLM(provider, db, "news", 5), 2026, now)
    assert counts == {"news_items": 1, "news_extracted": 1, "news_facts": 1, "llm_calls": 1}
    assert db.query("select player_id, summary from news_facts") == [{"player_id": "NACUA", "summary": "Q."}]
    # Without an LLM, or in fetch-only mode, the items are still stored.
    assert run_news(db, bodies, None, 2026, now) == {"news_items": 1}
    assert run_news(db, bodies, BudgetedLLM(provider, db, "news", 5), 2026, now, extract=False) == {"news_items": 1}


def test_new_columns_are_added_to_an_existing_table(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)  # a games table from before betting lines existed
    old.execute(
        "CREATE TABLE games (game_id TEXT PRIMARY KEY, season INTEGER NOT NULL, week INTEGER NOT NULL, "
        "gameday TEXT, home_team TEXT NOT NULL, away_team TEXT NOT NULL)"
    )
    old.execute("INSERT INTO games VALUES ('g1', 2026, 7, NULL, 'PHI', 'DAL')")
    old.commit()
    old.close()
    with Database(path) as db:
        assert db.query("select spread_line, gametime from games") == [{"spread_line": None, "gametime": None}]
        db.upsert(
            "games",
            [
                dict(
                    game_id="g1",
                    season=2026,
                    week=7,
                    gameday=None,
                    gametime=None,
                    home_team="PHI",
                    away_team="DAL",
                    spread_line=3.5,
                    total_line=49.5,
                )
            ],
        )
        assert db.team_game(2026, 7, "DAL")["total_line"] == 49.5


def test_visitor_column_is_added_to_an_old_llm_calls_table(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)  # llm_calls from before per-visitor limits
    old.execute(
        "CREATE TABLE llm_calls (purpose TEXT NOT NULL, model TEXT NOT NULL, ok INTEGER NOT NULL, "
        "called_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    old.execute("INSERT INTO llm_calls (purpose, model, ok, called_at) VALUES ('news', 'm', 1, '2026-10-25 01:00:00')")
    old.commit()
    old.close()
    now = datetime(2026, 10, 25, 2, 0, tzinfo=UTC)
    with Database(path) as db:
        db.record_llm_call("decide", "m", True, now, visitor="v1")
        assert db.llm_calls_since(datetime(2026, 10, 25, tzinfo=UTC)) == 2
        assert db.llm_calls_since(now, visitor="v1") == 1


def test_postgres_tables_have_row_level_security(db):
    if not db.is_postgres:
        pytest.skip("Postgres only")
    rows = db.query("SELECT relname, relrowsecurity FROM pg_class WHERE relname IN ('players', 'llm_calls')")
    assert {r["relname"]: r["relrowsecurity"] for r in rows} == {"players": True, "llm_calls": True}


def test_optional_steps_log_failures_without_stopping(caplog):
    from startsit.ingest.__main__ import _optional

    def boom():
        raise RuntimeError("source down")

    _optional("rankings re-check", boom)
    assert "rankings re-check failed; continuing" in caplog.text
