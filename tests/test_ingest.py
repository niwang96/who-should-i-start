import polars as pl
import pytest

from startsit.db import Database
from startsit.features import compute_usage_features
from startsit.ingest import nflverse
from startsit.ingest.__main__ import run


def stat_row(player_id, week, position="WR", season_type="REG", **kw):
    row = dict(
        player_id=player_id, season=2026, week=week, season_type=season_type, position=position,
        team="DET", opponent_team="GB", targets=5, carries=0, receptions=4,
        target_share=0.2, air_yards_share=0.25, wopr=0.3, fantasy_points=6.0, fantasy_points_ppr=10.0,
    )
    return row | kw


def make_raw(stats=(), snaps=(), opp=(), injuries=()):
    players = pl.DataFrame(
        dict(
            gsis_id=["G1", "G2", "G3"], pfr_id=["P1", "P2", None], display_name=["Wide Out", "Snap Only", "Lineman"],
            position=["WR", "FB", "OL"], latest_team=["DET", "GB", "DET"],
        )
    )
    snap_schema = dict(
        season=pl.Int64, week=pl.Int64, game_type=pl.String, pfr_player_id=pl.String, position=pl.String,
        team=pl.String, opponent=pl.String, offense_snaps=pl.Float64, offense_pct=pl.Float64,
    )
    opp_schema = dict(
        player_id=pl.String, season=pl.Int64, week=pl.Float64,
        total_fantasy_points_exp=pl.Float64, receptions_exp=pl.Float64,
    )
    inj_schema = dict(
        gsis_id=pl.String, season=pl.Int64, week=pl.Int64, game_type=pl.String, position=pl.String, team=pl.String,
        report_status=pl.String, practice_status=pl.String,
        report_primary_injury=pl.String, practice_primary_injury=pl.String,
    )
    return nflverse.RawNflverse(
        player_stats=pl.DataFrame(list(stats)) if stats else pl.DataFrame([stat_row("G1", 1)]).clear(),
        snap_counts=pl.DataFrame(list(snaps), schema=snap_schema),
        ff_opportunity=pl.DataFrame(list(opp), schema=opp_schema),
        injuries=pl.DataFrame(list(injuries), schema=inj_schema),
        players=players,
    )


def snap(pfr, week, pct, position="WR", game_type="REG"):
    return dict(season=2026, week=week, game_type=game_type, pfr_player_id=pfr, position=position,
                team="DET", opponent="GB", offense_snaps=pct * 60, offense_pct=pct)


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


def test_players_keeps_listed_position_and_latest_team():
    raw = make_raw(stats=[stat_row("G1", 1, team="DET"), stat_row("G1", 2, team="CHI")], snaps=[snap("P2", 1, 0.3, position="RB")])
    players = nflverse.build_players(raw, nflverse.build_player_weeks(raw))
    assert players["player_id"].to_list() == ["G1", "G2"]
    assert players.filter(pl.col("player_id") == "G1")["team"].item() == "CHI"
    assert players.filter(pl.col("player_id") == "G2")["position"].item() == "FB"


def weeks_frame(rows):
    base = dict(season=2026, targets=5, carries=1, target_share=0.2, fp_std=5.0, fp_half=7.0, fp_ppr=9.0,
                xfp_std=6.0, xfp_half=8.0, xfp_ppr=10.0)
    return pl.DataFrame([base | r for r in rows])


def test_features_use_last_three_games_and_trend():
    weeks = weeks_frame(
        [dict(player_id="A", week=w, snap_pct=p, fp_ppr=fp) for w, p, fp in
         [(1, 0.4, 1.0), (2, 0.5, 2.0), (3, 0.6, 3.0), (5, 0.8, 4.0), (6, 0.8, 5.0), (7, 0.8, 6.0)]]
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
        [dict(player_id="A", week=1, snap_pct=0.9), dict(player_id="B", week=1, snap_pct=0.5),
         dict(player_id="B", week=4, snap_pct=0.5)]
    )
    a = compute_usage_features(weeks).filter((pl.col("player_id") == "A") & (pl.col("through_week") == 4))
    assert a["last_game_week"].item() == 1


def test_run_is_idempotent():
    raw = make_raw(
        stats=[stat_row("G1", 1), stat_row("G1", 2)],
        snaps=[snap("P1", 1, 0.8), snap("P1", 2, 0.9)],
        injuries=[dict(gsis_id="G1", season=2026, week=2, game_type="REG", position="WR", team="DET",
                       report_status="Questionable", practice_status="Limited Participation in Practice",
                       report_primary_injury="Hamstring", practice_primary_injury="Hamstring")],
    )
    db = Database(":memory:")
    first = run(db, raw, 2026)
    second = run(db, raw, 2026)
    assert first == second == {"players": 1, "player_weeks": 2, "injuries": 1, "usage_features": 2}
    assert db.query("select count(*) n from player_weeks")[0]["n"] == 2
    assert db.query("select injury from injuries")[0]["injury"] == "Hamstring"
    assert db.query("select count(*) n from ingest_runs")[0]["n"] == 8


def test_upsert_updates_existing_rows():
    db = Database(":memory:")
    db.upsert("players", [dict(player_id="G1", name="A", position="WR", team="DET", pfr_id=None)])
    db.upsert("players", [dict(player_id="G1", name="A", position="WR", team="CHI", pfr_id=None)])
    assert db.query("select team from players") == [{"team": "CHI"}]
