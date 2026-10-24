import pytest

from startsit.db import Database
from startsit.evidence import PlayerLookupError, build_packet, current_week, find_player


def usage(player_id, through_week, last_game_week=None, **kw):
    row = dict(
        player_id=player_id, season=2026, through_week=through_week, games=6,
        last_game_week=last_game_week or through_week, snap_pct_l3=0.85, snap_pct_trend=-0.1, target_share_l3=0.34,
        opportunities_l3=11.7, fp_std_l3=12.0, fp_half_l3=15.0, fp_ppr_l3=18.0, xfp_std_l3=10.0, xfp_half_l3=12.0,
        xfp_ppr_l3=14.0, xfp_ppr_season=14.0, fp_minus_xfp_ppr_l3=4.0,
    )
    return row | kw


def ranking(player_id, proj_ppr, scrape_date="2026-10-23", **kw):
    row = dict(
        season=2026, week=7, scrape_date=scrape_date, fantasypros_id=f"fp-{player_id}", player_id=player_id,
        player_name=player_id, position="WR", team="PHI", opponent="vs. DAL", ecr=16.2, ecr_sd=2.53, ecr_best=9.0,
        ecr_worst=25.0, proj_ppr=proj_ppr, pos_rank="WR16", start_sit_grade="A-", game_status="scheduled",
        source_url=f"https://www.fantasypros.com/nfl/players/{player_id}.php",
    )
    return row | kw


def game(week, home, away):
    return dict(game_id=f"2026_{week:02d}_{away}_{home}", season=2026, week=week, gameday=None,
                home_team=home, away_team=away)


@pytest.fixture
def db():
    db = Database(":memory:")
    db.upsert("players", [
        dict(player_id="SMITH", name="DeVonta Smith", position="WR", team="PHI", pfr_id=None),
        dict(player_id="NACUA", name="Puka Nacua", position="WR", team="LA", pfr_id=None),
        dict(player_id="HALL", name="Breece Hall", position="RB", team="NYJ", pfr_id=None),
        dict(player_id="BYE", name="Bye Guy", position="WR", team="DET", pfr_id=None),
        dict(player_id="GENO", name="Geno Smith", position="QB", team="NYJ", pfr_id=None),
    ])
    db.upsert("games", [
        game(w, home, away)
        for w in range(1, 8)
        for home, away in [("PHI", "DAL"), ("LV", "LA"), ("NYJ", "MIA")] + ([("DET", "GB")] if w != 7 else [])
    ])
    db.upsert("usage_features", [
        usage("SMITH", 5), usage("SMITH", 6),
        usage("NACUA", 6, xfp_ppr_l3=20.0, xfp_std_l3=10.0),
        usage("HALL", 6, last_game_week=3),
        usage("BYE", 6),
        usage("SMITH", 7, snap_pct_l3=0.1),  # after the week asked about: must be ignored
    ])
    db.upsert("rankings", [
        ranking("SMITH", 14.0, scrape_date="2026-10-21"),
        ranking("SMITH", 15.1),
        ranking("NACUA", 24.5, pos_rank="WR1", opponent="at LV"),
        ranking("HALL", 9.0, position="RB", pos_rank="RB30"),
    ])
    db.upsert("injuries", [
        dict(player_id="NACUA", season=2026, week=7, team="LA", report_status=None,
             practice_status="Limited Participation in Practice", injury="Calf"),
        dict(player_id="HALL", season=2026, week=7, team="NYJ", report_status="Out",
             practice_status="Did Not Participate In Practice", injury="Quadricep"),
    ])
    yield db
    db.close()


def test_packet_converts_scoring_and_picks_by_consensus(db):
    p = build_packet(db, ["SMITH", "NACUA"], "half", 2026, 7)
    smith, nacua = p.players
    assert smith.baseline.points == 13.1  # 15.1 PPR - 0.5 x 4 expected catches, from the latest scrape
    assert nacua.baseline.points == 19.5  # 24.5 - 0.5 x 10
    assert (p.baseline_pick, p.baseline_margin) == ("NACUA", 6.4)
    assert [i.id for i in smith.items + nacua.items] == [f"E{n}" for n in range(1, 8)]
    assert p.item_ids == {f"E{n}" for n in range(1, 8)}
    assert smith.items[0].source_url == "https://www.fantasypros.com/nfl/players/SMITH.php"

    text = p.to_text()
    assert text.startswith("Week 7 2026, half-PPR scoring. Baseline (100% FantasyPros consensus) favors Puka Nacua by 6.4 pts.")
    assert "E1 [consensus] FantasyPros consensus WR16 vs. DAL (avg rank 16.2, sd 2.53, range 9-25" in text
    assert "about 13.1 half-PPR (minus 0.5 x 4.0 expected catches/game)" in text
    assert "85% snaps (-10 pts vs prior 3)" in text and "0.1" not in text  # week-7 usage not leaked
    assert "E7 [injury] Week 7 injury report: no game status yet (Calf), practice: Limited" in text


def test_out_bye_and_unranked_players_score_zero(db):
    p = build_packet(db, ["HALL", "BYE", "GENO"], "ppr", 2026, 7)
    hall, bye, geno = p.players
    assert (hall.baseline.points, hall.baseline.note) == (0.0, "ruled Out")
    assert any("has missed NYJ's last 3 games (last played week 3)" in i.text for i in hall.items)
    assert (bye.baseline.points, bye.baseline.note) == (0.0, "on bye")
    assert bye.items[0].text == "DET is on bye in week 7"
    assert not any("has missed" in i.text for i in bye.items)
    assert geno.baseline.note == "not in FantasyPros weekly rankings"
    assert "Baseline (100% FantasyPros consensus) has them tied" in p.to_text()


def test_packet_rejects_bad_input(db):
    with pytest.raises(ValueError):
        build_packet(db, ["SMITH", "SMITH"], "ppr", 2026, 7)
    with pytest.raises(ValueError):
        build_packet(db, ["SMITH", "NACUA"], "superflex", 2026, 7)
    with pytest.raises(PlayerLookupError):
        build_packet(db, ["SMITH", "NOBODY"], "ppr", 2026, 7)


def test_find_player_and_current_week(db):
    assert find_player(db, "puka nacua")["player_id"] == "NACUA"
    assert find_player(db, "Breece")["player_id"] == "HALL"
    with pytest.raises(PlayerLookupError, match="ambiguous"):
        find_player(db, "Smith")
    with pytest.raises(PlayerLookupError, match="no player"):
        find_player(db, "Zzz")
    assert current_week(db, 2026) == 7
    assert current_week(db, 2025) == 1
