from datetime import UTC, date, datetime

import polars as pl

from startsit.ingest import rankings
from startsit.ingest.__main__ import run_rankings


def ts(utc: str) -> str:
    return str(int(datetime.fromisoformat(utc).replace(tzinfo=UTC).timestamp()))


GAMES = pl.DataFrame(
    dict(
        season=[2026, 2026, 2026, 2026],
        week=[6, 7, 7, 8],
        gameday=["2026-10-18", "2026-10-22", "2026-10-25", "2026-11-01"],
    )
)


def rank_row(fp_id, name, page="ppr-wr", pos="WR", ecr="16.2", r2p="15.1", kickoff="2026-10-25T17:00:00", **kw):
    row = dict(
        page=page,
        page_pos=pos,
        scrape_date="2026-10-23",
        fantasypros_id=fp_id,
        player_name=name,
        team="PHI",
        player_opponent="vs. DAL",
        ecr=ecr,
        sd="2.53",
        best="9",
        worst="25",
        r2p_pts=r2p,
        pos_rank=f"{pos}16",
        start_sit_grade="A-",
        player_game_status="scheduled",
        player_game_kickoff_ts=ts(kickoff),
        player_page_url=f"https://www.fantasypros.com/nfl/players/{fp_id}.php",
    )
    return row | kw


def raw(rows, ids=None):
    ids = ids or [
        dict(fantasypros_id="19222", gsis_id="G1"),
        dict(fantasypros_id="23180", gsis_id="G2"),
        dict(fantasypros_id="23180", gsis_id="G2-dup"),
        dict(fantasypros_id=None, gsis_id="G9"),
    ]
    return rankings.RawRankings(
        rankings=pl.DataFrame(rows, schema={k: pl.String for k in rows[0]}),
        player_ids=pl.DataFrame(ids, schema={"fantasypros_id": pl.String, "gsis_id": pl.String}),
    )


def test_infer_week_uses_eastern_dates_and_most_common_week():
    # Thursday 8:15pm ET is after midnight UTC; it must still count as Thursday's game.
    kickoffs = pl.Series([ts("2026-10-23T00:15:00"), ts("2026-10-25T17:00:00"), ts("2026-10-25T20:25:00"), None])
    assert rankings.infer_week(kickoffs, GAMES) == 7
    assert rankings.infer_week(pl.Series([ts("2030-01-01T17:00:00")]), GAMES) is None


def test_build_rankings_maps_ids_and_keeps_skill_positions():
    rows = [
        rank_row("19222", "DeVonta Smith"),
        rank_row("23180", "Puka Nacua", ecr="1.2", r2p="24.5"),
        rank_row("99999", "Unmapped Guy", page="ppr-te", pos="TE", r2p="NA"),
        rank_row("11111", "Some Kicker", page="k", pos="K"),
        rank_row("22222", "Some Linebacker", page="lb", pos="LB"),
    ]
    out = rankings.build_rankings(raw(rows), GAMES, 2026)
    assert sorted(out["player_name"].to_list()) == ["DeVonta Smith", "Puka Nacua", "Unmapped Guy"]
    assert out["week"].unique().to_list() == [7]
    smith = out.filter(pl.col("fantasypros_id") == "19222").row(0, named=True)
    assert smith["player_id"] == "G1"
    assert (smith["ecr"], smith["ecr_sd"], smith["ecr_best"], smith["proj_ppr"]) == (16.2, 2.53, 9.0, 15.1)
    assert out.filter(pl.col("fantasypros_id") == "23180")["player_id"].item() == "G2"  # first of duplicate ids
    unmapped = out.filter(pl.col("fantasypros_id") == "99999").row(0, named=True)
    assert unmapped["player_id"] is None and unmapped["proj_ppr"] is None


def test_build_rankings_skips_a_scrape_from_another_season():
    out = rankings.build_rankings(raw([rank_row("19222", "DeVonta Smith")]), GAMES, 2025)
    assert out.is_empty() and out.columns == list(rankings.RANKINGS_SCHEMA)


def test_run_rankings_archives_each_scrape_once(db):
    first = raw([rank_row("19222", "DeVonta Smith")])
    today = date(2026, 10, 24)
    assert run_rankings(db, first, GAMES, 2026, today) == 1
    assert run_rankings(db, first, GAMES, 2026, today) == 1  # same scrape re-ingested: no duplicate
    later = raw([rank_row("19222", "DeVonta Smith", scrape_date="2026-10-24", ecr="14.0")])
    run_rankings(db, later, GAMES, 2026, today)
    assert db.query("select scrape_date, ecr from rankings order by scrape_date") == [
        {"scrape_date": "2026-10-23", "ecr": 16.2},
        {"scrape_date": "2026-10-24", "ecr": 14.0},
    ]
    assert db.latest_ranking("G1", 2026, 7)["ecr"] == 14.0
    assert db.latest_rankings_week(2026) == 7
    assert run_rankings(db, first, GAMES, 2025, today) == 0


def test_run_rankings_warns_about_a_stale_scrape(db, caplog):
    run_rankings(db, raw([rank_row("19222", "DeVonta Smith")]), GAMES, 2026, today=date(2026, 10, 24))
    assert "days old" not in caplog.text
    run_rankings(db, raw([rank_row("19222", "DeVonta Smith")]), GAMES, 2026, today=date(2026, 10, 30))
    assert "rankings scrape is 7 days old" in caplog.text
