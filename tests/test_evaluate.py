import json
import sys
from datetime import UTC, datetime

import pytest
from test_cli import _week
from test_evidence import game, player, ranking

from startsit import evaluate
from startsit.db import Database

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)


def week7(player_id, team, fp_ppr):
    return _week(player_id, 7) | dict(team=team, fp_ppr=fp_ppr, fp_half=fp_ppr - 1, fp_std=fp_ppr - 2.004)


def rec(rec_id, players, pick, baseline_pick=None, scoring="ppr", source="llm", confidence="medium", **kw):
    """A logged recommendation; `players` maps player id -> team as asked."""
    row = dict(
        rec_id=rec_id,
        created_at="2026-10-24T18:00:00+00:00",
        season=2026,
        week=7,
        scoring=scoring,
        player_ids=json.dumps(list(players)),
        pick=pick,
        baseline_pick=baseline_pick or pick,
        confidence=confidence,
        source=source,
        model="m",
        decision_json="{}",
        packet_text="",
        packet_json=json.dumps(
            {"players": [{"player_id": p, "name": f"Player {p}", "team": t} for p, t in players.items()]}
        ),
        visitor="v1",
    )
    return row | kw


@pytest.fixture
def week(db):
    """Week 7: PHI-DAL is final (stats loaded), NYJ-MIA isn't yet, GB is on bye."""
    db.upsert("games", [game(7, "PHI", "DAL"), game(7, "NYJ", "MIA")])
    db.upsert("player_weeks", [week7("A", "PHI", 20.0), week7("B", "DAL", 12.0), week7("C", "PHI", 12.0)])
    db.upsert(
        "rankings",
        [
            ranking("A", 14.0, scrape_date="2026-10-23", position="RB", ecr=20.0),
            ranking("B", 11.0, scrape_date="2026-10-23", position="RB", ecr=12.0),
            ranking("B", 11.0, scrape_date="2026-10-26", position="RB", ecr=30.0),  # after the question
            ranking("W", 15.0, scrape_date="2026-10-23", position="WR", ecr=5.0),
        ],
    )
    return db


@pytest.mark.parametrize(
    "player_id,team,expected",
    [
        ("A", "PHI", 20.0),  # played
        ("D", "PHI", 0.0),  # his team's game is in the data, he has no stats: didn't play
        ("D", "GB", 0.0),  # bye
        ("D", None, 0.0),  # no team
        ("D", "NYJ", None),  # game not in the data yet
    ],
)
def test_actual_points(week, player_id, team, expected):
    assert evaluate.actual_points(week, {"player_id": player_id, "team": team}, 2026, 7, "ppr") == expected


def test_actual_points_use_the_question_scoring(week):
    assert evaluate.actual_points(week, {"player_id": "A", "team": "PHI"}, 2026, 7, "std") == 18.0  # rounded


@pytest.mark.parametrize(
    "pick,points,expected",
    [
        ("A", {"A": 20.0, "B": 12.0}, 1),
        ("B", {"A": 20.0, "B": 12.0}, 0),
        ("A", {"A": 12.0, "B": 12.0}, None),  # tie at the top
        ("C", {"A": 12.0, "B": 12.0, "C": 3.0}, 0),  # others tied, the pick lost
        (None, {"A": 1.0, "B": 2.0}, None),
    ],
)
def test_hit(pick, points, expected):
    assert evaluate.hit(pick, points) == expected


@pytest.mark.parametrize(
    "ids,expected",
    [
        (["A", "B"], "B"),  # same position: better rank, from the scrape on or before the day asked
        (["A", "W"], "W"),  # across positions: higher PPR projection
        (["A", "X"], "A"),  # an unranked player loses
        (["X", "Y"], None),  # nobody ranked
    ],
)
def test_consensus_pick(week, ids, expected):
    assert evaluate.consensus_pick(week, rec("r", dict.fromkeys(ids, "PHI"), ids[0]), ids) == expected


def test_consensus_pick_ties_and_missing_projections(db):
    db.upsert(
        "rankings",
        [
            ranking("A", 14.0, position="RB", ecr=10.0),
            ranking("B", 11.0, position="RB", ecr=10.0),
            ranking("C", None, position="WR"),
            ranking("D", None, position="TE"),
        ],
    )
    assert evaluate.consensus_pick(db, rec("r", {"A": "PHI", "B": "PHI"}, "A"), ["A", "B"]) is None
    assert evaluate.consensus_pick(db, rec("r", {"C": "PHI", "D": "PHI"}, "C"), ["C", "D"]) is None


def test_score_pending_scores_only_finished_matchups_once(week):
    week.upsert(
        "recommendations",
        [
            rec("done", {"A": "PHI", "B": "DAL"}, pick="B", baseline_pick="A"),
            rec("waiting", {"A": "PHI", "J": "NYJ"}, pick="A"),
        ],
    )
    assert evaluate.score_pending(week, NOW) == 1
    [row] = week.query("SELECT * FROM evaluations")
    assert json.loads(row["points_json"]) == {"A": 20.0, "B": 12.0}
    assert (row["top_scorer"], row["pick_hit"], row["baseline_hit"]) == ("A", 0, 1)
    assert (row["consensus_pick"], row["consensus_hit"]) == ("B", 0)
    assert row["scored_at"] == "2026-10-27T12:00:00+00:00"
    assert evaluate.score_pending(week, NOW) == 0  # nothing new; "waiting" stays until NYJ's game is in


def test_a_tied_matchup_is_a_push(week):
    week.upsert("recommendations", [rec("tie", {"B": "DAL", "C": "PHI"}, pick="B")])
    evaluate.score_pending(week, NOW)
    [row] = week.query("SELECT top_scorer, pick_hit FROM evaluations")
    assert row == {"top_scorer": None, "pick_hit": None}


def test_report_compares_the_picks_on_the_matchups_the_ai_answered(week):
    ab = {"A": "PHI", "B": "DAL"}
    week.upsert(
        "recommendations",
        [
            rec("ai-right-override", ab, pick="A", baseline_pick="B", confidence="high"),
            rec("ai-wrong", ab, pick="B", baseline_pick="B", source="cache", confidence="low"),
            rec("no-ai", ab, pick="B", source="baseline"),  # counted as a matchup, not in the rates
            rec("bench", ab, pick="A", visitor=evaluate.BENCHMARK),
            rec("backfill", ab, pick="B", visitor=evaluate.BACKFILL),
        ],
    )
    evaluate.score_pending(week, NOW)
    report = evaluate.build_report(week, 2026)
    users = report.totals["users"]
    assert (users.matchups, users.ai_answered) == (3, 2)
    assert (str(users.baseline_all), str(users.consensus_all)) == ("0/3 (0%)", "0/3 (0%)")
    assert (str(users.ai), str(users.baseline), str(users.consensus)) == ("1/2 (50%)", "0/2 (0%)", "0/2 (0%)")
    assert str(users.overrides) == "1/1 (100%)"
    assert (str(users.by_confidence["high"]), str(users.by_confidence["medium"])) == ("1/1 (100%)", "-")
    assert report.weeks["users"][7].matchups == 3
    assert str(report.totals["benchmark"].ai) == "1/1 (100%)"
    assert str(report.totals["backfill"].ai) == "0/1 (0%)"
    [override] = report.overrides["users"]
    assert override == evaluate.Override(7, "ppr", "Player A", 20.0, "Player B", 12.0, 1)
    assert "benchmark" not in report.overrides  # it agreed with the baseline

    text = evaluate.report_text(report)
    assert "users:\n    week 7: 2 AI-answered | AI 1/2 (50%) | baseline 0/2 (0%)" in text
    assert "|| all 3: baseline 0/3 (0%) | consensus 0/3 (0%)" in text
    assert (
        "benchmark:" in text
        and "backfill:" in text
        and "AI by confidence: high 1/1 (100%), medium -, low 0/1 (0%)" in text
    )


def test_report_without_data():
    with Database(":memory:") as db:
        assert evaluate.report_text(evaluate.build_report(db, 2026)).endswith("Nothing scored yet.")


def test_cli_scores_and_prints_the_report(tmp_path, monkeypatch, capsys):
    path = str(tmp_path / "e.db")
    with Database(path) as db:
        db.upsert("games", [game(7, "PHI", "DAL")])
        db.upsert("player_weeks", [week7("A", "PHI", 20.0), week7("B", "DAL", 12.0)])
        db.upsert("players", [player("A", "Alpha", "RB", "PHI"), player("B", "Bravo", "RB", "DAL")])
        db.upsert("recommendations", [rec("r1", {"A": "PHI", "B": "DAL"}, pick="A")])
    monkeypatch.setattr(sys, "argv", ["prog", "--db", path])
    evaluate.main()
    out = capsys.readouterr().out
    assert out.startswith("Scored 1 new recommendations.\nSeason 2026 track record")
    assert "AI 1/1 (100%)" in out


def test_margin_of_error():
    assert evaluate.Rate(hits=50, decided=100).margin == pytest.approx(0.098, abs=0.001)
    assert evaluate.Rate().margin is None
