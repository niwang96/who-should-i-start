import pytest

from startsit.baseline import BaselineInput, BaselineScore, convert_ppr, leaders, pick, score


def inp(**kw):
    return BaselineInput(**(dict(player_id="A", position="WR", proj_ppr=15.0, exp_receptions=6.0) | kw))


@pytest.mark.parametrize("scoring,expected", [("ppr", 15.0), ("half", 12.0), ("std", 9.0)])
def test_converts_ppr_projection_by_expected_catches(scoring, expected):
    assert score(inp(), scoring).points == expected


def test_conversion_never_goes_negative():
    assert convert_ppr(2.0, 6.0, "std") == 0.0


def test_qbs_and_missing_usage_keep_the_ppr_projection():
    assert score(inp(position="QB", exp_receptions=None), "std").points == 15.0
    s = score(inp(exp_receptions=None), "half")
    assert s.points == 15.0 and "no reception data" in s.note


@pytest.mark.parametrize(
    "kw,note",
    [
        (dict(on_bye=True), "on bye"),
        (dict(report_status="Out"), "ruled Out"),
        (dict(proj_ppr=None), "not in FantasyPros weekly rankings"),
        (dict(roster_status="CUT"), "released, not on the active roster"),
        (dict(roster_status="RET"), "retired, not on the active roster"),
        (dict(roster_status="RES"), "on a reserve list (IR/PUP/NFI), not on the active roster"),
        (dict(roster_status="DEV"), "on the practice squad, not on the active roster"),
        (dict(roster_status="EXE"), "on the exempt list, not on the active roster"),
        (dict(roster_status="TRT"), "roster status TRT, not on the active roster"),
    ],
)
def test_players_who_cant_score_get_zero(kw, note):
    s = score(inp(**kw), "ppr")
    assert (s.points, s.note) == (0.0, note)


def test_active_questionable_and_unknown_roster_are_left_to_the_consensus():
    assert score(inp(report_status="Questionable"), "ppr").points == 15.0
    assert score(inp(roster_status="ACT"), "ppr").points == 15.0
    assert score(inp(roster_status=None), "ppr").points == 15.0


def test_pick_returns_winner_and_margin():
    assert pick([BaselineScore("A", 11.8), BaselineScore("B", 19.5), BaselineScore("C", 3.0)]) == ("B", 7.7)
    assert pick([BaselineScore("A", 10.0), BaselineScore("B", 10.0)]) == ("A", 0.0)


def test_leaders_returns_everyone_tied_at_the_top():
    scores = [BaselineScore("A", 9.0), BaselineScore("B", 12.0), BaselineScore("C", 12.0)]
    assert [s.player_id for s in leaders(scores)] == ["B", "C"]


def test_fantasypros_receptions_are_preferred_for_conversion():
    assert score(inp(proj_receptions=4.0), "std").points == 11.0  # 15 - 4, not 15 - 6 from usage
    assert score(inp(proj_receptions=None), "std").points == 9.0  # falls back to usage


@pytest.mark.parametrize(
    "kw,points,note",
    [
        (dict(news_status="out"), 0.0, "ruled out (news)"),
        (dict(roster_status="EXE", news_status="active"), 15.0, None),  # activated after the roster snapshot
        (
            dict(roster_status="RES", news_status="questionable"),
            0.0,
            "on a reserve list (IR/PUP/NFI), not on the active roster",
        ),
        (dict(news_status="questionable"), 15.0, None),
    ],
)
def test_this_weeks_news_status(kw, points, note):
    s = score(inp(**kw), "ppr")
    assert (s.points, s.note) == (points, note)


def test_game_status_prefers_news_over_the_injury_report():
    assert inp(report_status="Questionable").game_status == "questionable"
    assert inp(report_status="Questionable", news_status="out").game_status == "out"
    assert inp().game_status is None
