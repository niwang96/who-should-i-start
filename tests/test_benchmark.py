from datetime import UTC, datetime

import pytest
from test_evidence import game, player, ranking

from startsit import benchmark
from startsit.evaluate import BACKFILL, BENCHMARK
from startsit.llm import BudgetedLLM, FakeProvider, RateLimited

# Valid for any two-player packet here: cites the first player's consensus items, states no numbers.
ANSWER = {
    "pick": "P1",
    "confidence": "low",
    "factors": [
        {"text": "The experts rank him a little higher.", "evidence_ids": ["E1"]},
        {"text": "His projection is a little higher.", "evidence_ids": ["E2"]},
    ],
    "summary": "A close call that leans on the consensus [E1].",
    "baseline_disagreement": "The consensus edge outweighs the rest [E1].",
}
SATURDAY = datetime(2026, 10, 31, 18, 0, tzinfo=UTC)
SUNDAY_MORNING = datetime(2026, 11, 1, 15, 0, tzinfo=UTC)  # 10am Eastern (EST from Nov 1)
SUNDAY_KICKOFF = datetime(2026, 11, 1, 18, 0, tzinfo=UTC)
MONDAY = datetime(2026, 11, 2, 15, 0, tzinfo=UTC)
TEAMS = ["PHI", "DAL", "NYJ", "MIA", "GB", "NYG"]


@pytest.fixture
def week8(db, monkeypatch):
    """Week 8: GB-NYG on Thursday, the rest on Sunday at 1pm Eastern; six WRs and four RBs ranked."""
    db.upsert(
        "games",
        [
            game(8, "PHI", "DAL", 3.0, 45.0, "2026-11-01", "13:00"),
            game(8, "NYJ", "MIA", -1.0, 40.0, "2026-11-01", "13:00"),
            game(8, "GB", "NYG", 6.5, 42.0, "2026-10-29", "20:15"),
        ],
    )
    wrs = [(f"W{i}", TEAMS[i]) for i in range(6)]  # W4 and W5 play Thursday
    rbs = [(f"R{i}", TEAMS[i]) for i in range(4)]
    db.upsert("players", [player(pid, pid, "WR", t) for pid, t in wrs] + [player(pid, pid, "RB", t) for pid, t in rbs])
    db.upsert(
        "rankings",
        [ranking(pid, 15.0 - i, week=8, ecr=float(i + 1), position="WR") for i, (pid, _) in enumerate(wrs)]
        + [ranking(pid, 12.0 - i, week=8, ecr=float(i + 1), position="RB") for i, (pid, _) in enumerate(rbs)]
        + [ranking("NOID", 9.0, week=8, position="WR") | {"player_id": None}],
    )
    monkeypatch.setattr(benchmark, "PAIRS", {"QB": 1, "RB": 1, "WR": 3, "TE": 1})
    monkeypatch.setattr(benchmark, "WINDOW_START", {"QB": 1, "RB": 2, "WR": 1, "TE": 1})
    return db


def test_benchmark_is_due_three_hours_before_the_first_sunday_kickoff(week8):
    assert benchmark.benchmark_time(week8.week_games(2026, 8)) == SUNDAY_MORNING
    assert not benchmark.due(week8, 2026, 8, SATURDAY)
    assert benchmark.due(week8, 2026, 8, SUNDAY_MORNING)
    assert not benchmark.due(week8, 2026, 8, SUNDAY_KICKOFF)  # window over: left to the backfill
    assert not benchmark.due(week8, 2026, 9, SUNDAY_MORNING)  # no schedule for the week


def test_benchmark_time_ignores_a_sunday_morning_international_game(week8):
    games = [*week8.week_games(2026, 8), game(8, "JAX", "LV", gameday="2026-11-01", gametime="09:30")]
    assert benchmark.benchmark_time(games) == SUNDAY_MORNING


def test_benchmark_time_without_sunday_games():
    games = [game(18, "PHI", "DAL", gameday="2027-01-09", gametime="16:30")]  # a Saturday
    assert benchmark.benchmark_time(games) == datetime(2027, 1, 9, 18, 30, tzinfo=UTC)
    assert benchmark.benchmark_time([game(18, "PHI", "DAL")]) is None


def test_matchups_pair_rank_neighbors_whose_games_have_not_started(week8):
    assert benchmark.matchups(week8, 2026, 8, SUNDAY_MORNING) == [
        ("R1", "R2", "ppr"),  # RBs from rank 2; R3 has no partner
        ("W1", "W0", "half"),  # the order alternates
        ("W2", "W3", "std"),  # W4 and W5 played Thursday
    ]


def test_run_asks_the_ai_and_logs_benchmark_recommendations(week8):
    provider = FakeProvider(respond=lambda prompt: ANSWER)
    llm = BudgetedLLM(provider, week8, "benchmark", 10, clock=lambda: SUNDAY_MORNING)
    sleeps: list[float] = []
    assert benchmark.run(week8, llm, 2026, 8, clock=lambda: SUNDAY_MORNING, sleep=sleeps.append) == 3
    rows = week8.query("SELECT visitor, source, pick, scoring FROM recommendations ORDER BY scoring")
    assert [(r["visitor"], r["source"], r["scoring"]) for r in rows] == [
        (BENCHMARK, "llm", "half"),
        (BENCHMARK, "llm", "ppr"),
        (BENCHMARK, "llm", "std"),
    ]
    assert sleeps == [benchmark.CALL_SPACING] * 2  # before each AI call after the first
    assert not benchmark.due(week8, 2026, 8, SUNDAY_MORNING)  # done for the week
    assert week8.llm_calls_since(SATURDAY, "benchmark") == 3


def test_run_stops_when_the_budget_is_spent(week8):
    llm = BudgetedLLM(FakeProvider(respond=lambda prompt: ANSWER), week8, "benchmark", 1, clock=lambda: SUNDAY_MORNING)
    assert benchmark.run(week8, llm, 2026, 8, clock=lambda: SUNDAY_MORNING, sleep=lambda s: None) == 1


def test_backfill_picks_the_newest_week_whose_window_passed_without_a_set(week8):
    week8.upsert("games", [game(7, "PHI", "DAL", gameday="2026-10-25", gametime="13:00")])
    assert benchmark.backfill_week(week8, 2026, 8, SUNDAY_MORNING) == (7, datetime(2026, 10, 25, 14, 0, tzinfo=UTC))
    assert benchmark.backfill_week(week8, 2026, 8, MONDAY) == (8, SUNDAY_MORNING)
    week8.upsert("games", [game(1, "PHI", "DAL", gameday="2026-09-13", gametime="13:00")])
    week8.execute("DELETE FROM games WHERE week IN (7, 8)")
    assert benchmark.backfill_week(week8, 2026, 8, MONDAY) is None  # week 1 is never backfilled


def test_a_backfilled_set_is_asked_and_logged_as_of_the_past(week8):
    llm = BudgetedLLM(FakeProvider(respond=lambda prompt: ANSWER), week8, "benchmark", 10, clock=lambda: MONDAY)
    n = benchmark.run(
        week8, llm, 2026, 8, as_of=SUNDAY_MORNING, tag=BACKFILL, clock=lambda: MONDAY, sleep=lambda s: None
    )
    assert n == 3
    rows = week8.query("SELECT DISTINCT visitor, created_at FROM recommendations")
    assert rows == [{"visitor": BACKFILL, "created_at": "2026-11-01T15:00:00+00:00"}]
    assert benchmark.done(week8, 2026, 8) and benchmark.backfill_week(week8, 2026, 8, MONDAY) is None


def test_a_set_stops_when_the_ai_does_not_answer_and_resumes_later(week8):
    calls = {"n": 0}

    def respond(prompt):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RateLimited("busy")
        return ANSWER

    llm = BudgetedLLM(FakeProvider(respond=respond), week8, "benchmark", 10, clock=lambda: SUNDAY_MORNING)
    run = dict(clock=lambda: SUNDAY_MORNING, sleep=lambda s: None)
    assert benchmark.run(week8, llm, 2026, 8, **run) == 1  # the busy answer isn't logged as the AI's
    assert not benchmark.done(week8, 2026, 8) and benchmark.due(week8, 2026, 8, SUNDAY_MORNING)
    assert benchmark.calls_needed(week8, 2026, 8) == sum(benchmark.PAIRS.values()) - 1
    assert benchmark.run(week8, llm, 2026, 8, **run) == 2  # resumes with the matchups not yet asked
    assert benchmark.done(week8, 2026, 8)
    rows = week8.query("SELECT source FROM recommendations")
    assert [r["source"] for r in rows] == ["llm"] * 3  # no duplicates, no baseline rows


def test_an_invalid_answer_is_logged_and_its_retry_counts_toward_the_pause(week8):
    bad = dict(ANSWER, pick="P9")
    replies = [bad, bad, ANSWER, ANSWER]
    llm = BudgetedLLM(FakeProvider(responses=replies), week8, "benchmark", 10, clock=lambda: SUNDAY_MORNING)
    sleeps: list[float] = []
    assert benchmark.run(week8, llm, 2026, 8, clock=lambda: SUNDAY_MORNING, sleep=sleeps.append) == 3
    assert sleeps == [2 * benchmark.CALL_SPACING, benchmark.CALL_SPACING]  # two calls, then one
    sources = [r["source"] for r in week8.query("SELECT source FROM recommendations ORDER BY created_at")]
    assert sorted(sources) == ["baseline", "llm", "llm"]  # the AI answered badly: that counts against it


def test_the_daily_budgets_fit_geminis_free_tier_and_a_whole_benchmark_set():
    from startsit.llm import DAILY_BUDGETS, GEMINI_DAILY_REQUESTS

    assert sum(DAILY_BUDGETS.values()) <= GEMINI_DAILY_REQUESTS - 5  # headroom for count-then-call overshoot
    assert DAILY_BUDGETS["benchmark"] >= sum(benchmark.PAIRS.values())
