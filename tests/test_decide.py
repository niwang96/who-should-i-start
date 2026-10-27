from datetime import UTC, datetime, timedelta

import pytest

from startsit.baseline import BaselineScore
from startsit.decide import Decision, Factor, build_prompt, decide, validate
from startsit.evidence import EvidenceItem, EvidencePacket, PlayerEvidence, Question
from startsit.llm import BudgetedLLM, FakeProvider, LLMError, QuotaExceeded, RateLimited, VisitorLimit


def make_packet(margin=6.4, a_points=13.1, b_points=19.5):
    a = PlayerEvidence(
        "SMITH",
        "DeVonta Smith",
        "WR",
        "PHI",
        BaselineScore("SMITH", a_points),
        [
            EvidenceItem("E1", "SMITH", "consensus", "FantasyPros consensus WR16", "https://fp/smith"),
            EvidenceItem("E2", "SMITH", "news", "Ruled out Monday.", "https://espn/smith"),
        ],
    )
    b = PlayerEvidence(
        "NACUA",
        "Puka Nacua",
        "WR",
        "LA",
        BaselineScore("NACUA", b_points),
        [
            EvidenceItem("E3", "NACUA", "consensus", "FantasyPros consensus WR1", "https://fp/nacua"),
            EvidenceItem("E4", "NACUA", "projection", "Consensus projection 24.5 PPR pts", "https://fp/nacua"),
        ],
    )
    pick = "NACUA" if b_points >= a_points else "SMITH"
    q = Question(2026, 7, "half", datetime(2026, 10, 25, tzinfo=UTC))
    return EvidencePacket(q, [a, b], pick, margin)


SUMMARY = "Nacua is the consensus WR1 [E3] and projects 24.5 PPR points [E4], while Smith is ruled out [E2]."


def answer(pick="P2", confidence="high", factors=None, disagreement="", summary=SUMMARY):
    factors = factors or [
        {"text": "Smith is ruled out.", "evidence_ids": ["E2"]},
        {"text": "Nacua is the consensus WR1.", "evidence_ids": ["E3", "E4"]},
    ]
    return {
        "pick": pick,
        "confidence": confidence,
        "summary": summary,
        "factors": factors,
        "baseline_disagreement": disagreement,
    }


def budgeted(db, provider, budget=10):
    return BudgetedLLM(provider, db, "decide", budget)


def test_build_prompt_labels_players_and_the_baseline():
    prompt = build_prompt(make_packet())
    assert prompt.startswith("Players:\nP1 = DeVonta Smith (WR, PHI)\nP2 = Puka Nacua (WR, LA)\nBaseline pick: P2\n")
    assert "E2 [news] Ruled out Monday." in prompt


def test_validate_accepts_a_grounded_answer():
    decision, problems = validate(answer(), make_packet())
    assert problems == []
    assert decision == Decision(
        pick="NACUA",
        confidence="high",
        factors=[Factor("Smith is ruled out.", ["E2"]), Factor("Nacua is the consensus WR1.", ["E3", "E4"])],
        baseline_pick="NACUA",
        summary=SUMMARY,
    )
    assert decision.agrees_with_baseline


@pytest.mark.parametrize(
    "raw,problem",
    [
        (answer(pick="P9"), "pick must be one of P1, P2"),
        (answer(confidence="certain"), "confidence must be high, medium or low"),
        (answer(factors=[{"text": "Made up.", "evidence_ids": ["E99"]}]), "ids not in the packet: E99"),
        (answer(factors=[{"text": "No citation.", "evidence_ids": []}]), "at least one evidence id"),
        (answer(factors=[{"text": f"f{i}", "evidence_ids": ["E1"]} for i in range(4)]), "give 2 or 3 factors"),
        (answer(factors=[{"text": "Only one.", "evidence_ids": ["E1"]}]), "give 2 or 3 factors"),
        (answer(factors=[{"text": "A string, not a list.", "evidence_ids": "E1"}] * 2), "at least one evidence id"),
        (answer(pick="P1"), "baseline_disagreement must explain why"),
        (answer(summary=" "), "summary is required"),
        (answer(summary="x" * 901), "at most 900 characters"),
        (answer(summary="Start Puka Smith [E3]."), "these player names are mixed up: Puka Smith"),
        (answer(summary="Smith is out [E9]."), "text cites ids not in the packet: E9"),
        (answer(summary="Nacua projects 31.2 points."), "these numbers aren't in the evidence: 31.2"),
        (
            answer(factors=[{"text": "Nacua had 140 yards.", "evidence_ids": ["E3"]}] * 2),
            "these numbers aren't in the evidence: 140",
        ),
        ("not json", "not a JSON object"),
    ],
)
def test_validate_rejects(raw, problem):
    decision, problems = validate(raw, make_packet())
    assert decision is None
    assert any(problem in p for p in problems)


def test_swapped_names_allow_players_who_share_a_last_name():
    from startsit.decide import swapped_names

    names = ["Amon-Ra St. Brown", "Equanimeous St. Brown"]
    assert swapped_names("Start Amon-Ra St. Brown over Equanimeous St. Brown.", names) == []
    assert swapped_names("Tyler Kelce edges Travis Higbee.", ["Travis Kelce", "Tyler Higbee"]) == [
        "Travis Higbee",
        "Tyler Kelce",
    ]


def test_citations_that_are_not_ids_are_dropped_and_an_uncited_summary_is_accepted():
    factors = [
        {"text": "Smith is ruled out.", "evidence_ids": ["E2", "Week 7 2026, half-PPR scoring. Baseline..."]},
        {"text": "Nacua is the consensus WR1.", "evidence_ids": ["[E3]", "E4"]},
    ]
    decision, problems = validate(answer(factors=factors, summary="Nacua is the safer start."), make_packet())
    assert problems == []
    assert [f.evidence_ids for f in decision.factors] == [["E2"], ["E3", "E4"]]
    _, problems = validate(answer(factors=[{"text": "Header only.", "evidence_ids": ["Baseline"]}] * 2), make_packet())
    assert "every factor needs text and at least one evidence id" in problems


def test_numbers_may_come_from_the_packet_rounded_or_as_a_difference():
    # 24.5 is in the packet; 24 is it rounded; 6.4 is 19.5 - 13.1 (both baseline scores); 3 is a small count.
    summary = "Nacua projects 24.5 (about 24) points [E4], 6.4 more than Smith, over his last 3 games."
    decision, problems = validate(answer(summary=summary), make_packet())
    assert problems == [] and decision.summary == summary


def test_a_disagreement_is_kept_only_when_the_pick_differs():
    decision, _ = validate(answer(pick="P1", disagreement="News says Nacua is out. [E2]"), make_packet())
    assert decision.pick == "SMITH" and decision.baseline_disagreement.startswith("News says")
    same, _ = validate(answer(disagreement="Agree, but noting E1."), make_packet())
    assert same.baseline_disagreement == ""


def test_a_tie_needs_no_disagreement():
    decision, problems = validate(answer(pick="P1"), make_packet(margin=0, a_points=10, b_points=10))
    assert problems == [] and decision.pick == "SMITH"


def test_decide_caches_and_attaches_citations(db):
    provider = FakeProvider(responses=[answer()])
    first = decide(db, make_packet(), budgeted(db, provider))
    assert (first.source, first.model) == ("llm", "fake-model")
    assert first.cited_urls == {"E2": "https://espn/smith", "E3": "https://fp/nacua", "E4": "https://fp/nacua"}
    again = decide(db, make_packet(), budgeted(db, provider))
    assert again.source == "cache" and again.factors == first.factors
    assert len(provider.prompts) == 1  # the cache made no second request
    # Different evidence means a fresh answer.
    changed = make_packet(margin=7.0)
    provider.responses.append(answer())
    assert decide(db, changed, budgeted(db, provider)).source == "llm"


def test_decide_retries_once_with_the_problems(db):
    provider = FakeProvider(responses=[answer(factors=[{"text": "x", "evidence_ids": ["E42"]}]), answer()])
    decision = decide(db, make_packet(), budgeted(db, provider))
    assert decision.source == "llm"
    assert "previous answer was rejected" in provider.prompts[1] and "E42" in provider.prompts[1]


def test_decide_falls_back_to_the_baseline_after_two_bad_answers(db):
    provider = FakeProvider(responses=[answer(pick="P1"), answer(pick="P1")])
    decision = decide(db, make_packet(), budgeted(db, provider))
    assert (decision.source, decision.pick, decision.confidence) == ("baseline", "NACUA", "high")
    assert decision.note == "The AI's answer failed validation; showing the baseline pick."
    assert decision.fallback == "invalid"
    assert decision.factors == [
        Factor("Baseline (100% FantasyPros consensus) favors Puka Nacua by 6.4 pts.", ["E3", "E4"])
    ]
    assert len(provider.prompts) == 2
    assert db.query("select count(*) as n from decision_cache")[0]["n"] == 0  # failed answers aren't cached


@pytest.mark.parametrize(
    "error,note,reason",
    [
        (QuotaExceeded("daily limit"), "Daily AI limit reached; showing the baseline pick.", "daily_limit"),
        (
            RateLimited("per minute"),
            "The AI is busy right now; showing the baseline pick. Ask again in a minute.",
            "busy",
        ),
        (LLMError("boom"), "AI unavailable; showing the baseline pick.", "unavailable"),
    ],
)
def test_decide_falls_back_when_the_llm_fails(db, error, note, reason):
    decision = decide(db, make_packet(), budgeted(db, FakeProvider(responses=[error])))
    assert (decision.source, decision.note, decision.fallback) == ("baseline", note, reason)


def test_decide_respects_the_daily_budget_and_works_without_an_llm(db):
    spent = decide(db, make_packet(), budgeted(db, FakeProvider(responses=[answer()]), budget=0))
    assert spent.note == "Daily AI limit reached; showing the baseline pick."
    none = decide(db, make_packet(margin=1.0), None)
    assert (none.source, none.confidence, none.fallback) == ("baseline", "low", "unavailable")


def test_decide_stops_a_visitor_at_their_limit_but_still_serves_cached_answers(db):
    def visitor_llm(provider):
        limits = [VisitorLimit(1, timedelta(minutes=10), "10 minutes")]
        return BudgetedLLM(provider, db, "decide", 10, visitor="v1", visitor_limits=limits)

    first = decide(db, make_packet(), visitor_llm(FakeProvider(responses=[answer()])))
    assert first.source == "llm"
    assert decide(db, make_packet(), visitor_llm(FakeProvider())).source == "cache"  # no call, no limit
    other = decide(db, make_packet(margin=6.5), visitor_llm(FakeProvider(responses=[answer()])))
    assert (other.source, other.note) == (
        "baseline",
        "You've reached the limit of 1 new AI answers per 10 minutes; showing the baseline pick.",
    )


def test_decision_round_trips_through_json():
    decision, _ = validate(answer(), make_packet())
    assert Decision.from_json(decision.to_json()) == decision


def test_confidence_is_capped_for_a_questionable_pick(db):
    packet = make_packet()
    packet.players[1].game_status = "questionable"
    decision = decide(db, packet, budgeted(db, FakeProvider(responses=[answer(confidence="high")])))
    assert decision.confidence == "medium"
    assert decision.note == "Confidence capped at medium: Puka Nacua is listed questionable."
    # The baseline fallback is capped too, and the note keeps both reasons.
    fallback = decide(db, make_packet(margin=7.5), None)
    assert fallback.confidence == "high"
    packet2 = make_packet(margin=7.5)
    packet2.players[1].game_status = "doubtful"
    capped = decide(db, packet2, None)
    assert capped.confidence == "medium" and capped.note.startswith("AI unavailable") and "doubtful" in capped.note


def test_low_or_medium_confidence_is_left_alone(db):
    packet = make_packet()
    packet.players[1].game_status = "questionable"
    decision = decide(db, packet, budgeted(db, FakeProvider(responses=[answer(confidence="low")])))
    assert (decision.confidence, decision.note) == ("low", None)
