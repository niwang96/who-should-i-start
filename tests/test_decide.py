from datetime import UTC, datetime

import pytest

from startsit.baseline import BaselineScore
from startsit.decide import Decision, Factor, build_prompt, decide, validate
from startsit.evidence import EvidenceItem, EvidencePacket, PlayerEvidence, Question
from startsit.llm import BudgetedLLM, FakeProvider, LLMError, QuotaExceeded


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


def answer(pick="P2", confidence="high", factors=None, disagreement=""):
    factors = factors or [
        {"text": "Smith is ruled out.", "evidence_ids": ["E2"]},
        {"text": "Nacua is the consensus WR1.", "evidence_ids": ["E3", "E4"]},
    ]
    return {"pick": pick, "confidence": confidence, "factors": factors, "baseline_disagreement": disagreement}


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
        ("not json", "not a JSON object"),
    ],
)
def test_validate_rejects(raw, problem):
    decision, problems = validate(raw, make_packet())
    assert decision is None
    assert any(problem in p for p in problems)


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
    assert decision.factors == [
        Factor("Baseline (100% FantasyPros consensus) favors Puka Nacua by 6.4 pts.", ["E3", "E4"])
    ]
    assert len(provider.prompts) == 2
    assert db.query("select count(*) as n from decision_cache")[0]["n"] == 0  # failed answers aren't cached


@pytest.mark.parametrize(
    "error,note",
    [
        (QuotaExceeded("rate limit"), "Daily AI limit reached; showing the baseline pick."),
        (LLMError("boom"), "AI unavailable; showing the baseline pick."),
    ],
)
def test_decide_falls_back_when_the_llm_fails(db, error, note):
    decision = decide(db, make_packet(), budgeted(db, FakeProvider(responses=[error])))
    assert (decision.source, decision.note) == ("baseline", note)


def test_decide_respects_the_daily_budget_and_works_without_an_llm(db):
    spent = decide(db, make_packet(), budgeted(db, FakeProvider(responses=[answer()]), budget=0))
    assert spent.note == "Daily AI limit reached; showing the baseline pick."
    none = decide(db, make_packet(margin=1.0), None)
    assert (none.source, none.confidence) == ("baseline", "low")


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
