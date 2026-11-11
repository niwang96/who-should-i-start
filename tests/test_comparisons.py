import pytest

from startsit.baseline import BaselineScore
from startsit.comparisons import backwards, comparisons, head_to_head
from startsit.evidence import EvidenceItem, EvidencePacket, PlayerEvidence, Question


def player(pid, name, position, points, implied, line, allowed):
    edge = f"favored by {line:g}" if line > 0 else f"underdog by {-line:g}"
    items = [
        EvidenceItem(
            f"E{pid}1",
            pid,
            "matchup",
            f"BAL vs JAX: betting lines imply {implied} team points ({edge}, game total 40.5)",
        ),
        EvidenceItem(
            f"E{pid}2", pid, "defense", f"JAX defense vs RBs this season: {allowed} PPR pts allowed per game to RBs"
        ),
    ]
    return PlayerEvidence(pid, name, position, "BAL", BaselineScore(pid, points), items)


@pytest.fixture
def packet():
    """Henry: higher projection, lower implied points, an underdog, the tougher matchup."""
    henry = player("1", "Derrick Henry", "RB", 13.1, 15.2, -10, 16.8)
    tuten = player("2", "Bhayshul Tuten", "RB", 12.1, 25.2, 10, 19.4)
    return EvidencePacket(Question(2026, 9, "ppr"), [henry, tuten], "1", 1.0)


def test_head_to_head_gives_each_direction(packet):
    assert head_to_head(packet) == [
        "projection: Derrick Henry 13.1, Bhayshul Tuten 12.1 (Derrick Henry's is higher)",
        "implied team points: Derrick Henry 15.2, Bhayshul Tuten 25.2 (Bhayshul Tuten's is higher)",
        "points allowed per game by the opponent's defense to the position: Derrick Henry 16.8, Bhayshul Tuten 19.4 "
        "(Bhayshul Tuten has the easier matchup, Derrick Henry the tougher)",
    ]


def test_defense_is_compared_only_within_a_position(packet):
    packet.players[1].position = "WR"
    assert [c.metric for c in comparisons(packet)] == ["projection", "implied"]
    packet.players.append(packet.players[0])
    assert comparisons(packet) == [] and head_to_head(packet) == []  # only two-player questions


@pytest.mark.parametrize(
    "text",
    [
        "Henry has higher implied team points than Tuten.",
        "Henry has a lower projection than Tuten [E2].",
        "Derrick Henry faces an easier matchup against a defense than Bhayshul Tuten.",
        "Tuten is a 10-point underdog this week.",
        "The Ravens are favored, so Derrick Henry is favored by 10.",
    ],
)
def test_backwards_comparisons_are_caught(packet, text):
    [problem] = backwards(text, packet)
    assert text.split(".")[0] in problem


@pytest.mark.parametrize(
    "text",
    [
        "Tuten has higher implied team points than Henry.",
        "Henry has a higher projection than Tuten, and he faces the tougher defense.",  # no "than" for the defense
        "Henry faces a tougher matchup than Tuten.",
        "Henry is an underdog by 10 while Tuten is favored.",
        "Henry faces the favored Jaguars.",  # "favored" isn't said of Henry
        "Tuten has a lower projection than Henry [E2].",
        "Henry has more carries than Tuten.",  # no metric code can check
        "Henry has a higher projection and lower implied points than Tuten.",  # two metrics: not checked
    ],
)
def test_correct_or_uncheckable_sentences_pass(packet, text):
    assert backwards(text, packet) == []


def test_a_tie_is_never_backwards(packet):
    packet.players[1].score = BaselineScore("2", 13.1)
    assert backwards("Tuten has a higher projection than Henry.", packet) == []
    assert "(equal)" in head_to_head(packet)[0]


def test_validation_rejects_a_backwards_answer_and_the_prompt_gives_the_directions(packet):
    from startsit.decide import build_prompt, validate

    def answer(summary):
        return {
            "pick": "P1",
            "confidence": "medium",
            "summary": summary,
            "factors": [{"text": "Henry leads the experts.", "evidence_ids": ["E11"]}] * 2,
            "baseline_disagreement": "",
        }

    decision, problems = validate(answer("Henry has higher implied team points than Tuten [E11]."), packet)
    assert decision is None and any("states the comparison backwards" in p for p in problems)
    decision, problems = validate(answer("Tuten has higher implied team points than Henry [E21]."), packet)
    assert decision is not None and problems == []
    assert "Head to head (worked out by code):\n- projection: Derrick Henry 13.1" in build_prompt(packet)
