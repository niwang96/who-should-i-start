"""Head-to-head comparisons worked out by code, and a check that an answer doesn't state one backwards.

The model once wrote that a 4.5-point underdog had the higher implied team points. The numbers were
right, so the number check passed, but the direction was wrong. For a two-player question, code
works out each comparison an answer is likely to make from the evidence it already wrote:
projections, implied team points (and who's favored), and how many points each opponent's defense
allows to the player's position (same position only). The prompt gives the model these directions
(`head_to_head`), and validation rejects a sentence that states one backwards (`backwards`), which
gets the model the usual retry.

The check is deliberately narrow, to avoid rejecting good answers: it reads only sentences shaped like
"A has higher implied points than B" (a subject player, a comparative word, "than", the other player)
about exactly one of the metrics, and "A is favored / an underdog" about a player's own team.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from startsit.evidence import EvidencePacket, PlayerEvidence

_IMPLIED = re.compile(r"imply ([\d.]+) team points")
_SPREAD = re.compile(r"\((favored|underdog) by ([\d.]+)")
_ALLOWED = re.compile(r"([\d.]+) \S+ pts allowed per game")

UP = {
    "higher",
    "more",
    "bigger",
    "larger",
    "greater",
    "better",
    "easier",
    "softer",
    "weaker",
    "friendlier",
    "favorable",
}
DOWN = {"lower", "fewer", "less", "smaller", "worse", "tougher", "harder", "stronger", "unfavorable", "difficult"}
# For the defense metric, "up" means the opponent allows more points: an easier matchup.
_COMPARATIVE = re.compile(r"\b(" + "|".join(sorted(UP | DOWN)) + r")\b", re.IGNORECASE)
# Words that tie a sentence to each metric; a sentence touching more than one isn't checked.
KEYWORDS = {
    "projection": re.compile(r"\bproject|\bbaseline\b|\bconsensus\b", re.IGNORECASE),
    "implied": re.compile(r"\bimplied\b|\bteam total\b|\bteam points\b|\bbetting\b", re.IGNORECASE),
    "defense": re.compile(r"\bdefen[cs]e|\bmatchup\b|\bpoints allowed\b|\ballows?\b", re.IGNORECASE),
}
LABELS = {
    "projection": "projection",
    "implied": "implied team points",
    "defense": "points allowed per game by the opponent's defense to the position",
}
_ROLE = re.compile(
    r"\b(?:is|are|as)\s+(?:an?\s+|the\s+)?(?:heavy\s+|slight\s+|big\s+|[\d.]+-point\s+)?(favored|favorite|underdog)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Comparison:
    metric: str  # projection | implied | defense
    values: dict[str, float]  # player name -> value

    @property
    def leader(self) -> str | None:
        """The player with the higher value, or None for a tie."""
        (a, va), (b, vb) = self.values.items()
        return None if va == vb else a if va > vb else b

    def describe(self) -> str:
        values = ", ".join(f"{n} {v:g}" for n, v in self.values.items())
        if self.leader is None:
            return f"{LABELS[self.metric]}: {values} (equal)"
        if self.metric == "defense":
            other = next(n for n in self.values if n != self.leader)
            return f"{LABELS[self.metric]}: {values} ({self.leader} has the easier matchup, {other} the tougher)"
        return f"{LABELS[self.metric]}: {values} ({self.leader}'s is higher)"


def _first(pattern: re.Pattern[str], p: PlayerEvidence, kind: str) -> re.Match[str] | None:
    for item in p.items_of(kind):  # type: ignore[arg-type]
        if m := pattern.search(item.text):
            return m
    return None


def spread(p: PlayerEvidence) -> float | None:
    """The player's team spread: positive when favored, negative when an underdog."""
    m = _first(_SPREAD, p, "matchup")
    if m is None:
        return None
    return float(m[2]) if m[1] == "favored" else -float(m[2])


def comparisons(packet: EvidencePacket) -> list[Comparison]:
    """The comparisons code can make for a two-player question, in the order shown to the model."""
    if len(packet.players) != 2:
        return []
    a, b = packet.players
    found = [Comparison("projection", {a.name: a.score.points, b.name: b.score.points})]
    implied = [_first(_IMPLIED, p, "matchup") for p in (a, b)]
    if implied[0] and implied[1]:
        found.append(Comparison("implied", {a.name: float(implied[0][1]), b.name: float(implied[1][1])}))
    allowed = [_first(_ALLOWED, p, "defense") for p in (a, b)]
    if allowed[0] and allowed[1] and a.position == b.position:
        found.append(Comparison("defense", {a.name: float(allowed[0][1]), b.name: float(allowed[1][1])}))
    return found


def head_to_head(packet: EvidencePacket) -> list[str]:
    """Lines for the prompt: each comparison's values and direction."""
    return [c.describe() for c in comparisons(packet)]


def _aliases(packet: EvidencePacket) -> dict[str, re.Pattern[str]]:
    """Each player's full name, or last name when only he has it, as a pattern."""
    last = [p.name.split()[-1] for p in packet.players]
    out = {}
    for p, surname in zip(packet.players, last, strict=True):
        names = [re.escape(p.name)]
        if last.count(surname) == 1 and len(surname) > 2:
            names.append(re.escape(surname))
        out[p.name] = re.compile(r"\b(?:" + "|".join(names) + r")\b", re.IGNORECASE)
    return out


def _mentions(sentence: str, aliases: dict[str, re.Pattern[str]]) -> list[tuple[int, int, str]]:
    """(start, end, player name) of every player mentioned, in order."""
    found = [(m.start(), m.end(), name) for name, pat in aliases.items() for m in pat.finditer(sentence)]
    return sorted(found)


def backwards(text: str, packet: EvidencePacket) -> list[str]:
    """Problems with comparisons `text` states the wrong way round (empty if none)."""
    by_metric = {c.metric: c for c in comparisons(packet)}
    if not by_metric:
        return []
    aliases = _aliases(packet)
    spreads = {p.name: spread(p) for p in packet.players}
    problems = []
    for sentence in re.split(r"(?<=[.!?;])\s+", text):
        mentions = _mentions(sentence, aliases)
        problems += _role_problems(sentence, mentions, spreads)
        problem = _than_problem(sentence, mentions, by_metric)
        if problem:
            problems.append(problem)
    return problems


def _than_problem(sentence: str, mentions: list[tuple[int, int, str]], by_metric: dict[str, Comparison]) -> str | None:
    than = re.search(r"\bthan\b", sentence, re.IGNORECASE)
    metrics = [m for m, pat in KEYWORDS.items() if pat.search(sentence)]
    if not than or len(metrics) != 1 or metrics[0] not in by_metric or not mentions:
        return None
    start, end, subject = mentions[0]
    after = [name for s, _, name in mentions if s > than.start() and name != subject]
    if start > than.start() or not after:
        return None
    words = _COMPARATIVE.findall(sentence[end : than.start()])
    comparison = by_metric[metrics[0]]
    if not words or comparison.leader is None:
        return None
    claims_up = words[-1].lower() in UP
    if claims_up == (comparison.leader == subject):
        return None
    return f'"{sentence.strip()}" states the comparison backwards: {comparison.describe()}'


def _role_problems(sentence: str, mentions: list[tuple[int, int, str]], spreads: dict[str, float | None]) -> list[str]:
    """A player called favored/underdog when his team is the other."""
    problems = []
    for i, (_, end, name) in enumerate(mentions):
        stop = mentions[i + 1][0] if i + 1 < len(mentions) else len(sentence)
        segment, line = sentence[end:stop], spreads.get(name)
        m = _ROLE.search(segment)
        if m is None or line is None or line == 0:
            continue
        says_favored = m[1].lower() != "underdog"
        if says_favored != (line > 0):
            actual = f"favored by {line:g}" if line > 0 else f"an underdog by {-line:g}"
            problems.append(f'"{sentence.strip()}" gets the betting line backwards: {name}\'s team is {actual}')
    return problems
