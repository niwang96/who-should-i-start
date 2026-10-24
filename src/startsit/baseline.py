"""Code-only baseline: the pick the LLM has to beat or explain disagreeing with.

Currently 100% FantasyPros consensus. Each player's consensus projection (PPR, from
the weekly rankings) is converted to the league's scoring format, then zeroed for
players who can't play (bye, ruled Out, cut or retired). Usage-based adjustments are
deliberately left out until the weekly evaluation shows whether they help.

Points, not ranks, are compared, because ECR ranks are per position (is RB20 better
than WR16?).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

Scoring = Literal["ppr", "half", "std"]
SCORINGS: tuple[Scoring, ...] = get_args(Scoring)
SCORING_LABELS: dict[Scoring, str] = {"ppr": "PPR", "half": "half-PPR", "std": "standard"}
POINTS_PER_RECEPTION: dict[Scoring, float] = {"ppr": 1.0, "half": 0.5, "std": 0.0}

# Shown to the LLM and users; update it whenever the formula changes.
DESCRIPTION = "100% FantasyPros consensus"

# Injury report statuses that mean the player won't play. Doubtful/Questionable are left
# to the consensus (experts already discount them) and to the LLM.
OUT_STATUSES = {"Out"}
# Roster statuses with no path to playing this week. Reserve (IR) and practice-squad
# players can be activated before kickoff, so those are left to the consensus too.
NO_TEAM_STATUSES = {"CUT": "cut", "RET": "retired"}


@dataclass(frozen=True)
class BaselineInput:
    player_id: str
    position: str
    proj_ppr: float | None  # None: not in the weekly rankings
    exp_receptions: float | None  # per game, from recent usage; None: no games yet
    on_bye: bool = False
    report_status: str | None = None
    roster_status: str | None = None


@dataclass(frozen=True)
class BaselineScore:
    player_id: str
    points: float
    note: str | None = None  # why the score is zero or approximate


def converts(position: str, scoring: Scoring) -> bool:
    """Whether a PPR projection needs converting for this player and format."""
    return scoring != "ppr" and position != "QB"


def convert_ppr(points_ppr: float, exp_receptions: float, scoring: Scoring) -> float:
    """PPR points to `scoring` by removing the per-catch points it doesn't award."""
    return max(0.0, points_ppr - (1.0 - POINTS_PER_RECEPTION[scoring]) * exp_receptions)


def score(inp: BaselineInput, scoring: Scoring) -> BaselineScore:
    if inp.roster_status in NO_TEAM_STATUSES:
        return BaselineScore(inp.player_id, 0.0, f"{NO_TEAM_STATUSES[inp.roster_status]}, not on an active roster")
    if inp.on_bye:
        return BaselineScore(inp.player_id, 0.0, "on bye")
    if inp.report_status in OUT_STATUSES:
        return BaselineScore(inp.player_id, 0.0, f"ruled {inp.report_status}")
    if inp.proj_ppr is None:
        return BaselineScore(inp.player_id, 0.0, "not in FantasyPros weekly rankings")
    if not converts(inp.position, scoring):
        return BaselineScore(inp.player_id, inp.proj_ppr)
    if inp.exp_receptions is None:
        return BaselineScore(inp.player_id, inp.proj_ppr, "PPR projection; no usage data to convert")
    return BaselineScore(inp.player_id, round(convert_ppr(inp.proj_ppr, inp.exp_receptions, scoring), 1))


def leaders(scores: list[BaselineScore]) -> list[BaselineScore]:
    """Everyone tied for the top score (usually one player)."""
    top = max(s.points for s in scores)
    return [s for s in scores if s.points == top]


def pick(scores: list[BaselineScore]) -> tuple[str, float]:
    """The highest-scoring player (first listed, if tied) and their margin over the next best."""
    ranked = sorted(scores, key=lambda s: s.points, reverse=True)
    margin = ranked[0].points - ranked[1].points if len(ranked) > 1 else 0.0
    return ranked[0].player_id, round(margin, 1)
