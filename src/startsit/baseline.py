"""Code-only baseline: the pick the LLM has to beat or explain disagreeing with.

Currently 100% FantasyPros consensus. Each player's consensus projection (PPR, from
the weekly rankings) is converted to the league's scoring format, then zeroed for
players who can't play (bye, ruled Out). Usage-based adjustments are deliberately left
out until the weekly evaluation shows whether they help.

Points, not ranks, are compared, because ECR ranks are per position (is RB20 better
than WR16?).
"""

from __future__ import annotations

from dataclasses import dataclass

SCORINGS = ("ppr", "half", "std")
SCORING_LABELS = {"ppr": "PPR", "half": "half-PPR", "std": "standard"}
POINTS_PER_RECEPTION = {"ppr": 1.0, "half": 0.5, "std": 0.0}

# Report statuses that mean the player won't play. Doubtful/Questionable are left to
# the consensus (experts already discount them) and to the LLM.
OUT_STATUSES = {"Out"}


@dataclass
class BaselineInput:
    player_id: str
    position: str
    proj_ppr: float | None  # None: not in the weekly rankings
    exp_receptions: float | None  # per game, from recent usage; None: no games yet
    on_bye: bool = False
    report_status: str | None = None


@dataclass
class BaselineScore:
    player_id: str
    points: float
    note: str | None = None  # why the score is zero or approximate


def convert_ppr(points_ppr: float, exp_receptions: float, scoring: str) -> float:
    """PPR points to `scoring` by removing the per-catch points it doesn't award."""
    return max(0.0, points_ppr - (1.0 - POINTS_PER_RECEPTION[scoring]) * exp_receptions)


def score(inp: BaselineInput, scoring: str) -> BaselineScore:
    if inp.on_bye:
        return BaselineScore(inp.player_id, 0.0, "on bye")
    if inp.report_status in OUT_STATUSES:
        return BaselineScore(inp.player_id, 0.0, f"ruled {inp.report_status}")
    if inp.proj_ppr is None:
        return BaselineScore(inp.player_id, 0.0, "not in FantasyPros weekly rankings")
    if scoring == "ppr" or inp.position == "QB":
        return BaselineScore(inp.player_id, inp.proj_ppr)
    if inp.exp_receptions is None:
        return BaselineScore(inp.player_id, inp.proj_ppr, "PPR projection; no usage data to convert")
    return BaselineScore(inp.player_id, round(convert_ppr(inp.proj_ppr, inp.exp_receptions, scoring), 1))


def pick(scores: list[BaselineScore]) -> tuple[str, float]:
    """The highest-scoring player and their margin over the next best."""
    ranked = sorted(scores, key=lambda s: s.points, reverse=True)
    margin = ranked[0].points - ranked[1].points if len(ranked) > 1 else 0.0
    return ranked[0].player_id, round(margin, 1)
