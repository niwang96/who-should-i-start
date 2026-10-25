"""Code-only baseline: the pick the LLM has to beat or explain disagreeing with.

Currently 100% FantasyPros consensus. Each player's consensus projection (PPR, from
the weekly rankings) is converted to the league's scoring format (with FantasyPros'
projected receptions, else recent usage), then zeroed for players who can't play: on bye,
ruled out (injury report or this week's news), or not on the active roster unless this
week's news says they're active. Usage-based adjustments are deliberately left out until
the weekly evaluation shows whether they help.

Points, not ranks, are compared, because ECR ranks are per position (is RB20 better
than WR16?).
"""

from __future__ import annotations

from dataclasses import dataclass

from startsit.scoring import Scoring, from_ppr

# Shown to the LLM and users; update it whenever the formula changes.
DESCRIPTION = "100% FantasyPros consensus"

# Injury report statuses that mean the player won't play. Doubtful/Questionable are left
# to the consensus (experts already discount them) and to the LLM.
OUT_STATUSES = {"Out"}
# Only players on the active roster can play; every other roster status scores zero
# (decided Oct 24 2026). nflverse can lag an activation or practice-squad elevation by a
# day or two, which the packet's roster-status item makes visible to the LLM.
PLAYABLE_STATUSES = {"ACT"}
ROSTER_STATUS_LABELS = {
    "RES": "on a reserve list (IR/PUP/NFI)",
    "DEV": "on the practice squad",
    "EXE": "on the exempt list",
    "INA": "inactive",
    "CUT": "released",
    "RET": "retired",
}
# Statuses where the player has no team at all.
NO_TEAM_STATUSES = {"CUT", "RET"}
# Game statuses (lowercase, injury report or news) that cap the decision's confidence.
CAUTION_STATUSES = {"questionable", "doubtful"}


def roster_status_label(status: str) -> str:
    return ROSTER_STATUS_LABELS.get(status, f"roster status {status}")


@dataclass(frozen=True)
class BaselineInput:
    player_id: str
    position: str
    proj_ppr: float | None  # None: not in the weekly rankings
    exp_receptions: float | None  # per game, from recent usage; None: no games yet
    on_bye: bool = False
    report_status: str | None = None  # official injury report: Out, Doubtful, Questionable
    roster_status: str | None = None
    proj_receptions: float | None = None  # FantasyPros projected catches, preferred for conversion
    news_status: str | None = None  # this week's newest news status: out | doubtful | questionable | active

    @property
    def game_status(self) -> str | None:
        """The freshest game status, lowercase: news (newer) first, else the injury report."""
        return self.news_status or (self.report_status.lower() if self.report_status else None)


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
    return max(0.0, from_ppr(points_ppr, exp_receptions, scoring))


def conversion_receptions(inp: BaselineInput) -> float | None:
    """Catches to convert PPR points with: FantasyPros' projection, else recent usage."""
    return inp.proj_receptions if inp.proj_receptions is not None else inp.exp_receptions


def score(inp: BaselineInput, scoring: Scoring) -> BaselineScore:
    # The weekly roster lags activations by a day or two; this week's news can show a player is back.
    if inp.roster_status is not None and inp.roster_status not in PLAYABLE_STATUSES and inp.news_status != "active":
        return BaselineScore(inp.player_id, 0.0, f"{roster_status_label(inp.roster_status)}, not on the active roster")
    if inp.on_bye:
        return BaselineScore(inp.player_id, 0.0, "on bye")
    if inp.news_status == "out":
        return BaselineScore(inp.player_id, 0.0, "ruled out (news)")
    if inp.report_status in OUT_STATUSES:
        return BaselineScore(inp.player_id, 0.0, f"ruled {inp.report_status}")
    if inp.proj_ppr is None:
        return BaselineScore(inp.player_id, 0.0, "not in FantasyPros weekly rankings")
    if not converts(inp.position, scoring):
        return BaselineScore(inp.player_id, inp.proj_ppr)
    receptions = conversion_receptions(inp)
    if receptions is None:
        return BaselineScore(inp.player_id, inp.proj_ppr, "PPR projection; no reception data to convert")
    return BaselineScore(inp.player_id, round(convert_ppr(inp.proj_ppr, receptions, scoring), 1))


def leaders(scores: list[BaselineScore]) -> list[BaselineScore]:
    """Everyone tied for the top score (usually one player)."""
    top = max(s.points for s in scores)
    return [s for s in scores if s.points == top]


def pick(scores: list[BaselineScore]) -> tuple[str, float]:
    """The highest-scoring player (first listed, if tied) and their margin over the next best."""
    ranked = sorted(scores, key=lambda s: s.points, reverse=True)
    margin = ranked[0].points - ranked[1].points if len(ranked) > 1 else 0.0
    return ranked[0].player_id, round(margin, 1)
