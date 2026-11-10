"""Step 6a: score every recommendation against what happened, and report hit rates.

A pick hits when the chosen player outscored the other in the question's scoring format.
Each recommendation is scored for four picks, so the report can show whether each layer
adds value:
- the pick shown (the AI's, or the baseline's when the AI wasn't used),
- the code baseline (consensus projection, adjusted for availability and scoring),
- the raw FantasyPros consensus as of the day asked (better rank within a position, else the
  higher PPR projection; no availability or scoring adjustments),
- the Reddit crowd's lean, when it had one (`EvidencePacket.crowd_pick`, logged when asked).

Actual points come from the nflverse weekly stats the daily ingest loads. A player with no
stats scores 0 once his team's game is in the data (or his team was on bye); until then the
recommendation stays unscored, so a partly loaded week is never scored early.

Run: python -m startsit.evaluate [--season 2026] [--db path-or-url]   (scores, then prints the report)
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

from startsit.db import Database, EvaluationRow, RecommendationRow, utc_iso
from startsit.decide import CONFIDENCES, OVERRIDE_KINDS
from startsit.scoring import Scoring, column

# recommendations.visitor of the weekly benchmark set and its backfill of past weeks (startsit.benchmark)
BENCHMARK = "benchmark"
BACKFILL = "benchmark-backfill"
GROUPS = {BENCHMARK: "benchmark", BACKFILL: "backfill"}  # anything else is a real user
AI_SOURCES = {"llm", "cache"}
REASON_CHARS = 160  # of an override's reason, in the text report
# Why the AI went against the baseline: a definite fact it cited (the only reason the override rule
# accepts when the baseline favors someone), a tie it broke, or neither (from before the rule).
FACT, TIE, OTHER = "fact", "tie", "other"
OVERRIDE_LABELS = {FACT: "definite fact", TIE: "tie-break", OTHER: "other"}


def cited_kinds(packet_json: str, decision_json: str) -> tuple[tuple[str, ...], str, float]:
    """The evidence kinds the AI's baseline_disagreement cites (in order), that text, and the baseline margin."""
    packet = json.loads(packet_json)
    reason = json.loads(decision_json).get("baseline_disagreement", "")
    kinds = {i["id"]: i["kind"] for p in packet["players"] for i in p.get("items", [])}
    cited = [kinds[i] for i in dict.fromkeys(re.findall(r"\bE\d+\b", reason)) if i in kinds]
    return tuple(dict.fromkeys(cited)), reason, float(packet.get("baseline_margin") or 0.0)


def override_kind(kinds: tuple[str, ...], margin: float) -> str:
    if set(kinds) & OVERRIDE_KINDS:
        return FACT
    return TIE if margin == 0 else OTHER


def actual_points(db: Database, player: dict[str, Any], season: int, week: int, scoring: Scoring) -> float | None:
    """The player's points that week; None while his team's game isn't in the data yet.

    `player` is a player entry of the stored packet (player_id and the team he was on when asked).
    """
    row = db.player_week(player["player_id"], season, week)
    if row is not None:
        return round(cast("dict[str, float | None]", row)[column("fp", scoring)] or 0.0, 2)
    team = player["team"]
    if team is None:  # no team: can't play
        return 0.0
    if not any(team in (g["home_team"], g["away_team"]) for g in db.week_games(season, week)):
        return 0.0  # bye
    return 0.0 if db.team_has_stats(season, week, team) else None


def hit(pick: str | None, points: dict[str, float]) -> int | None:
    """1 if `pick` outscored everyone, 0 if someone beat it, None for a tie at the top (or no pick)."""
    if pick is None:
        return None
    top = max(points.values())
    if points[pick] < top:
        return 0
    return None if list(points.values()).count(top) > 1 else 1


def consensus_pick(db: Database, rec: RecommendationRow, player_ids: list[str]) -> str | None:
    """The raw FantasyPros consensus pick on the day asked; None when it can't choose."""
    day = rec["created_at"][:10]
    ranks = {pid: db.latest_ranking(pid, rec["season"], rec["week"], through_day=day) for pid in player_ids}
    ranked = {pid: r for pid, r in ranks.items() if r is not None and r["ecr"] is not None}
    if len(ranked) <= 1:
        return next(iter(ranked), None)  # an unranked player loses to a ranked one
    if len({r["position"] for r in ranked.values()}) == 1:
        # Same position: the better (lower) average expert rank.
        values = {pid: -cast(float, r["ecr"]) for pid, r in ranked.items()}
    else:
        # Ranks aren't comparable across positions (is RB20 better than WR16?): compare projections.
        values = {pid: r["proj_ppr"] for pid, r in ranked.items() if r["proj_ppr"] is not None}
    if not values:
        return None
    best = max(values.values())
    leaders = [pid for pid, v in values.items() if v == best]
    return leaders[0] if len(leaders) == 1 else None


def evaluate(db: Database, rec: RecommendationRow, now: datetime) -> EvaluationRow | None:
    """The scored recommendation, or None while any player's game isn't in the data yet."""
    players = json.loads(rec["packet_json"])["players"]
    scoring = cast(Scoring, rec["scoring"])
    points: dict[str, float] = {}
    for p in players:
        value = actual_points(db, p, rec["season"], rec["week"], scoring)
        if value is None:
            return None
        points[p["player_id"]] = value
    top = max(points.values())
    leaders = [pid for pid, v in points.items() if v == top]
    consensus = consensus_pick(db, rec, list(points))
    return EvaluationRow(
        rec_id=rec["rec_id"],
        scored_at=utc_iso(now),
        season=rec["season"],
        week=rec["week"],
        points_json=json.dumps(points),
        top_scorer=leaders[0] if len(leaders) == 1 else None,
        pick_hit=hit(rec["pick"], points),
        baseline_hit=hit(rec["baseline_pick"], points),
        consensus_pick=consensus,
        consensus_hit=hit(consensus, points),
        crowd_hit=hit(rec["crowd_pick"], points),
    )


def score_pending(db: Database, now: datetime | None = None) -> int:
    """Score every recommendation whose games are all in the data; returns how many were scored."""
    now = now or datetime.now(UTC)
    rows = [row for rec in db.unscored_recommendations() if (row := evaluate(db, rec, now)) is not None]
    return db.upsert("evaluations", rows)


# --- report -------------------------------------------------------------------------------


@dataclass
class Rate:
    hits: int = 0
    decided: int = 0  # matchups with a winner (ties don't count either way)

    def add(self, result: int | None) -> None:
        if result is not None:
            self.decided += 1
            self.hits += result

    @property
    def pct(self) -> float | None:
        return self.hits / self.decided if self.decided else None

    @property
    def margin(self) -> float | None:
        """The 95% margin of error of `pct` (normal approximation), as a fraction."""
        p = self.pct
        return 1.96 * math.sqrt(p * (1 - p) / self.decided) if p is not None else None

    def __str__(self) -> str:
        return f"{self.hits}/{self.decided} ({self.pct:.0%})" if self.pct is not None else "-"


@dataclass
class Summary:
    """Hit rates for a set of scored recommendations.

    The AI, baseline and consensus are compared on the same matchups: the ones the AI answered.
    The baseline and consensus are also scored on every matchup (`*_all`). The crowd is scored
    only where it had a pick, so its rates count fewer matchups.
    """

    matchups: int = 0
    ai_answered: int = 0
    baseline_all: Rate = field(default_factory=Rate)
    consensus_all: Rate = field(default_factory=Rate)
    crowd_all: Rate = field(default_factory=Rate)
    ai: Rate = field(default_factory=Rate)
    baseline: Rate = field(default_factory=Rate)
    consensus: Rate = field(default_factory=Rate)
    crowd: Rate = field(default_factory=Rate)
    overrides: Rate = field(default_factory=Rate)  # AI picks that went against the baseline: was the AI right?
    overrides_by_kind: dict[str, Rate] = field(default_factory=lambda: {k: Rate() for k in OVERRIDE_LABELS})
    by_confidence: dict[str, Rate] = field(default_factory=lambda: {c: Rate() for c in CONFIDENCES})

    def add(self, row: dict[str, Any]) -> None:
        self.matchups += 1
        self.baseline_all.add(row["baseline_hit"])
        self.consensus_all.add(row["consensus_hit"])
        self.crowd_all.add(row["crowd_hit"])
        if row["source"] not in AI_SOURCES:
            return
        self.ai_answered += 1
        self.ai.add(row["pick_hit"])
        self.baseline.add(row["baseline_hit"])
        self.consensus.add(row["consensus_hit"])
        self.crowd.add(row["crowd_hit"])
        if row["pick"] != row["baseline_pick"]:
            self.overrides.add(row["pick_hit"])
            kinds, _, margin = cited_kinds(row["packet_json"], row["decision_json"])
            self.overrides_by_kind[override_kind(kinds, margin)].add(row["pick_hit"])
        if row["confidence"] in self.by_confidence:
            self.by_confidence[row["confidence"]].add(row["pick_hit"])


@dataclass(frozen=True)
class Matchup:
    """One scored recommendation, by player name, for the track record and its "Why" links."""

    rec_id: str
    week: int
    scoring: str
    points: dict[str, float]  # player name -> actual points, in the order asked
    pick: str  # the pick shown
    baseline_pick: str
    consensus_pick: str | None
    ai_answered: bool
    pick_hit: int | None  # 1/0, None for a tie
    baseline_hit: int | None
    consensus_hit: int | None
    crowd_pick: str | None = None
    crowd_hit: int | None = None
    reason: str = ""  # why the AI went against the baseline (its baseline_disagreement), if it did
    reason_kinds: tuple[str, ...] = ()  # the kinds of evidence that reason cites, e.g. ("news",)
    baseline_margin: float = 0.0  # the baseline's edge in points when asked (0 = a tie)

    @property
    def overrode(self) -> bool:
        """Whether the AI went against the baseline."""
        return self.ai_answered and self.pick != self.baseline_pick

    @property
    def override_kind(self) -> str:
        """fact, tie or other (OVERRIDE_LABELS), for an override."""
        return override_kind(self.reason_kinds, self.baseline_margin)

    @staticmethod
    def from_row(row: dict[str, Any]) -> Matchup:
        players = json.loads(row["packet_json"])["players"]
        names = {p["player_id"]: p["name"] for p in players}
        points = json.loads(row["points_json"])
        kinds, reason, margin = cited_kinds(row["packet_json"], row["decision_json"])
        return Matchup(
            rec_id=row["rec_id"],
            week=row["week"],
            scoring=row["scoring"],
            points={names[pid]: points[pid] for pid in names},
            pick=names[row["pick"]],
            baseline_pick=names[row["baseline_pick"]],
            consensus_pick=names.get(row["consensus_pick"]) if row["consensus_pick"] else None,
            ai_answered=row["source"] in AI_SOURCES,
            pick_hit=row["pick_hit"],
            baseline_hit=row["baseline_hit"],
            consensus_hit=row["consensus_hit"],
            crowd_pick=names.get(row["crowd_pick"]) if row["crowd_pick"] else None,
            crowd_hit=row["crowd_hit"],
            reason=reason,
            reason_kinds=kinds,
            baseline_margin=margin,
        )


@dataclass
class Report:
    season: int
    weeks: dict[str, dict[int, Summary]]  # group ("users" | "benchmark" | "backfill") -> week -> summary
    totals: dict[str, Summary]
    matchups: dict[str, list[Matchup]] = field(default_factory=dict)  # group -> newest week first

    def overrides(self, group: str) -> list[Matchup]:
        """The group's matchups where the AI went against the baseline, newest week first."""
        return [m for m in self.matchups.get(group, []) if m.overrode]


def group_of(row: dict[str, Any]) -> str:
    return GROUPS.get(row["visitor"], "users")


def build_report(db: Database, season: int) -> Report:
    weeks: dict[str, dict[int, Summary]] = defaultdict(lambda: defaultdict(Summary))
    totals: dict[str, Summary] = defaultdict(Summary)
    matchups: dict[str, list[Matchup]] = defaultdict(list)
    for row in db.scored_recommendations(season):
        group = group_of(row)
        weeks[group][row["week"]].add(row)
        totals[group].add(row)
        matchups[group].append(Matchup.from_row(row))
    return Report(
        season,
        {g: dict(w) for g, w in weeks.items()},
        dict(totals),
        {g: sorted(m, key=lambda x: -x.week) for g, m in matchups.items()},  # stable: asked order within a week
    )


def report_text(report: Report) -> str:
    lines = [
        f"Season {report.season} track record. On the matchups the AI answered: AI, baseline, consensus, "
        "Reddit crowd (where it leaned), and the AI's picks against the baseline. On all matchups: "
        "baseline, consensus, crowd."
    ]
    for group in ("users", *GROUPS.values()):
        if group not in report.totals:
            continue
        lines.append(f"\n{group}:")
        rows = [*(f"week {w}" for w in sorted(report.weeks[group])), "season"]
        summaries = [*(report.weeks[group][w] for w in sorted(report.weeks[group])), report.totals[group]]
        for label, s in zip(rows, summaries, strict=True):
            lines.append(
                f"  {label:>8}: {s.ai_answered} AI-answered | AI {s.ai} | baseline {s.baseline} "
                f"| consensus {s.consensus} | crowd {s.crowd} | AI overrides {s.overrides} || all {s.matchups}: "
                f"baseline {s.baseline_all} | consensus {s.consensus_all} | crowd {s.crowd_all}"
            )
        total = report.totals[group]
        lines.append("  AI by confidence: " + ", ".join(f"{c} {total.by_confidence[c]}" for c in CONFIDENCES))
        if total.overrides.decided:
            by_kind = total.overrides_by_kind
            lines.append(
                "  AI overrides by reason: " + ", ".join(f"{OVERRIDE_LABELS[k]} {by_kind[k]}" for k in by_kind)
            )
        overrides = report.overrides(group)
        if overrides:
            lines.append("  AI overrides of the baseline (reason type; actual points; evidence cited; reason):")
        for m in overrides:
            verdict = "right" if m.pick_hit == 1 else "wrong" if m.pick_hit == 0 else "tie"
            lines.append(
                f"    week {m.week} {m.scoring}: {verdict:5} {OVERRIDE_LABELS[m.override_kind]}; "
                f"{m.pick} {m.points[m.pick]:g} over "
                f"{m.baseline_pick} {m.points[m.baseline_pick]:g}; {', '.join(m.reason_kinds) or 'no ids'}; "
                f"{m.reason[:REASON_CHARS]}"
            )
    if len(lines) == 1:
        lines.append("Nothing scored yet.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score recommendations against actual points and print hit rates.")
    parser.add_argument("--season", type=int, default=None, help="default: the latest season in the data")
    parser.add_argument("--db", default=None, help="SQLite path or postgres:// URL")
    args = parser.parse_args()
    with Database(args.db) as db:
        scored = score_pending(db)
        season = args.season or db.latest_season()
        print(f"Scored {scored} new recommendations.")
        if season is not None:
            print(report_text(build_report(db, season)))


if __name__ == "__main__":
    main()
