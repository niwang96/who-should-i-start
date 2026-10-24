"""Stage 2: assemble the evidence packet for a start/sit question (code only, no LLM).

Three steps, kept separate so new signals (news, Reddit) slot in without touching the rest:
1. `load_context` gathers everything known about one player for the week (typed rows).
2. Each function in `ITEM_BUILDERS` turns a context into zero or more facts of one kind.
3. `build_packet` scores the baseline, runs the builders, and numbers the facts E1, E2, ...

The Stage 3 LLM must cite those ids; `EvidencePacket.to_text()` is what it sees, so it
stays compact and unambiguous.

Try it: python -m startsit.evidence "DeVonta Smith" "Puka Nacua" --scoring half
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, NamedTuple, cast

from startsit import baseline
from startsit.db import Database, InjuryRow, PlayerRow, RankingRow, RosterRow, UsageRow
from startsit.scoring import LABELS, SCORINGS, Scoring, column, reception_discount

NFLVERSE_URL = "https://github.com/nflverse/nflverse-data"

EvidenceKind = Literal["availability", "consensus", "projection", "usage", "injury"]


@dataclass(frozen=True)
class Question:
    season: int
    week: int
    scoring: Scoring

    @property
    def label(self) -> str:
        return LABELS[self.scoring]


@dataclass(frozen=True)
class PlayerContext:
    """Everything known about one player for the question's week."""

    player: PlayerRow
    team: str | None  # roster team that week, else the latest known; None if cut/retired
    roster: RosterRow | None
    ranking: RankingRow | None
    usage: UsageRow | None
    injury: InjuryRow | None
    on_bye: bool
    missed_games: int  # games the player's team played since the player last played (0 if unknown)

    @property
    def exp_receptions(self) -> float | None:
        # FF Opportunity's PPR and standard expected points differ by exactly the expected catches.
        return self.usage["xfp_ppr_l3"] - self.usage["xfp_std_l3"] if self.usage else None


class Fact(NamedTuple):
    kind: EvidenceKind
    text: str
    source_url: str | None = None


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    player_id: str
    kind: EvidenceKind
    text: str
    source_url: str | None = None


@dataclass
class PlayerEvidence:
    player_id: str
    name: str
    position: str
    team: str | None
    score: baseline.BaselineScore
    items: list[EvidenceItem] = field(default_factory=list)


@dataclass
class EvidencePacket:
    question: Question
    players: list[PlayerEvidence]
    baseline_pick: str
    baseline_margin: float

    @property
    def item_ids(self) -> set[str]:
        return {i.id for p in self.players for i in p.items}

    def verdict(self) -> str:
        leaders = baseline.leaders([p.score for p in self.players])
        names = {p.player_id: p.name for p in self.players}
        prefix = f"Baseline ({baseline.DESCRIPTION})"
        if len(leaders) > 1:
            tied = " and ".join(names[s.player_id] for s in leaders)
            return f"{prefix} has {tied} tied at {leaders[0].points:.1f} pts"
        return f"{prefix} favors {names[self.baseline_pick]} by {self.baseline_margin:.1f} pts"

    def to_text(self) -> str:
        q = self.question
        lines = [f"Week {q.week} {q.season}, {q.label} scoring. {self.verdict()}."]
        for p in self.players:
            note = f" ({p.score.note})" if p.score.note else ""
            lines.append(f"{p.name} ({p.position}, {p.team or 'FA'}): baseline {p.score.points:.1f} pts{note}")
            lines.extend(f"  {i.id} [{i.kind}] {i.text}" for i in p.items)
        return "\n".join(lines)


class PlayerLookupError(ValueError):
    pass


def find_player(db: Database, name: str, team: str | None = None) -> PlayerRow:
    """The one player matching `name` (and `team`, if given), else PlayerLookupError."""
    rows = db.players_named(name, team)
    if len(rows) == 1:
        return rows[0]
    on_team = f" on {team}" if team else ""
    if not rows:
        raise PlayerLookupError(f"no player matches {name!r}{on_team}")
    options = ", ".join(f"{r['name']} ({r['team']})" for r in rows[:5])
    raise PlayerLookupError(
        f'{name!r}{on_team} is ambiguous: {options}; add the team, e.g. "{rows[0]["name"]} ({rows[0]["team"]})"'
    )


def current_week(db: Database, season: int) -> int:
    """The latest week with rankings, else the week after the latest game played."""
    week = db.latest_rankings_week(season)
    if week is not None:
        return week
    return (db.latest_played_week(season) or 0) + 1


# --- 1. load ------------------------------------------------------------------------


def load_context(db: Database, player_id: str, q: Question) -> PlayerContext:
    player = db.player(player_id)
    if player is None:
        raise PlayerLookupError(f"unknown player id {player_id!r}")
    roster = db.roster_status(player_id, q.season, q.week)
    team = (roster and roster["team"]) or player["team"]
    if roster and roster["status"] in baseline.NO_TEAM_STATUSES:
        team = None  # the roster still lists the old team
    usage = db.usage_before(player_id, q.season, q.week)
    on_bye = (
        team is not None
        and db.week_has_games(q.season, q.week)
        and db.team_games(q.season, team, q.week - 1, q.week) == 0
    )
    missed = 0
    if usage:
        after, through = usage["last_game_week"], usage["through_week"]
        by_roster = db.missed_games(player_id, q.season, after, through)
        if by_roster is not None:
            missed = by_roster
        elif team is not None:
            missed = db.team_games(q.season, team, after, through)
    return PlayerContext(
        player=player,
        team=team,
        roster=roster,
        ranking=db.latest_ranking(player_id, q.season, q.week),
        usage=usage,
        injury=db.injury(player_id, q.season, q.week),
        on_bye=on_bye,
        missed_games=missed,
    )


def baseline_input(ctx: PlayerContext) -> baseline.BaselineInput:
    return baseline.BaselineInput(
        player_id=ctx.player["player_id"],
        position=ctx.player["position"],
        proj_ppr=ctx.ranking["proj_ppr"] if ctx.ranking else None,
        exp_receptions=ctx.exp_receptions,
        on_bye=ctx.on_bye,
        report_status=ctx.injury["report_status"] if ctx.injury else None,
        roster_status=ctx.roster["status"] if ctx.roster else None,
    )


# --- 2. facts -------------------------------------------------------------------------


def _num(x: float | None) -> str:
    return "?" if x is None else f"{x:g}"


def _dec(x: float | None) -> str:
    return "?" if x is None else f"{x:.1f}"


def _plural(n: float, word: str) -> str:
    return f"{n:g} {word}" if n == 1 else f"{n:g} {word}s"


def _pct(x: float | None) -> str:
    return "?%" if x is None else f"{x * 100:.0f}%"


def availability_facts(ctx: PlayerContext, q: Question) -> list[Fact]:
    facts = []
    status = ctx.roster["status"] if ctx.roster else None
    if status is not None and status not in baseline.PLAYABLE_STATUSES:
        label = baseline.roster_status_label(status)
        facts.append(Fact("availability", f"Roster status: {label} (not on the active roster)", NFLVERSE_URL))
    if ctx.on_bye:
        facts.append(Fact("availability", f"{ctx.team} is on bye in week {q.week}", NFLVERSE_URL))
    return facts


def consensus_facts(ctx: PlayerContext, q: Question) -> list[Fact]:
    r = ctx.ranking
    if r is None:
        if ctx.on_bye:
            return []
        return [Fact("consensus", f"Not in FantasyPros week {q.week} rankings (experts don't rank them)")]
    grade = f"; start/sit grade {r['start_sit_grade']}" if r["start_sit_grade"] else ""
    rank_range = f"{_num(r['ecr_best'])}-{_num(r['ecr_worst'])}"
    opponent = f" {r['opponent']}" if r["opponent"] else ""
    facts = [
        Fact(
            "consensus",
            f"FantasyPros consensus {r['pos_rank']}{opponent} (avg rank {_num(r['ecr'])}, "
            f"sd {_num(r['ecr_sd'])}, range {rank_range}{grade}; rankings as of {r['scrape_date']})",
            r["source_url"],
        )
    ]
    if r["proj_ppr"] is not None:
        exp_rec = ctx.exp_receptions
        if not baseline.converts(ctx.player["position"], q.scoring):
            text = f"Consensus projection {r['proj_ppr']:.1f} pts"
        elif exp_rec is None:
            text = f"Consensus projection {r['proj_ppr']:.1f} PPR pts (no usage data to convert to {q.label})"
        else:
            converted = baseline.convert_ppr(r["proj_ppr"], exp_rec, q.scoring)
            per_catch = reception_discount(q.scoring)
            text = (
                f"Consensus projection {r['proj_ppr']:.1f} PPR pts, about {converted:.1f} {q.label} "
                f"(minus {per_catch:g} x {exp_rec:.1f} expected catches/game)"
            )
        facts.append(Fact("projection", text, r["source_url"]))
    return facts


def usage_facts(ctx: PlayerContext, q: Question) -> list[Fact]:
    u = ctx.usage
    if u is None:
        return []
    trend = ""
    if u["snap_pct_trend"] is not None:
        direction = "up" if u["snap_pct_trend"] >= 0 else "down"
        points = round(abs(u["snap_pct_trend"]) * 100)
        trend = f" ({_plural(points, 'percentage point')} {direction} vs prior 3)"
    stale = ""
    if ctx.missed_games:
        stale = (
            f"; has missed {_plural(ctx.missed_games, 'team game')} since last playing in week {u['last_game_week']}"
        )
    per_format = cast("dict[str, float]", u)  # per-format columns are looked up by computed name
    expected = per_format[column("xfp", q.scoring, "_l3")]
    actual = per_format[column("fp", q.scoring, "_l3")]
    return [
        Fact(
            "usage",
            f"Last {_plural(min(u['games'], 3), 'game')} played: {_pct(u['snap_pct_l3'])} of snaps{trend}, "
            f"{_pct(u['target_share_l3'])} target share, {_dec(u['opportunities_l3'])} opportunities/game; "
            f"{expected:.1f} expected vs {actual:.1f} actual {q.label} fantasy pts/game{stale}",
            NFLVERSE_URL,
        )
    ]


def injury_facts(ctx: PlayerContext, q: Question) -> list[Fact]:
    i = ctx.injury
    if i is None or not (i["report_status"] or i["practice_status"]):
        return []
    status = i["report_status"] or "no game status yet"
    practice = f", practice: {i['practice_status']}" if i["practice_status"] else ""
    return [
        Fact(
            "injury", f"Week {q.week} injury report: {status} ({i['injury'] or 'unspecified'}){practice}", NFLVERSE_URL
        )
    ]


# Order here is the order facts appear in the packet. Add new signals (news, Reddit) here.
ITEM_BUILDERS: tuple[Callable[[PlayerContext, Question], list[Fact]], ...] = (
    availability_facts,
    consensus_facts,
    usage_facts,
    injury_facts,
)


# --- 3. assemble ----------------------------------------------------------------------


def build_packet(db: Database, player_ids: list[str], scoring: Scoring, season: int, week: int) -> EvidencePacket:
    if scoring not in SCORINGS:
        raise ValueError(f"scoring must be one of {SCORINGS}")
    if len(set(player_ids)) < 2 or len(set(player_ids)) != len(player_ids):
        raise ValueError("need at least two different players")
    q = Question(season, week, scoring)

    players: list[PlayerEvidence] = []
    n = 0
    for pid in player_ids:
        ctx = load_context(db, pid, q)
        pe = PlayerEvidence(
            pid, ctx.player["name"], ctx.player["position"], ctx.team, baseline.score(baseline_input(ctx), scoring)
        )
        for build in ITEM_BUILDERS:
            for fact in build(ctx, q):
                n += 1
                pe.items.append(EvidenceItem(f"E{n}", pid, fact.kind, fact.text, fact.source_url))
        players.append(pe)

    winner, margin = baseline.pick([p.score for p in players])
    return EvidencePacket(q, players, winner, margin)


# --- CLI ------------------------------------------------------------------------------

_NAME_WITH_TEAM = re.compile(r"^(?P<name>.+?)\s*\((?P<team>[A-Za-z]{2,3})\)$")


def parse_player_arg(arg: str) -> tuple[str, str | None]:
    """'Ryan Miller (NE)' -> ('Ryan Miller', 'NE'); 'Puka Nacua' -> ('Puka Nacua', None)."""
    m = _NAME_WITH_TEAM.match(arg.strip())
    return (m["name"], m["team"].upper()) if m else (arg.strip(), None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the evidence packet for a start/sit question.")
    parser.add_argument(
        "players", nargs="+", help='two or more player names, optionally with a team: "Ryan Miller (NE)"'
    )
    parser.add_argument("--scoring", choices=SCORINGS, default="ppr")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None, help="default: latest week with rankings")
    parser.add_argument("--db", default=None, help="SQLite path or postgres:// URL")
    args = parser.parse_args()

    with Database(args.db) as db:
        try:
            season = args.season or db.latest_season()
            if season is None:
                parser.exit(1, "error: no data yet; run python -m startsit.ingest first\n")
            week = args.week or current_week(db, season)
            ids = [find_player(db, *parse_player_arg(a))["player_id"] for a in args.players]
            print(build_packet(db, ids, cast(Scoring, args.scoring), season, week).to_text())
        except (PlayerLookupError, ValueError) as e:
            parser.exit(1, f"error: {e}\n")


if __name__ == "__main__":
    main()
