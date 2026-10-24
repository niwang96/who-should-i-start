"""Stage 2: assemble the evidence packet for a start/sit question (code only, no LLM).

Each fact is an EvidenceItem with an id (E1, E2, ...) that the Stage 3 LLM must cite;
the packet's text form is what the LLM sees, so it stays compact.

Try it: python -m startsit.evidence "DeVonta Smith" "Puka Nacua" --scoring half
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from startsit import baseline
from startsit.db import Database

NFLVERSE_URL = "https://github.com/nflverse/nflverse-data"


@dataclass
class EvidenceItem:
    id: str
    player_id: str
    kind: str  # consensus | projection | usage | injury | availability
    text: str
    source_url: str | None = None


@dataclass
class PlayerEvidence:
    player_id: str
    name: str
    position: str
    team: str | None
    baseline: baseline.BaselineScore
    items: list[EvidenceItem] = field(default_factory=list)


@dataclass
class EvidencePacket:
    season: int
    week: int
    scoring: str
    players: list[PlayerEvidence]
    baseline_pick: str
    baseline_margin: float

    @property
    def item_ids(self) -> set[str]:
        return {i.id for p in self.players for i in p.items}

    def to_text(self) -> str:
        label = baseline.SCORING_LABELS[self.scoring]
        names = {p.player_id: p.name for p in self.players}
        if self.baseline_margin == 0:
            verdict = "Baseline (100% FantasyPros consensus) has them tied"
        else:
            verdict = (f"Baseline (100% FantasyPros consensus) favors {names[self.baseline_pick]} "
                       f"by {self.baseline_margin:.1f} pts")
        lines = [f"Week {self.week} {self.season}, {label} scoring. {verdict}."]
        for p in self.players:
            note = f" ({p.baseline.note})" if p.baseline.note else ""
            lines.append(f"{p.name} ({p.position}, {p.team}): baseline {p.baseline.points:.1f} pts{note}")
            lines.extend(f"  {i.id} [{i.kind}] {i.text}" for i in p.items)
        return "\n".join(lines)


class PlayerLookupError(ValueError):
    pass


def find_player(db: Database, name: str) -> dict:
    """Exact (case-insensitive) name match, else a unique substring match."""
    rows = db.query("SELECT * FROM players WHERE lower(name) = lower(?)", (name,))
    if not rows:
        rows = db.query("SELECT * FROM players WHERE lower(name) LIKE lower(?)", (f"%{name}%",))
    if len(rows) == 1:
        return rows[0]
    if not rows:
        raise PlayerLookupError(f"no player matches {name!r}")
    options = ", ".join(f"{r['name']} ({r['position']}, {r['team']})" for r in rows[:5])
    raise PlayerLookupError(f"{name!r} is ambiguous: {options}")


def current_week(db: Database, season: int) -> int:
    """The latest week with rankings, else the week after the latest game played."""
    row = db.query("SELECT max(week) w FROM rankings WHERE season = ?", (season,))[0]
    if row["w"] is not None:
        return row["w"]
    row = db.query("SELECT max(week) w FROM player_weeks WHERE season = ?", (season,))[0]
    return (row["w"] or 0) + 1


def _one(db: Database, sql: str, params: tuple) -> dict | None:
    rows = db.query(sql, params)
    return rows[0] if rows else None


def _fmt_rank(x: float | None) -> str:
    return "?" if x is None else f"{x:g}"


def build_packet(db: Database, player_ids: list[str], scoring: str, season: int, week: int) -> EvidencePacket:
    if scoring not in baseline.SCORINGS:
        raise ValueError(f"scoring must be one of {baseline.SCORINGS}")
    if len(set(player_ids)) < 2:
        raise ValueError("need at least two different players")
    label = baseline.SCORING_LABELS[scoring]
    week_has_games = _one(db, "SELECT 1 x FROM games WHERE season = ? AND week = ?", (season, week)) is not None
    players: list[PlayerEvidence] = []
    counter = 0

    def item(pid: str, kind: str, text: str, url: str | None = None) -> EvidenceItem:
        nonlocal counter
        counter += 1
        return EvidenceItem(f"E{counter}", pid, kind, text, url)

    for pid in player_ids:
        player = _one(db, "SELECT * FROM players WHERE player_id = ?", (pid,))
        if player is None:
            raise PlayerLookupError(f"unknown player id {pid!r}")
        rank = _one(
            db,
            "SELECT * FROM rankings WHERE season = ? AND week = ? AND player_id = ? "
            "ORDER BY scrape_date DESC LIMIT 1",
            (season, week, pid),
        )
        usage = _one(
            db,
            "SELECT * FROM usage_features WHERE season = ? AND player_id = ? AND through_week < ? "
            "ORDER BY through_week DESC LIMIT 1",
            (season, pid, week),
        )
        injury = _one(db, "SELECT * FROM injuries WHERE season = ? AND week = ? AND player_id = ?", (season, week, pid))
        on_bye = week_has_games and _one(
            db,
            "SELECT 1 x FROM games WHERE season = ? AND week = ? AND ? IN (home_team, away_team)",
            (season, week, player["team"]),
        ) is None
        exp_rec = usage["xfp_ppr_l3"] - usage["xfp_std_l3"] if usage else None

        score = baseline.score(
            baseline.BaselineInput(
                player_id=pid,
                position=player["position"],
                proj_ppr=rank["proj_ppr"] if rank else None,
                exp_receptions=exp_rec,
                on_bye=on_bye,
                report_status=injury["report_status"] if injury else None,
            ),
            scoring,
        )
        pe = PlayerEvidence(pid, player["name"], player["position"], player["team"], score)

        if on_bye:
            pe.items.append(item(pid, "availability", f"{player['team']} is on bye in week {week}", NFLVERSE_URL))
        if rank:
            grade = f"; start/sit grade {rank['start_sit_grade']}" if rank["start_sit_grade"] else ""
            pe.items.append(item(
                pid, "consensus",
                f"FantasyPros consensus {rank['pos_rank']} {rank['opponent']} (avg rank {_fmt_rank(rank['ecr'])}, "
                f"sd {_fmt_rank(rank['ecr_sd'])}, range {_fmt_rank(rank['ecr_best'])}-{_fmt_rank(rank['ecr_worst'])}"
                f"{grade}; rankings as of {rank['scrape_date']})",
                rank["source_url"],
            ))
            if scoring == "ppr" or player["position"] == "QB" or exp_rec is None:
                proj = f"Consensus projection {rank['proj_ppr']:.1f} pts"
            else:
                proj = (f"Consensus projection {rank['proj_ppr']:.1f} PPR, about {score.points:.1f} {label} "
                        f"(minus {1 - baseline.POINTS_PER_RECEPTION[scoring]:g} x {exp_rec:.1f} expected catches/game)")
            pe.items.append(item(pid, "projection", proj, rank["source_url"]))
        elif not on_bye:
            pe.items.append(item(pid, "consensus", f"Not in FantasyPros week {week} rankings (experts don't rank them)"))
        if usage:
            trend = ""
            if usage["snap_pct_trend"] is not None:
                trend = f" ({usage['snap_pct_trend'] * 100:+.0f} pts vs prior 3)"
            missed = _one(
                db,
                "SELECT count(*) n FROM games WHERE season = ? AND week > ? AND week <= ? "
                "AND ? IN (home_team, away_team)",
                (season, usage["last_game_week"], usage["through_week"], player["team"]),
            )["n"]
            stale = f"; has missed {player['team']}'s last {missed} games (last played week {usage['last_game_week']})" if missed else ""
            pe.items.append(item(
                pid, "usage",
                f"Last {min(usage['games'], 3)} games played: "
                f"{_pct(usage['snap_pct_l3'])} snaps{trend}, {_pct(usage['target_share_l3'])} target share, "
                f"{usage['opportunities_l3']:.1f} opportunities/game; {usage[f'xfp_{scoring}_l3']:.1f} expected vs "
                f"{usage[f'fp_{scoring}_l3']:.1f} actual {label} pts/game{stale}",
                NFLVERSE_URL,
            ))
        if injury and (injury["report_status"] or injury["practice_status"]):
            status = injury["report_status"] or "no game status yet"
            practice = f", practice: {injury['practice_status']}" if injury["practice_status"] else ""
            pe.items.append(item(
                pid, "injury",
                f"Week {week} injury report: {status} ({injury['injury'] or 'unspecified'}){practice}",
                NFLVERSE_URL,
            ))
        players.append(pe)

    winner, margin = baseline.pick([p.baseline for p in players])
    return EvidencePacket(season, week, scoring, players, winner, margin)


def _pct(x: float | None) -> str:
    return "?%" if x is None else f"{x * 100:.0f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the evidence packet for a start/sit question.")
    parser.add_argument("players", nargs="+", help="two or more player names")
    parser.add_argument("--scoring", choices=baseline.SCORINGS, default="ppr")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None, help="default: latest week with rankings")
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    db = Database(args.db)
    try:
        season = args.season or db.query("SELECT max(season) s FROM player_weeks")[0]["s"]
        week = args.week or current_week(db, season)
        ids = [find_player(db, n)["player_id"] for n in args.players]
        print(build_packet(db, ids, args.scoring, season, week).to_text())
    except PlayerLookupError as e:
        parser.exit(1, f"error: {e}\n")
    finally:
        db.close()


if __name__ == "__main__":
    main()
