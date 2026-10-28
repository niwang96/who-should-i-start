"""Stage 2: assemble the evidence packet for a start/sit question (code only, no LLM).

Three steps, kept separate so new signals (news, Reddit) slot in without touching the rest:
1. `load_context` gathers everything known about one player for the week (typed rows).
2. Each function in `ITEM_BUILDERS` turns a context into zero or more facts of one kind.
3. `build_packet` scores the baseline, runs the builders, and numbers the facts E1, E2, ...

The Stage 3 LLM must cite those ids; `EvidencePacket.to_text()` is what it sees, so it
stays compact and unambiguous.

See it: python -m startsit.decide "DeVonta Smith" "Puka Nacua" --scoring half --no-llm
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal, NamedTuple, cast
from zoneinfo import ZoneInfo

from startsit import baseline
from startsit.db import (
    Database,
    GameRow,
    InjuryRow,
    NewsFactRow,
    PlayerRow,
    ProjectionRow,
    RankingRow,
    RosterRow,
    UsageRow,
    utc_day,
    utc_iso,
    utc_now,
)
from startsit.names import PlayerMatcher
from startsit.scoring import LABELS, SCORINGS, Scoring, column, reception_discount

NFLVERSE_URL = "https://github.com/nflverse/nflverse-data"

EvidenceKind = Literal["availability", "consensus", "projection", "matchup", "usage", "injury", "news"]

NEWS_LOOKBACK = timedelta(days=7)
NEWS_PER_PLAYER = 3
EASTERN = ZoneInfo("America/New_York")  # the schedule's game days and kickoff times are US Eastern
GAME_LENGTH = timedelta(hours=4)  # kickoff to final whistle, with margin


@dataclass(frozen=True)
class Question:
    season: int
    week: int
    scoring: Scoring
    as_of: datetime = field(default_factory=utc_now)  # news cutoff: nothing after this

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
    news: list[NewsFactRow]  # newest first
    game: GameRow | None  # the team's game this week, with betting lines
    on_bye: bool
    missed_games: int  # games the player's team played since the player last played (0 if unknown)
    projection: ProjectionRow | None = None  # FantasyPros per-format projection
    news_status: NewsFactRow | None = None  # newest news fact this week that states a game status

    @property
    def exp_receptions(self) -> float | None:
        # FF Opportunity's PPR and standard expected points differ by exactly the expected catches.
        return self.usage["xfp_ppr_l3"] - self.usage["xfp_std_l3"] if self.usage else None

    @property
    def baseline_input(self) -> baseline.BaselineInput:
        return baseline.BaselineInput(
            player_id=self.player["player_id"],
            position=self.player["position"],
            proj_ppr=self.ranking["proj_ppr"] if self.ranking else None,
            exp_receptions=self.exp_receptions,
            on_bye=self.on_bye,
            report_status=self.injury["report_status"] if self.injury else None,
            roster_status=self.roster["status"] if self.roster else None,
            proj_receptions=self.projection["receptions"] if self.projection else None,
            news_status=self.news_status["game_status"] if self.news_status else None,
        )


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
    game_status: str | None = None  # freshest of news and injury report, lowercase
    kickoff: datetime | None = None  # UTC; None if unknown or no game
    game_started: bool = False  # kicked off before the question was asked (locked in most leagues)

    def items_of(self, kind: EvidenceKind) -> list[EvidenceItem]:
        return [i for i in self.items if i.kind == kind]


@dataclass
class EvidencePacket:
    question: Question
    players: list[PlayerEvidence]
    baseline_pick: str
    baseline_margin: float

    @property
    def items(self) -> list[EvidenceItem]:
        return [i for p in self.players for i in p.items]

    @property
    def item_ids(self) -> set[str]:
        return {i.id for i in self.items}

    @property
    def names(self) -> dict[str, str]:
        """Player id -> name."""
        return {p.player_id: p.name for p in self.players}

    def player(self, player_id: str) -> PlayerEvidence:
        return next(p for p in self.players if p.player_id == player_id)

    def verdict(self) -> str:
        leaders = baseline.leaders([p.score for p in self.players])
        names = self.names
        prefix = f"Baseline ({baseline.DESCRIPTION})"
        if len(leaders) > 1:
            tied = " and ".join(names[s.player_id] for s in leaders)
            return f"{prefix} has {tied} tied at {leaders[0].points:.1f} pts"
        return f"{prefix} favors {names[self.baseline_pick]} by {self.baseline_margin:.1f} pts"

    def to_json(self) -> str:
        """The whole packet (question, scores, evidence items with ids and links), for the record."""
        return json.dumps(asdict(self), default=lambda o: o.isoformat() if isinstance(o, datetime) else str(o))

    @classmethod
    def from_json(cls, text: str) -> EvidencePacket:
        """A packet stored by `to_json` (with a recommendation), to show it again exactly as it was."""
        data = json.loads(text)
        q = data["question"]
        players = [
            PlayerEvidence(
                player_id=p["player_id"],
                name=p["name"],
                position=p["position"],
                team=p["team"],
                score=baseline.BaselineScore(**p["score"]),
                items=[EvidenceItem(**i) for i in p["items"]],
                game_status=p.get("game_status"),
                kickoff=datetime.fromisoformat(p["kickoff"]) if p.get("kickoff") else None,
                game_started=p.get("game_started", False),
            )
            for p in data["players"]
        ]
        question = Question(q["season"], q["week"], q["scoring"], datetime.fromisoformat(q["as_of"]))
        return cls(question, players, data["baseline_pick"], data["baseline_margin"])

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


def find_player(db: Database, name: str, team: str | None = None, matcher: PlayerMatcher | None = None) -> PlayerRow:
    """The one player matching `name` (and `team`, if given), else PlayerLookupError.

    Pass a `matcher` to reuse one player index across several lookups.
    """
    rows = (matcher or PlayerMatcher(db.all_players())).candidates(name, team)
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


def kickoff(game: GameRow) -> datetime | None:
    """Kickoff in UTC, from the schedule's US Eastern date and time; None if either is missing."""
    if not game["gameday"] or not game["gametime"]:
        return None
    try:
        local = datetime.fromisoformat(f"{game['gameday']}T{game['gametime']}")
    except ValueError:
        return None
    return local.replace(tzinfo=EASTERN).astimezone(UTC)


def game_over_by(game: GameRow) -> datetime | None:
    """A time by which the game is surely over: kickoff + 4 hours, else the end of its game day (US Eastern)."""
    if start := kickoff(game):
        return start + GAME_LENGTH
    if not game["gameday"]:
        return None
    return datetime.fromisoformat(f"{game['gameday']}T23:59:59").replace(tzinfo=EASTERN).astimezone(UTC)


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
    since, until = utc_iso(q.as_of - NEWS_LOOKBACK), utc_iso(q.as_of)
    asked_on = utc_day(q.as_of)  # rankings and projections archived on or before the day asked
    news = db.news_for_player(player_id, since, until, NEWS_PER_PLAYER)
    # A status only counts for this week if it came after the team's previous game ended, so
    # last week's "ruled out" or an in-game "won't return" doesn't carry over.
    last_game = db.team_last_game(q.season, team, q.week) if team else None
    last_game_over = game_over_by(last_game) if last_game else None
    status_since = max(since, utc_iso(last_game_over)) if last_game_over else since
    news_status = db.latest_news_status(player_id, status_since, until)
    if news_status and news_status not in news:
        news = sorted([*news, news_status], key=lambda n: n["published_at"], reverse=True)
    return PlayerContext(
        player=player,
        team=team,
        roster=roster,
        ranking=db.latest_ranking(player_id, q.season, q.week, through_day=asked_on),
        usage=usage,
        injury=db.injury(player_id, q.season, q.week),
        game=db.team_game(q.season, q.week, team) if team else None,
        news=news,
        on_bye=on_bye,
        missed_games=missed,
        projection=db.latest_projection(player_id, q.season, q.week, through_day=asked_on),
        news_status=news_status,
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
        inp = ctx.baseline_input
        receptions = baseline.conversion_receptions(inp)
        if not baseline.converts(ctx.player["position"], q.scoring):
            text = f"Consensus projection {r['proj_ppr']:.1f} pts"
        elif receptions is None:
            text = f"Consensus projection {r['proj_ppr']:.1f} PPR pts (no reception data to convert to {q.label})"
        else:
            converted = baseline.convert_ppr(r["proj_ppr"], receptions, q.scoring)
            per_catch = reception_discount(q.scoring)
            basis = (
                "FantasyPros projected catches"
                if inp.proj_receptions is not None
                else "expected catches/game from recent usage"
            )
            text = (
                f"Consensus projection {r['proj_ppr']:.1f} PPR pts, about {converted:.1f} {q.label} "
                f"(minus {per_catch:g} x {receptions:.1f} {basis})"
            )
        facts.append(Fact("projection", text, r["source_url"]))
    return facts


def team_spread(game: GameRow, team: str) -> float | None:
    """Points the team is favored by (negative: underdog)."""
    if game["spread_line"] is None:
        return None
    return game["spread_line"] if team == game["home_team"] else -game["spread_line"]


def implied_points(game: GameRow, team: str) -> float | None:
    """The team's implied points from the betting lines: half the total, plus half the spread."""
    spread = team_spread(game, team)
    if spread is None or game["total_line"] is None:
        return None
    return (game["total_line"] + spread) / 2


def eastern_time(dt: datetime) -> str:
    """'Sun 1:00 PM ET' (portable: no platform-specific strftime flags)."""
    local = dt.astimezone(EASTERN)
    return f"{local:%a} {local.hour % 12 or 12}:{local:%M %p} ET"


def matchup_facts(ctx: PlayerContext, q: Question) -> list[Fact]:
    g, team = ctx.game, ctx.team
    if g is None or team is None:
        return []
    implied, spread = implied_points(g, team), team_spread(g, team)
    if implied is None or spread is None:
        return []
    home = team == g["home_team"]
    opponent = g["away_team"] if home else g["home_team"]
    edge = f"favored by {spread:g}" if spread > 0 else f"underdog by {-spread:g}" if spread < 0 else "pick'em"
    start = kickoff(g)
    when = f", kickoff {eastern_time(start)}" if start else ""
    return [
        Fact(
            "matchup",
            f"{team} {'vs' if home else 'at'} {opponent}{when}: betting lines imply {implied:.1f} team points "
            f"({edge}, game total {g['total_line']:g})",
            NFLVERSE_URL,
        )
    ]


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


def news_facts(ctx: PlayerContext, q: Question) -> list[Fact]:
    return [
        Fact(
            "news",
            f"{n['published_at'][:10]} {n['source']}: {n['summary']} ({n['event']}, {n['severity']} impact)",
            n["url"],
        )
        for n in ctx.news
    ]


# Order here is the order facts appear in the packet. Add new signals (e.g. Reddit) here.
ITEM_BUILDERS: tuple[Callable[[PlayerContext, Question], list[Fact]], ...] = (
    availability_facts,
    consensus_facts,
    matchup_facts,
    usage_facts,
    injury_facts,
    news_facts,
)


# --- 3. assemble ----------------------------------------------------------------------


def build_packet(
    db: Database,
    player_ids: list[str],
    scoring: Scoring,
    season: int,
    week: int,
    as_of: datetime | None = None,
) -> EvidencePacket:
    """The packet for a question; `as_of` (default now) cuts off news, so past packets can be rebuilt."""
    if scoring not in SCORINGS:
        raise ValueError(f"scoring must be one of {SCORINGS}")
    if len(set(player_ids)) < 2 or len(set(player_ids)) != len(player_ids):
        raise ValueError("need at least two different players")
    q = Question(season, week, scoring, as_of) if as_of else Question(season, week, scoring)

    players: list[PlayerEvidence] = []
    n = 0
    for pid in player_ids:
        ctx = load_context(db, pid, q)
        inp = ctx.baseline_input
        start = kickoff(ctx.game) if ctx.game else None
        pe = PlayerEvidence(
            pid,
            ctx.player["name"],
            ctx.player["position"],
            ctx.team,
            baseline.score(inp, scoring),
            game_status=inp.game_status,
            kickoff=start,
            game_started=start is not None and start <= q.as_of,
        )
        for build in ITEM_BUILDERS:
            for fact in build(ctx, q):
                n += 1
                pe.items.append(EvidenceItem(f"E{n}", pid, fact.kind, fact.text, fact.source_url))
        players.append(pe)

    winner, margin = baseline.pick([p.score for p in players])
    return EvidencePacket(q, players, winner, margin)


# --- resolving a question from player names -------------------------------------------

_NAME_WITH_TEAM = re.compile(r"^(?P<name>.+?)\s*\((?P<team>[A-Za-z]{2,3})\)$")


def parse_player_arg(arg: str) -> tuple[str, str | None]:
    """'Ryan Miller (NE)' -> ('Ryan Miller', 'NE'); 'Puka Nacua' -> ('Puka Nacua', None)."""
    m = _NAME_WITH_TEAM.match(arg.strip())
    return (m["name"], m["team"].upper()) if m else (arg.strip(), None)


def packet_for_names(
    db: Database,
    names: list[str],
    scoring: Scoring,
    season: int | None = None,
    week: int | None = None,
    matcher: PlayerMatcher | None = None,
    as_of: datetime | None = None,
) -> EvidencePacket:
    """Resolve player names (optionally "Name (TEAM)") and default season/week, then build the packet."""
    season = season or db.latest_season()
    if season is None:
        raise ValueError("no data yet; run python -m startsit.ingest first")
    week = week or current_week(db, season)
    matcher = matcher or PlayerMatcher(db.all_players())
    ids = [find_player(db, *parse_player_arg(name), matcher=matcher)["player_id"] for name in names]
    return build_packet(db, ids, scoring, season, week, as_of)
