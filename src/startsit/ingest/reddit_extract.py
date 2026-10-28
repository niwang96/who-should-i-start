"""Turn Reddit WDIS comments into start/sit votes with one batched LLM call per chunk (step 6b).

Most comments in the "Who Do I Start" threads are questions ("Pick 2: AJB, Moore, Cooper"); the
crowd's opinion is in the replies ("AJB and Moore", "sit Cooper vs the Ravens"). The RSS feed is
flat (a reply doesn't show what it answers), so the LLM reads each comment on its own and returns
only the recommendations it states. Questions and comments that don't recommend anyone yield
nothing.

Division of labor, as with news: the LLM returns the comment id, the player's name and team, the
verdict and a short restated reason. Code matches the player to our players table (by full name,
else by a one-word name unique among the week's ranked players; unmatched names are dropped),
attaches the comment's link and date, and caches every processed comment in `reddit_extractions`
so none is sent twice.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from startsit.db import CrowdVoteRow, Database, PlayerRow, RedditCommentRow, RedditExtractionRow, utc_iso
from startsit.evidence import current_week
from startsit.llm import BudgetedLLM, LLMError
from startsit.names import PlayerMatcher

log = logging.getLogger("startsit.ingest.reddit_extract")

VERDICTS = ("start", "sit")
CHUNK_SIZE = 100  # comments per LLM call; they're short (a fetch reads up to 5 threads x 100)
LOOKBACK = timedelta(days=4)  # older comments are about last week's games
MAX_COMMENTS_PER_RUN = 500
SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}
REASON_MAX_CHARS = 120

SYSTEM_PROMPT = """You read comments from r/fantasyfootball's "Who Do I Start" threads and extract the
start/sit recommendations each comment makes.
Rules:
- Use only each comment's own text. Never add outside knowledge or guess what a comment replies to.
- Extract a vote only when the comment clearly recommends starting or sitting a specific QB, RB, WR
  or TE: e.g. "AJB and Moore" or "start Coker" (start), "sit Cooper, bad matchup" (sit). A reply
  that is just one or more player names recommends starting them.
- A question asking for advice ("Pick 2: AJB, Moore, Cooper?", "Who do I start, X or Y?") is not a
  vote, even if it lists players. The asker's own lean ("leaning X") is not a vote either.
- Skip jokes, insults, comments about leagues or trades, and kickers and defenses.
- Use the player's full name if you're sure who a nickname or last name means in the comment (e.g.
  "AJB" -> "A.J. Brown"); otherwise use the name as written. Add the team abbreviation only if the
  comment gives it or names the opponent unambiguously; else an empty string.
- reason: the comment's stated reason in at most 12 words (e.g. "better matchup vs Miami"), or an
  empty string if it gives none. Never invent a reason."""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "votes": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "comment_id": {"type": "STRING"},
                    "player": {"type": "STRING"},
                    "team": {"type": "STRING"},
                    "verdict": {"type": "STRING", "enum": list(VERDICTS)},
                    "reason": {"type": "STRING"},
                },
                "required": ["comment_id", "player", "team", "verdict", "reason"],
            },
        }
    },
    "required": ["votes"],
}


@dataclass(frozen=True)
class ExtractionResult:
    comments_processed: int
    votes: int
    calls: int
    stopped_early: str | None = None


def build_prompt(comments: Sequence[RedditCommentRow]) -> str:
    lines = ["Comments:"]
    lines.extend(f"[C{i}] {c['text']}" for i, c in enumerate(comments, 1))
    return "\n".join(lines)


def ranked_players(db: Database, players: Sequence[PlayerRow]) -> list[PlayerRow]:
    """The QBs, RBs, WRs and TEs in the current week's rankings: the pool for one-word names."""
    season = db.latest_season()
    if season is None:
        return []
    ranked = {r["player_id"] for r in db.rankings_for_week(season, current_week(db, season))}
    return [p for p in players if p["player_id"] in ranked and p["position"] in SKILL_POSITIONS]


def votes_from_response(
    response: Any,
    comments: Sequence[RedditCommentRow],
    matcher: PlayerMatcher,
    ranked: PlayerMatcher | None = None,
) -> list[CrowdVoteRow]:
    """Validate the LLM output against the chunk and attach code-owned fields (player id, link, date).

    Players are matched by full name, else (commenters often write "Egbuka" or "Puka") by a
    one-word name that only one of the `ranked` players has.
    """
    by_id = {f"C{i}": c for i, c in enumerate(comments, 1)}
    votes: dict[tuple[str, str], CrowdVoteRow] = {}  # by (comment_id, player_id): one vote per player per comment
    raw_votes = response.get("votes") if isinstance(response, dict) else None
    for raw in raw_votes if isinstance(raw_votes, list) else []:
        if not isinstance(raw, dict):
            continue
        comment = by_id.get(str(raw.get("comment_id", "")))
        verdict = raw.get("verdict")
        if comment is None or verdict not in VERDICTS:
            continue
        name, team = str(raw.get("player", "")), str(raw.get("team", ""))
        player_id = matcher.match(name, team) or (ranked.match_one_word(name, team) if ranked else None)
        if player_id is None:
            log.info("crowd vote for unmatched player %r (%s) skipped", raw.get("player"), comment["comment_id"])
            continue
        reason = " ".join(str(raw.get("reason", "")).split())[:REASON_MAX_CHARS]
        votes[(comment["comment_id"], player_id)] = CrowdVoteRow(
            comment_id=comment["comment_id"],
            player_id=player_id,
            verdict=str(verdict),
            reason=reason or None,
            published_at=comment["published_at"],
            url=comment["url"],
        )
    return list(votes.values())


def extract_votes(db: Database, llm: BudgetedLLM, now: datetime | None = None) -> ExtractionResult:
    """Process unextracted recent comments in chunks until done, out of budget, or the LLM fails."""
    now = now or datetime.now(UTC)
    pending = db.unextracted_reddit(utc_iso(now - LOOKBACK), MAX_COMMENTS_PER_RUN)
    if not pending:
        return ExtractionResult(0, 0, 0)
    players = db.all_players()
    matcher, ranked = PlayerMatcher(players), PlayerMatcher(ranked_players(db, players))
    processed = vote_count = calls = 0
    stopped: str | None = None
    for start in range(0, len(pending), CHUNK_SIZE):
        if llm.remaining(now) <= 0:
            stopped = f"daily {llm.purpose} budget of {llm.daily_budget} LLM calls is spent"
            break
        chunk = pending[start : start + CHUNK_SIZE]
        calls += 1
        try:
            response = llm.generate_json(build_prompt(chunk), RESPONSE_SCHEMA, SYSTEM_PROMPT)
        except LLMError as e:
            stopped = str(e)  # leave this chunk unextracted; the next run retries it
            break
        votes = votes_from_response(response, chunk, matcher, ranked)
        per_comment: dict[str, int] = defaultdict(int)
        for v in votes:
            per_comment[v["comment_id"]] += 1
        db.upsert("crowd_votes", votes)
        db.upsert(
            "reddit_extractions",
            [
                RedditExtractionRow(
                    comment_id=c["comment_id"],
                    model=llm.provider.model,
                    extracted_at=utc_iso(now),
                    vote_count=per_comment[c["comment_id"]],
                )
                for c in chunk
            ],
        )
        processed += len(chunk)
        vote_count += len(votes)
    if stopped:
        log.warning("reddit extraction stopped early: %s", stopped)
    return ExtractionResult(processed, vote_count, calls, stopped)
