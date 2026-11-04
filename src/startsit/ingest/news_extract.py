"""Turn news items into structured, player-level facts with one batched LLM call per chunk.

Division of labor, so the LLM never invents facts:
- The LLM reads only the given headline and summary, and returns, per player it names:
  the item id, the player's name and team, an event type, a severity and a short
  restatement of the item.
- Code assigns everything else: the source URL and date come from the item, and the
  player is matched to our players table by name (and team, when names collide).
  Names it can't match are dropped.

Every processed item gets a `news_extractions` row (the permanent L2 cache, keyed by the
item's id: the feed's guid, else its URL),
even when it yields no facts, so no item is ever sent twice.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from startsit.db import Database, NewsExtractionRow, NewsFactRow, NewsItemRow, PlayerRow, utc_iso
from startsit.llm import BudgetedLLM, LLMError
from startsit.names import PlayerMatcher

log = logging.getLogger("startsit.ingest.news_extract")

EVENTS = ("injury", "practice", "return", "role_change", "transaction", "suspension", "performance", "other")
SEVERITIES = ("high", "medium", "low")
NO_STATUS = "none"
GAME_STATUSES = ("out", "doubtful", "questionable", "active", NO_STATUS)
CHUNK_SIZE = 40  # items per LLM call; ~40 headlines is a few thousand tokens
LOOKBACK = timedelta(days=7)  # older items aren't worth a call
MAX_ITEMS_PER_RUN = 400

SYSTEM_PROMPT = """You extract fantasy-football facts from NFL news items.
Rules:
- Use only the text of each item. Never add outside knowledge.
- Only include QB, RB, WR or TE players named in the item, and only when the item reports a fact
  about their availability, health, role or a game already played. An item may produce no facts.
- Also include defensive players (DE, DT, LB, CB, S), but only when the item states their status
  for their next game (see game_status).
- Skip: opinions, quotes, predictions, betting or fantasy picks ("smash spot", "TD scorer picks",
  "expected to bounce back"), speculation about a future role ("may soon start", "could see more
  snaps", "might take over"), game previews and matchup hype, off-field news with no effect on
  playing, and team-level news, coaches and kickers.
- One fact per player per item. Use the player's full name as written and their team abbreviation
  if the item gives it (else an empty string).
- event: injury (hurt, status questionable/doubtful/out), practice (practice participation),
  return (cleared, activated, back from injury), role_change (named starter, benched, depth chart,
  snap or target share change), transaction (signed, released, traded, claimed), suspension,
  performance (a notable result in a game already played), other (any other playing-time fact).
- severity: impact on the player's fantasy value this week. high = likely to miss the game or a
  major role change; medium = uncertain status or meaningful change; low = minor.
- game_status: the player's status for their next game, only if the item states it: out (ruled
  out, inactive, placed on IR or suspended), doubtful, questionable, active (cleared, activated,
  no injury designation, will play); otherwise none. An in-game update ("won't return", "out for
  the rest of the game") is about the current game, not the next one: use none unless the item
  also states the next game's status.
- summary: at most 25 words, restating only what the item says."""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "facts": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "item_id": {"type": "STRING"},
                    "player": {"type": "STRING"},
                    "team": {"type": "STRING"},
                    "event": {"type": "STRING", "enum": list(EVENTS)},
                    "severity": {"type": "STRING", "enum": list(SEVERITIES)},
                    "summary": {"type": "STRING"},
                    "game_status": {"type": "STRING", "enum": list(GAME_STATUSES)},
                },
                "required": ["item_id", "player", "team", "event", "severity", "summary", "game_status"],
            },
        }
    },
    "required": ["facts"],
}


@dataclass(frozen=True)
class ExtractionResult:
    items_processed: int
    facts: int
    calls: int
    stopped_early: str | None = None  # why the run stopped before processing everything


def build_prompt(items: Sequence[NewsItemRow]) -> str:
    lines = ["News items:"]
    for i, item in enumerate(items, 1):
        summary = f" | {item['summary']}" if item["summary"] else ""
        lines.append(f"[N{i}] {item['title']}{summary}")
    return "\n".join(lines)


# Hedged wording: the model sometimes keeps speculation ("may soon be pressed to start") despite the prompt.
_SPECULATIVE = re.compile(r"\b(may|might|could)\b", re.IGNORECASE)


def is_noise(event: str, severity: str, summary: str, game_status: str | None) -> bool:
    """Facts the prompt asks the model to skip but it sometimes keeps; they're never about playing time."""
    if (event, severity) == ("other", "low"):
        return True  # quotes and color
    # Speculation about a role; a stated game status ("could return Sunday" + questionable) is kept.
    return event in ("role_change", "other") and game_status is None and bool(_SPECULATIVE.search(summary))


def facts_from_response(
    response: Any, items: Sequence[NewsItemRow], matcher: PlayerMatcher, defenders: PlayerMatcher | None = None
) -> list[NewsFactRow]:
    """Validate the LLM output against the chunk and attach code-owned fields (URL, date, player id).

    Names are matched to our QBs, RBs, WRs and TEs first, then to `defenders`. A defender's fact
    is kept only with a game status: that's all the evidence uses (an opponent's key defender out).
    """
    by_id = {f"N{i}": item for i, item in enumerate(items, 1)}
    facts: dict[tuple[str, str], NewsFactRow] = {}  # by (item_id, player_id)
    raw_facts = response.get("facts") if isinstance(response, dict) else None
    for raw in raw_facts if isinstance(raw_facts, list) else []:
        if not isinstance(raw, dict):
            continue
        item = by_id.get(str(raw.get("item_id", "")))
        event, severity = raw.get("event"), raw.get("severity")
        summary = str(raw.get("summary", "")).strip()
        status = raw.get("game_status")
        game_status = status if status in GAME_STATUSES and status != NO_STATUS else None
        if item is None or event not in EVENTS or severity not in SEVERITIES or not summary:
            continue
        if is_noise(event, severity, summary, game_status):
            continue
        name, team = str(raw.get("player", "")), str(raw.get("team", ""))
        player_id = matcher.match(name, team)
        if player_id is None and defenders is not None and game_status is not None:
            player_id = defenders.match(name, team)
        if player_id is None:
            log.info("news fact for unmatched player %r (%s) skipped", raw.get("player"), item["item_id"])
            continue
        facts[(item["item_id"], player_id)] = NewsFactRow(
            item_id=item["item_id"],
            player_id=player_id,
            url=item["url"],
            source=item["source"],
            published_at=item["published_at"],
            event=event,
            severity=severity,
            summary=summary,
            game_status=game_status,
        )
    return list(facts.values())


def extract_news(db: Database, llm: BudgetedLLM, now: datetime | None = None) -> ExtractionResult:
    """Process unextracted recent items in chunks until done, out of budget, or the LLM fails."""
    now = now or datetime.now(UTC)
    pending = db.unextracted_news(utc_iso(now - LOOKBACK), MAX_ITEMS_PER_RUN)
    matcher = PlayerMatcher(db.all_players())
    # Defender rows have the fields the matcher reads (player_id, name, team).
    defenders = PlayerMatcher(cast("list[PlayerRow]", db.all_defenders()))
    processed = fact_count = calls = 0
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
        facts = facts_from_response(response, chunk, matcher, defenders)
        per_item: dict[str, int] = defaultdict(int)
        for f in facts:
            per_item[f["item_id"]] += 1
        db.upsert("news_facts", facts)
        db.upsert(
            "news_extractions",
            [
                NewsExtractionRow(
                    item_id=i["item_id"],
                    model=llm.provider.model,
                    extracted_at=utc_iso(now),
                    fact_count=per_item[i["item_id"]],
                )
                for i in chunk
            ],
        )
        processed += len(chunk)
        fact_count += len(facts)
    if stopped:
        log.warning("news extraction stopped early: %s", stopped)
    return ExtractionResult(processed, fact_count, calls, stopped)
