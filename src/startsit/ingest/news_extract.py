"""Turn news items into structured, player-level facts with one batched LLM call per chunk.

Division of labor, so the LLM never invents facts:
- The LLM reads only the given headline and summary, and returns, per player it names:
  the item id, the player's name and team, an event type, a severity and a short
  restatement of the item.
- Code assigns everything else: the source URL and date come from the item, and the
  player is matched to our players table by name (and team, when names collide).
  Names it can't match are dropped.

Every processed item gets a `news_extractions` row (the permanent L2 cache keyed by URL),
even when it yields no facts, so no item is ever sent twice.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from startsit.db import Database, NewsExtractionRow, NewsFactRow, NewsItemRow, PlayerRow, utc_iso
from startsit.llm import BudgetedLLM, LLMError

log = logging.getLogger("startsit.ingest.news_extract")

EVENTS = ("injury", "practice", "return", "role_change", "transaction", "suspension", "performance", "other")
SEVERITIES = ("high", "medium", "low")
CHUNK_SIZE = 40  # items per LLM call; ~40 headlines is a few thousand tokens
LOOKBACK = timedelta(days=7)  # older items aren't worth a call
MAX_ITEMS_PER_RUN = 400

SYSTEM_PROMPT = """You extract fantasy-football facts from NFL news items.
Rules:
- Use only the text of each item. Never add outside knowledge.
- Only include QB, RB, WR or TE players named in the item, and only when the item says something
  about their availability, health, role or performance. Skip team-level news, coaches, defenders,
  kickers and items with no fantasy relevance; an item may produce no facts.
- One fact per player per item. Use the player's full name as written and their team abbreviation
  if the item gives it (else an empty string).
- event: injury (hurt, status questionable/doubtful/out), practice (practice participation),
  return (cleared, activated, back from injury), role_change (depth chart, starting, snaps, usage),
  transaction (signed, released, traded, claimed), suspension, performance (notable game), other.
- severity: impact on the player's fantasy value this week. high = likely to miss the game or a
  major role change; medium = uncertain status or meaningful change; low = minor or informational.
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
                },
                "required": ["item_id", "player", "team", "event", "severity", "summary"],
            },
        }
    },
    "required": ["facts"],
}

# Abbreviations news sources use that differ from nflverse's.
TEAM_ALIASES = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU"}
_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def normalize_name(name: str) -> str:
    """Lowercase, no punctuation or generational suffix: "Kenneth Walker III" -> "kenneth walker"."""
    name = re.sub(r"[^a-z ]", "", name.lower().replace("-", " "))
    return re.sub(r"\s+", " ", _SUFFIX.sub("", name)).strip()


class PlayerMatcher:
    """Matches an LLM-reported (name, team) to one player id, or None if missing or ambiguous."""

    def __init__(self, players: Sequence[PlayerRow]):
        self._by_name: dict[str, list[PlayerRow]] = defaultdict(list)
        for p in players:
            self._by_name[normalize_name(p["name"])].append(p)

    def match(self, name: str, team: str = "") -> str | None:
        candidates = self._by_name.get(normalize_name(name), [])
        if len(candidates) > 1 and team:
            team = TEAM_ALIASES.get(team.upper(), team.upper())
            candidates = [p for p in candidates if (p["team"] or "").upper() == team]
        return candidates[0]["player_id"] if len(candidates) == 1 else None


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


def facts_from_response(response: Any, items: Sequence[NewsItemRow], matcher: PlayerMatcher) -> list[NewsFactRow]:
    """Validate the LLM output against the chunk and attach code-owned fields (URL, date, player id)."""
    by_id = {f"N{i}": item for i, item in enumerate(items, 1)}
    facts: dict[tuple[str, str], NewsFactRow] = {}
    for raw in (response or {}).get("facts", []):
        item = by_id.get(str(raw.get("item_id", "")))
        event, severity = raw.get("event"), raw.get("severity")
        summary = str(raw.get("summary", "")).strip()
        if item is None or event not in EVENTS or severity not in SEVERITIES or not summary:
            continue
        player_id = matcher.match(str(raw.get("player", "")), str(raw.get("team", "")))
        if player_id is None:
            log.info("news fact for unmatched player %r (%s) skipped", raw.get("player"), item["url"])
            continue
        facts[(item["url"], player_id)] = NewsFactRow(
            url=item["url"],
            player_id=player_id,
            source=item["source"],
            published_at=item["published_at"],
            event=event,
            severity=severity,
            summary=summary,
        )
    return list(facts.values())


def extract_news(db: Database, llm: BudgetedLLM, now: datetime | None = None) -> ExtractionResult:
    """Process unextracted recent items in chunks until done, out of budget, or the LLM fails."""
    now = now or datetime.now(UTC)
    pending = db.unextracted_news(utc_iso(now - LOOKBACK), MAX_ITEMS_PER_RUN)
    matcher = PlayerMatcher(db.all_players())
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
        facts = facts_from_response(response, chunk, matcher)
        per_url: dict[str, int] = defaultdict(int)
        for f in facts:
            per_url[f["url"]] += 1
        db.upsert("news_facts", facts)
        db.upsert(
            "news_extractions",
            [
                NewsExtractionRow(
                    url=i["url"],
                    model=llm.provider.model,
                    extracted_at=utc_iso(now),
                    fact_count=per_url[i["url"]],
                )
                for i in chunk
            ],
        )
        processed += len(chunk)
        fact_count += len(facts)
    if stopped:
        log.warning("news extraction stopped early: %s", stopped)
    return ExtractionResult(processed, fact_count, calls, stopped)
