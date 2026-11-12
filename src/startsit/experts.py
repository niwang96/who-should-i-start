"""How many FantasyPros experts rank each of two players ahead of the other (expert agreement).

FantasyPros' compare-players endpoint returns every expert's individual weekly rank for 2 to 4
players, in all three scoring formats at once (tested Nov 11 2026: about 50 experts each, for
players outside the top 10 and for mixed positions via the flex list). So for a two-player
question code can say "34 of 50 experts start A over B", a different signal from the average
rank: how much the experts agree, and per format (our rankings file is PPR-only).

The free key's daily quota is small (about 45-50 calls; the daily projections fetch uses about
24), so calls are capped at DAILY_CALLS a day (UTC), shared first come first served by questions
on the site and the weekly benchmark. A pair is fetched at most once a day and cached in
`expert_splits`; any failure (quota, throttle, timeout) just leaves the split out, or falls back to
an older one. `ensure` runs before the packet is built, never inside it, so packets stay pure
database reads.
"""

from __future__ import annotations

import json
import logging
import urllib.error
from datetime import datetime
from typing import Any

from startsit.db import Database, ExpertSplitRow, utc_iso
from startsit.ingest.projections import FantasyProsClient, FantasyProsStopped
from startsit.scoring import SCORINGS

log = logging.getLogger("startsit.experts")

DAILY_CALLS = 20
API_FORMATS = {"ppr": "PPR", "half": "HALF", "std": "STD"}
FLEX = {"RB", "WR", "TE"}


def pair_key(fantasypros_ids: list[str]) -> str:
    return ":".join(sorted(fantasypros_ids))


def api_position(positions: list[str]) -> str | None:
    """The ranking list both players are on: their position, the flex list for mixed RB/WR/TE, else None."""
    if len(set(positions)) == 1 and positions[0] in {"QB", *FLEX}:
        return positions[0]
    return "FLX" if set(positions) <= FLEX else None


def count_split(rankings: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Per format (ppr/half/std): how many experts rank each player ahead of the other, and how many
    ranked both ("experts"). Ties count for neither."""
    out: dict[str, dict[str, int]] = {}
    for fmt, api in API_FORMATS.items():
        by_player = rankings.get(api) or {}
        if len(by_player) != 2:
            continue
        (a, ranks_a), (b, ranks_b) = by_player.items()
        ra = {e["expert_id"]: int(e["rank"]) for e in ranks_a}
        rb = {e["expert_id"]: int(e["rank"]) for e in ranks_b}
        both = ra.keys() & rb.keys()
        if both:
            out[fmt] = {
                a: sum(ra[e] < rb[e] for e in both),
                b: sum(rb[e] < ra[e] for e in both),
                "experts": len(both),
            }
    return out


def ensure(
    db: Database, client: FantasyProsClient | None, season: int, week: int, player_ids: list[str], now: datetime
) -> bool:
    """Fetch the pair's expert split if it hasn't been fetched today and the daily cap allows;
    returns whether a new split was stored. Never raises for FantasyPros trouble."""
    if client is None or len(player_ids) != 2:
        return False
    rankings = [db.latest_ranking(pid, season, week) for pid in player_ids]
    players = [db.player(pid) for pid in player_ids]
    if any(r is None for r in rankings) or any(p is None for p in players):
        return False
    position = api_position([p["position"] for p in players if p])
    if position is None:
        return False
    fp_ids = [str(r["fantasypros_id"]) for r in rankings if r]
    key, today = pair_key(fp_ids), utc_iso(now)[:10]
    if (split := db.expert_split(season, week, key, utc_iso(now))) and split["fetched_day"] == today:
        return False
    if db.expert_splits_fetched(today) >= DAILY_CALLS:
        return False
    try:
        data = client.get("compare-players", {"players": key, "position": position, "week": week})
        counts = count_split(data.get("rankings") or {}) if isinstance(data, dict) else {}
    except (FantasyProsStopped, urllib.error.URLError, OSError, ValueError) as e:
        log.warning("expert split for %s not fetched: %s", key, e)
        return False
    if not counts:
        log.info("expert split for %s: no experts ranked both", key)
        return False
    db.upsert(
        "expert_splits",
        [
            ExpertSplitRow(
                season=season,
                week=week,
                pair=key,
                fetched_day=today,
                fetched_at=utc_iso(now),
                splits_json=json.dumps(counts),
            )
        ],
    )
    return True


def split_for(
    db: Database, season: int, week: int, fantasypros_ids: list[str], scoring: str, until: str
) -> dict[str, int] | None:
    """The stored split for the question's format, fetched at or before `until`: {fantasypros_id:
    experts ahead, "experts": n, "fetched_day"...}; None without one."""
    if scoring not in SCORINGS or len(fantasypros_ids) != 2:
        return None
    row = db.expert_split(season, week, pair_key(fantasypros_ids), until)
    if row is None:
        return None
    counts = json.loads(row["splits_json"]).get(scoring)
    return counts or None


def fetched_day(db: Database, season: int, week: int, fantasypros_ids: list[str], until: str) -> str | None:
    row = db.expert_split(season, week, pair_key(fantasypros_ids), until)
    return row["fetched_day"] if row else None
