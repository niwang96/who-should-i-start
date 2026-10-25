"""FantasyPros projections for the players people are likely to ask about (free API key).

Used for one thing: converting the consensus projection (PPR only) to half-PPR and standard
with FantasyPros' own projected receptions, instead of our estimate from recent usage.

The free key returns at most 10 rows per call and has a daily quota (~75 calls), so each run
asks for the top players at each position by consensus rank, 10 per call (~26 calls), and
stops early, keeping what it has, if the quota runs out. Calls go in rank order across
positions (every position's top 10, then 11-20, ...), so a cut-off loses the least-asked-about
players. Rows are archived per fetch date.
Players FantasyPros expects to sit (e.g. ruled out) get no projection at all.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Any

from startsit.db import ProjectionRow, RankingRow
from startsit.http import get_bytes

log = logging.getLogger("startsit.ingest.projections")

BASE_URL = "https://api.fantasypros.com/public/v2/json/nfl"
CHUNK_SIZE = 10  # the free tier's row limit per call
# How deep to go at each position (by consensus rank): 4 + 8 + 10 + 4 = 26 calls.
POSITION_DEPTH = {"QB": 36, "RB": 72, "WR": 96, "TE": 36}


class FantasyProsQuotaExceeded(RuntimeError):
    pass


@dataclass
class FantasyProsClient:
    """Calls are spaced out: a burst of ~6 quick calls trips the API's per-second throttle (429)."""

    api_key: str
    timeout: float = 30
    min_interval: float = 2.0  # seconds between calls
    throttle_wait: float = 15.0  # seconds to wait before the one retry after a 429
    sleep: Callable[[float], None] = time.sleep
    _last_call: float = field(default=0.0, repr=False)

    @classmethod
    def from_env(cls) -> FantasyProsClient | None:
        key = os.environ.get("FANTASYPROS_API_KEY")
        return cls(key) if key else None

    def get(self, path: str, params: dict[str, Any]) -> Any:
        """GET with spacing; a 429 is retried once after a pause, then raised as FantasyProsQuotaExceeded."""
        url = f"{BASE_URL}/{path}?{urllib.parse.urlencode(params)}"
        for attempt in range(2):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                self.sleep(wait)
            self._last_call = time.monotonic()
            try:
                return json.loads(get_bytes(url, {"x-api-key": self.api_key}, self.timeout))
            except urllib.error.HTTPError as e:
                if e.code != 429:
                    raise
                if attempt == 0:
                    self.sleep(self.throttle_wait)
                    continue
                raise FantasyProsQuotaExceeded("FantasyPros rate limit or daily quota reached") from e
        raise AssertionError("unreachable")


def targets(rankings: Sequence[RankingRow]) -> dict[str, list[str]]:
    """FantasyPros ids to fetch, per position: the top players by consensus rank."""
    by_position: dict[str, list[RankingRow]] = {}
    for r in rankings:
        if r["position"] in POSITION_DEPTH and r["ecr"] is not None:
            by_position.setdefault(r["position"], []).append(r)
    return {
        pos: [r["fantasypros_id"] for r in sorted(rows, key=lambda r: r["ecr"] or 0)[: POSITION_DEPTH[pos]]]
        for pos, rows in by_position.items()
    }


def chunks_by_rank(wanted: dict[str, list[str]]) -> list[tuple[str, list[str]]]:
    """(position, up to 10 ids) per call, top-ranked chunks of every position first.

    If the quota runs out partway, what's lost is the lowest-ranked players of every
    position, not a whole position.
    """
    per_position = [
        [(pos, ids[i : i + CHUNK_SIZE]) for i in range(0, len(ids), CHUNK_SIZE)] for pos, ids in wanted.items()
    ]
    return [chunk for tier in zip_longest(*per_position) for chunk in tier if chunk is not None]


def fetch(client: FantasyProsClient, season: int, week: int, wanted: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Raw projection rows, 10 players per call; stops early (keeping what it has) when the quota runs out."""
    rows: list[dict[str, Any]] = []
    for position, chunk in chunks_by_rank(wanted):
        try:
            data = client.get(
                # Colon-separated: with commas the API silently ignores the filter.
                f"{season}/projections",
                {"position": position, "week": week, "players": ":".join(chunk)},
            )
        except FantasyProsQuotaExceeded as e:
            log.warning("%s; keeping %s projections fetched so far", e, len(rows))
            return rows
        rows.extend(data.get("players") or [])
    return rows


def build(
    raw: Sequence[dict[str, Any]], rankings: Sequence[RankingRow], season: int, week: int, fetched_date: str
) -> list[ProjectionRow]:
    """One row per player with per-format points and projected receptions, mapped to GSIS ids."""
    player_ids = {r["fantasypros_id"]: r["player_id"] for r in rankings}
    out: dict[str, ProjectionRow] = {}
    for p in raw:
        stats = p.get("stats") or {}
        fp_id = str(p.get("fpid", ""))
        if not fp_id or stats.get("points") is None:
            continue
        out[fp_id] = ProjectionRow(
            season=season,
            week=week,
            fetched_date=fetched_date,
            fantasypros_id=fp_id,
            player_id=player_ids.get(fp_id),
            points_ppr=_float(stats.get("points_ppr")),
            points_half=_float(stats.get("points_half")),
            points_std=_float(stats.get("points")),  # the API's default scoring is standard
            receptions=_float(stats.get("rec_rec")),
        )
    return list(out.values())


def _float(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
