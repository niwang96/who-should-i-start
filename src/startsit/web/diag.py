"""Diagnostics served only on Vercel preview deployments (which Vercel puts behind the owner's login).

- /diag/reddit: can Vercel's servers read Reddit's public .json endpoints? (step 6 depends on it)
- /diag/limits: which visitor this request counts as, and how much of each AI limit is used.

Neither makes an AI call. Raw IP addresses are never shown in full.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from typing import Any

from startsit.db import Database
from startsit.decide import DECIDE_DAILY_LLM_BUDGET, VISITOR_LIMITS
from startsit.http import get_bytes
from startsit.llm import REQUESTS_PER_MINUTE, utc_day_start, visitor_id

REDDIT_PROBES = (
    "https://www.reddit.com/r/fantasyfootball/new.json?limit=5",
    "https://www.reddit.com/r/fantasyfootball/top.json?t=week&limit=5",
    "https://old.reddit.com/r/fantasyfootball/new.json?limit=5",
)
ADDRESS_HEADERS = ("x-forwarded-for", "x-real-ip", "x-vercel-forwarded-for")


def probe(url: str, fetch: Callable[[str], bytes]) -> dict[str, Any]:
    """Fetch one Reddit listing (or comment thread) and summarize the response; never raises."""
    started = time.monotonic()
    result: dict[str, Any] = {"url": url}
    try:
        body = fetch(url)
        result |= {"status": 200, "bytes": len(body)}
        data = json.loads(body)
        listing = data[0] if isinstance(data, list) else data  # a comment thread is [post, comments]
        posts = [child["data"] for child in listing["data"]["children"]]
        result |= {"posts": len(posts), "first_title": posts[0].get("title") if posts else None}
        if isinstance(data, list):
            result["comments"] = len(data[1]["data"]["children"])
        elif posts:
            result["permalink"] = posts[0].get("permalink")
    except urllib.error.HTTPError as e:
        result |= {"status": e.code, "error": e.reason, "body_start": e.read()[:200].decode("utf-8", "replace")}
    except Exception as e:  # report everything: this page exists to see what goes wrong
        result["error"] = f"{type(e).__name__}: {e}"
    result["ms"] = round((time.monotonic() - started) * 1000)
    return result


def reddit_report(fetch: Callable[[str], bytes] | None = None) -> dict[str, Any]:
    fetch = fetch or get_bytes
    results = []
    for url in REDDIT_PROBES:
        results.append(probe(url, fetch))
        time.sleep(1)  # be polite
    if permalink := results[0].get("permalink"):
        results.append(probe(f"https://www.reddit.com{permalink.rstrip('/')}.json?limit=10", fetch))
    return {"ok": all(r.get("status") == 200 and "error" not in r for r in results), "probes": results}


def mask(address: str) -> str:
    """Enough of an IP to tell two networks apart, not enough to identify anyone."""
    if ":" in address:
        return ":".join(address.split(":")[:2]) + ":…"
    parts = address.split(".")
    return ".".join([*parts[:2], "x", "x"]) if len(parts) == 4 else "?"


def limits_report(db: Database, headers: Mapping[str, str], address: str, now: datetime) -> dict[str, Any]:
    visitor = visitor_id(address)
    return {
        "address": mask(address),
        "address_headers": {
            h: [mask(a.strip()) for a in headers[h].split(",")] for h in ADDRESS_HEADERS if h in headers
        },
        "visitor": visitor,
        "visitor_salt_set": bool(os.environ.get("VISITOR_SALT")),
        "visitor_limits": [
            {
                "window": limit.label,
                "limit": limit.calls,
                "used": db.llm_calls_since(now - limit.window, visitor=visitor),
            }
            for limit in VISITOR_LIMITS
        ],
        "decide_today": {
            "limit": DECIDE_DAILY_LLM_BUDGET,
            "used": db.llm_calls_since(utc_day_start(now), "decide"),
        },
        "all_calls_last_minute": {
            "limit": REQUESTS_PER_MINUTE,
            "used": db.llm_calls_since(now - timedelta(minutes=1)),
        },
    }
