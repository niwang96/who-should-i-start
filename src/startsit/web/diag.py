"""A diagnostics page served only on Vercel preview deployments (which Vercel puts behind the owner's login).

/diag/limits shows which visitor this request counts as (from the address Vercel forwards) and how
much of each AI limit is used. It makes no AI call, and raw IP addresses are never shown in full.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

from startsit.db import Database
from startsit.decide import VISITOR_LIMITS
from startsit.llm import DAILY_BUDGETS, REQUESTS_PER_MINUTE, utc_day_start, visitor_id

ADDRESS_HEADERS = ("x-forwarded-for", "x-real-ip", "x-vercel-forwarded-for")


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
            "limit": DAILY_BUDGETS["decide"],
            "used": db.llm_calls_since(utc_day_start(now), "decide"),
        },
        "all_calls_last_minute": {
            "limit": REQUESTS_PER_MINUTE,
            "used": db.llm_calls_since(now - timedelta(minutes=1)),
        },
    }
