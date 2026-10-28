"""LLM providers behind one small interface, so the model can be swapped and tests never call a real LLM.

`GeminiProvider` calls the Gemini REST API directly (no SDK) and asks for JSON that
matches a schema. `FakeProvider` returns canned responses for tests. Every real call is
logged to the database (`llm_calls`) by `BudgetedLLM`, which enforces the free-tier limits
before calling, across every process that shares the database (ingest jobs, web requests):

- a daily budget per purpose, so one feature can't spend the whole daily allowance;
- a shared per-minute cap below Gemini's 15 requests/minute;
- optional per-visitor limits, so one visitor can't spend everyone's allowance.

The checks count logged calls and then call, so concurrent requests can overshoot a limit
slightly; the caps sit below Gemini's real limits to leave room for that.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from startsit.db import Database, utc_now

log = logging.getLogger("startsit.llm")

DEFAULT_MODEL = "gemini-3.5-flash-lite"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# Gemini's free tier allows 15 requests/minute; leave headroom for requests checked at the same moment.
REQUESTS_PER_MINUTE = 12
# Gemini's free tier allows 500 requests/day for the model, shared by every purpose. Counting
# calls before making them can overshoot slightly under concurrency, hence the headroom.
GEMINI_DAILY_REQUESTS = 500
DAILY_BUDGETS: dict[str, int] = {
    "decide": 370,  # start/sit answers on the site (cached answers are free)
    "news": 55,  # news extraction; the hourly and game-day runs use ~40
    "benchmark": 45,  # the weekly 30 benchmark matchups, plus retries
    "reddit": 25,  # crowd votes; at most 5 calls per fetch
}


class LLMError(RuntimeError):
    pass


class QuotaExceeded(LLMError):
    """A daily limit was hit (the provider's, or our budget for this purpose); try again tomorrow."""


class RateLimited(LLMError):
    """The per-minute limit was hit (the provider's, or our shared cap); try again in a minute."""


class VisitorLimited(LLMError):
    """This visitor has used their allowance of AI calls for now."""


class LLMProvider(Protocol):
    model: str

    def generate_json(self, prompt: str, schema: dict[str, Any], system: str | None = None) -> Any:
        """Return the parsed JSON response, which must match `schema` (an OpenAPI-style schema)."""
        ...


class GeminiProvider:
    # Gemini answers "model overloaded" (503) during demand spikes; one short retry usually works.
    RETRY_STATUSES = (500, 502, 503, 504)
    RETRY_DELAY_SECONDS = 2.0

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: float = 60):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or ""
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
        self.timeout = timeout

    def _post(self, req: urllib.request.Request, retries: int) -> Any:
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            # The error body can be long; keep a short excerpt. It never contains the key.
            body = e.read().decode(errors="replace")
            detail = body[:300]
            if e.code in self.RETRY_STATUSES and retries > 0:
                time.sleep(self.RETRY_DELAY_SECONDS)
                return self._post(req, retries - 1)
            if e.code == 429:
                # The body names the quota that ran out, e.g. "...RequestsPerDayPerProjectPerModel...".
                if "PerDay" in body:
                    raise QuotaExceeded(f"Gemini daily limit: {detail}") from e
                raise RateLimited(f"Gemini rate limit: {detail}") from e
            raise LLMError(f"Gemini HTTP {e.code}: {detail}") from e
        except (OSError, ValueError) as e:
            # URLError, timeouts and dropped connections are OSErrors; a body that isn't JSON is a
            # ValueError. Callers catch LLMError, so nothing else may escape.
            raise LLMError(f"Gemini request failed: {e}") from e

    def generate_json(self, prompt: str, schema: dict[str, Any], system: str | None = None) -> Any:
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema, "temperature": 0},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        req = urllib.request.Request(
            GEMINI_URL.format(model=self.model),
            data=json.dumps(body).encode(),
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
        )
        data = self._post(req, retries=1)
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise LLMError(f"unexpected Gemini response: {str(data)[:300]}") from e


@dataclass
class FakeProvider:
    """Test double: returns `responses` in order (or calls `respond(prompt)`) and records prompts."""

    responses: list[Any] = field(default_factory=list)
    respond: Callable[[str], Any] | None = None
    model: str = "fake-model"
    prompts: list[str] = field(default_factory=list)

    def generate_json(self, prompt: str, schema: dict[str, Any], system: str | None = None) -> Any:
        self.prompts.append(prompt)
        if self.respond is not None:
            return self.respond(prompt)
        if not self.responses:
            raise LLMError("FakeProvider has no responses left")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def utc_day_start(now: datetime | None = None) -> datetime:
    """Midnight UTC at the start of `now`'s day (Gemini's daily quota resets daily; ours resets at UTC midnight)."""
    now = (now or datetime.now(UTC)).astimezone(UTC)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


@dataclass(frozen=True)
class VisitorLimit:
    calls: int
    window: timedelta
    label: str  # how users see the window, e.g. "10 minutes"


def visitor_id(address: str, salt: str | None = None) -> str:
    """A pseudonymous visitor key: a salted hash of the IP address, so raw addresses are never stored.

    The salt comes from $VISITOR_SALT; keep it secret, or IPv4 hashes could be reversed by brute force.
    IPv6 addresses count per /64 network: one home or phone gets a whole /64 and can rotate through
    it freely, so per-address limits would be trivial to dodge.
    """
    salt = salt if salt is not None else os.environ.get("VISITOR_SALT", "")
    return hashlib.sha256(f"{salt}|{visitor_network(address)}".encode()).hexdigest()[:16]


def visitor_network(address: str) -> str:
    """The address a visitor is counted by: IPv4 as is, IPv6 as its /64 network (anything else as given)."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return address
    if ip.version == 6:
        return str(ipaddress.ip_network(f"{ip}/64", strict=False))
    return str(ip)


@dataclass
class BudgetedLLM:
    """Wraps a provider: checks every limit before a call (raising instead of calling), and logs every call.

    One instance per caller: the ingest job makes one per run, the web app one per request
    (with that request's `visitor`).
    """

    provider: LLMProvider
    db: Database
    purpose: str
    daily_budget: int
    visitor: str | None = None
    visitor_limits: Sequence[VisitorLimit] = ()
    per_minute: int = REQUESTS_PER_MINUTE
    clock: Callable[[], datetime] = field(default=utc_now, repr=False)

    def remaining(self, now: datetime | None = None) -> int:
        """Calls left today for this purpose."""
        return self.daily_budget - self.db.llm_calls_since(utc_day_start(now or self.clock()), self.purpose)

    def check(self) -> None:
        """Raise QuotaExceeded, VisitorLimited or RateLimited if a call now would break a limit."""
        now = self.clock()
        if self.remaining(now) <= 0:
            raise QuotaExceeded(f"daily {self.purpose} budget of {self.daily_budget} LLM calls is spent")
        if self.visitor is not None:
            for limit in self.visitor_limits:
                if self.db.llm_calls_since(now - limit.window, visitor=self.visitor) >= limit.calls:
                    raise VisitorLimited(f"limit of {limit.calls} new AI answers per {limit.label}")
        if self.db.llm_calls_since(now - timedelta(minutes=1)) >= self.per_minute:
            raise RateLimited(f"shared limit of {self.per_minute} LLM calls per minute reached")

    def generate_json(self, prompt: str, schema: dict[str, Any], system: str | None = None) -> Any:
        self.check()
        try:
            result = self.provider.generate_json(prompt, schema, system)
        except LLMError:
            self._record(ok=False)  # a failed call still used a request
            raise
        self._record(ok=True)
        return result

    def _record(self, ok: bool) -> None:
        self.db.record_llm_call(self.purpose, self.provider.model, ok, self.clock(), self.visitor)


def budgeted_gemini(
    db: Database,
    purpose: str,
    unavailable: str,
    visitor: str | None = None,
    visitor_limits: Sequence[VisitorLimit] = (),
) -> BudgetedLLM | None:
    """A Gemini client held to `purpose`'s daily budget (DAILY_BUDGETS), or None (logging
    "<unavailable>: <why>") without a key."""
    try:
        provider = GeminiProvider()
    except LLMError as e:
        log.warning("%s: %s", unavailable, e)
        return None
    budget = DAILY_BUDGETS[purpose]
    return BudgetedLLM(provider, db, purpose, budget, visitor=visitor, visitor_limits=visitor_limits)
