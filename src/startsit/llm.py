"""LLM providers behind one small interface, so the model can be swapped and tests never call a real LLM.

`GeminiProvider` calls the Gemini REST API directly (no SDK) and asks for JSON that
matches a schema. `FakeProvider` returns canned responses for tests. Every real call is
logged to the database (`llm_calls`) by `BudgetedLLM`, which also enforces a daily budget
per purpose so one feature can't spend the whole free-tier allowance.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from startsit.db import Database

DEFAULT_MODEL = "gemini-3.5-flash-lite"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class LLMError(RuntimeError):
    pass


class QuotaExceeded(LLMError):
    """The provider's rate limit or our own daily budget was hit; try again later."""


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
            detail = e.read()[:300].decode(errors="replace")
            if e.code in self.RETRY_STATUSES and retries > 0:
                time.sleep(self.RETRY_DELAY_SECONDS)
                return self._post(req, retries - 1)
            if e.code == 429:
                raise QuotaExceeded(f"Gemini rate limit: {detail}") from e
            raise LLMError(f"Gemini HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError) as e:
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


def utc_day_start(now: datetime | None = None) -> str:
    """Start of the current UTC day, in the format llm_calls.called_at uses."""
    now = now or datetime.now(UTC)
    return now.strftime("%Y-%m-%d 00:00:00")


@dataclass
class BudgetedLLM:
    """Wraps a provider: refuses calls past `daily_budget` for `purpose`, and logs every call."""

    provider: LLMProvider
    db: Database
    purpose: str
    daily_budget: int

    def remaining(self, now: datetime | None = None) -> int:
        """Calls left today for this purpose."""
        return self.daily_budget - self.db.llm_calls_since(utc_day_start(now), self.purpose)

    def generate_json(self, prompt: str, schema: dict[str, Any], system: str | None = None) -> Any:
        if self.remaining() <= 0:
            raise QuotaExceeded(f"daily {self.purpose} budget of {self.daily_budget} LLM calls is spent")
        try:
            result = self.provider.generate_json(prompt, schema, system)
        except LLMError:
            self.db.record_llm_call(self.purpose, self.provider.model, ok=False)
            raise
        self.db.record_llm_call(self.purpose, self.provider.model, ok=True)
        return result
