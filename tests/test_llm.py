import io
import json
import urllib.error
from datetime import UTC, datetime

import pytest

from startsit import llm
from startsit.llm import BudgetedLLM, FakeProvider, GeminiProvider, LLMError, QuotaExceeded, utc_day_start


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def gemini_body(payload):
    return json.dumps({"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}).encode()


def test_gemini_sends_schema_and_parses_json(monkeypatch):
    sent = {}

    def fake_urlopen(req, timeout):
        sent["url"], sent["headers"], sent["body"] = req.full_url, dict(req.headers), json.loads(req.data)
        return FakeResponse(gemini_body({"facts": []}))

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    provider = GeminiProvider(api_key="test-key", model="gemini-3.5-flash-lite")
    assert provider.generate_json("prompt", {"type": "OBJECT"}, system="rules") == {"facts": []}
    assert sent["url"].endswith("/models/gemini-3.5-flash-lite:generateContent")
    assert sent["headers"]["X-goog-api-key"] == "test-key"
    assert sent["body"]["generationConfig"]["responseSchema"] == {"type": "OBJECT"}
    assert sent["body"]["systemInstruction"] == {"parts": [{"text": "rules"}]}


@pytest.mark.parametrize("code,error", [(429, QuotaExceeded), (500, LLMError)])
def test_gemini_maps_http_errors(monkeypatch, code, error):
    def fail(req, timeout):
        raise urllib.error.HTTPError(req.full_url, code, "err", {}, io.BytesIO(b'{"error": "nope"}'))

    monkeypatch.setattr(llm.urllib.request, "urlopen", fail)
    with pytest.raises(error):
        GeminiProvider(api_key="k").generate_json("p", {})


def test_gemini_rejects_a_malformed_response(monkeypatch):
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda req, timeout: FakeResponse(b'{"candidates": []}'))
    with pytest.raises(LLMError, match="unexpected Gemini response"):
        GeminiProvider(api_key="k").generate_json("p", {})


def test_gemini_requires_a_key_and_reads_the_model_from_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        GeminiProvider()
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "other-model")
    assert GeminiProvider().model == "other-model"
    monkeypatch.delenv("GEMINI_MODEL")
    assert GeminiProvider().model == llm.DEFAULT_MODEL == "gemini-3.5-flash-lite"


def test_budgeted_llm_logs_calls_and_enforces_the_daily_budget(db):
    wrapped = BudgetedLLM(FakeProvider(responses=[{"a": 1}, LLMError("x")]), db, "news", daily_budget=2)
    assert wrapped.generate_json("p", {}) == {"a": 1}
    with pytest.raises(LLMError):
        wrapped.generate_json("p", {})
    assert wrapped.remaining() == 0  # failed calls count too: they still used a request
    with pytest.raises(QuotaExceeded, match="budget of 2"):
        wrapped.generate_json("p", {})
    assert db.query("select purpose, model, ok from llm_calls") == [
        {"purpose": "news", "model": "fake-model", "ok": 1},
        {"purpose": "news", "model": "fake-model", "ok": 0},
    ]
    assert BudgetedLLM(FakeProvider(), db, "decide", daily_budget=5).remaining() == 5  # separate purpose


def test_utc_day_start():
    assert utc_day_start(datetime(2026, 10, 24, 23, 59, tzinfo=UTC)) == "2026-10-24 00:00:00"
