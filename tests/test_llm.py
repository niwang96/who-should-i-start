import io
import json
import urllib.error
from datetime import UTC, datetime, timedelta

import pytest

from startsit import llm
from startsit.llm import (
    BudgetedLLM,
    FakeProvider,
    GeminiProvider,
    LLMError,
    QuotaExceeded,
    RateLimited,
    VisitorLimit,
    VisitorLimited,
    utc_day_start,
    visitor_id,
)


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


PER_DAY = b'{"error": {"code": 429, "details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}}'
PER_MINUTE = b'{"error": {"code": 429, "details": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel"}]}}'


@pytest.mark.parametrize(
    "code,body,error,attempts",
    [
        (429, PER_DAY, QuotaExceeded, 1),
        (429, PER_MINUTE, RateLimited, 1),
        (503, b"busy", LLMError, 2),
        (400, b"bad", LLMError, 1),
    ],
)
def test_gemini_maps_http_errors_and_retries_overload_once(monkeypatch, code, body, error, attempts):
    calls = []

    def fail(req, timeout):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, code, "err", {}, io.BytesIO(body))

    monkeypatch.setattr(llm.urllib.request, "urlopen", fail)
    monkeypatch.setattr(GeminiProvider, "RETRY_DELAY_SECONDS", 0)
    with pytest.raises(error) as raised:
        GeminiProvider(api_key="k").generate_json("p", {})
    assert len(calls) == attempts
    assert error is LLMError or type(raised.value) is error


def test_gemini_recovers_when_the_retry_succeeds(monkeypatch):
    responses = [
        urllib.error.HTTPError("u", 503, "busy", {}, io.BytesIO(b"busy")),
        FakeResponse(gemini_body({"ok": 1})),
    ]

    def flaky(req, timeout):
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(llm.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(GeminiProvider, "RETRY_DELAY_SECONDS", 0)
    assert GeminiProvider(api_key="k").generate_json("p", {}) == {"ok": 1}


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
    assert utc_day_start(datetime(2026, 10, 24, 23, 59, tzinfo=UTC)) == datetime(2026, 10, 24, tzinfo=UTC)


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


def test_the_per_minute_cap_is_shared_across_purposes(db):
    clock = Clock(datetime(2026, 10, 25, 12, 0, 0, tzinfo=UTC))
    news = BudgetedLLM(FakeProvider(respond=lambda p: {}), db, "news", 100, per_minute=2, clock=clock)
    decide = BudgetedLLM(FakeProvider(respond=lambda p: {}), db, "decide", 100, per_minute=2, clock=clock)
    news.generate_json("p", {})
    decide.generate_json("p", {})
    with pytest.raises(RateLimited, match="2 LLM calls per minute"):
        decide.generate_json("p", {})
    clock.now += timedelta(seconds=61)
    assert decide.generate_json("p", {}) == {}  # the window moved on


def test_visitor_limits_count_only_that_visitors_calls(db):
    clock = Clock(datetime(2026, 10, 25, 12, 0, 0, tzinfo=UTC))
    limits = [VisitorLimit(2, timedelta(minutes=10), "10 minutes"), VisitorLimit(3, timedelta(days=1), "day")]

    def llm(visitor):
        return BudgetedLLM(FakeProvider(respond=lambda p: {}), db, "decide", 100, visitor, limits, clock=clock)

    llm("a").generate_json("p", {})
    llm("a").generate_json("p", {})
    with pytest.raises(VisitorLimited, match="2 new AI answers per 10 minutes"):
        llm("a").generate_json("p", {})
    llm("b").generate_json("p", {})  # another visitor is unaffected
    clock.now += timedelta(minutes=11)
    llm("a").generate_json("p", {})
    with pytest.raises(VisitorLimited, match="3 new AI answers per day"):
        llm("a").generate_json("p", {})
    BudgetedLLM(FakeProvider(respond=lambda p: {}), db, "news", 100, clock=clock).generate_json("p", {})  # no visitor


def test_visitor_id_hashes_the_address_with_a_salt():
    assert visitor_id("1.2.3.4", "s") == visitor_id("1.2.3.4", "s") != visitor_id("1.2.3.4", "t")
    assert "1.2.3.4" not in visitor_id("1.2.3.4", "s") and len(visitor_id("1.2.3.4", "s")) == 16


@pytest.mark.parametrize(
    "failure",
    [ConnectionResetError("reset by peer"), None],  # None: a body that isn't JSON
)
def test_gemini_turns_any_transport_or_decoding_failure_into_an_llm_error(monkeypatch, failure):
    def urlopen(req, timeout):
        if failure is not None:
            raise failure
        return FakeResponse(b"<html>gateway</html>")

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    with pytest.raises(llm.LLMError, match="Gemini request failed"):
        llm.GeminiProvider(api_key="k").generate_json("p", {})


@pytest.mark.parametrize(
    "address,network",
    [
        ("203.0.113.7", "203.0.113.7"),
        ("2001:db8:aaaa:bbbb:1:2:3:4", "2001:db8:aaaa:bbbb::/64"),
        ("2001:db8:aaaa:bbbb::ffff", "2001:db8:aaaa:bbbb::/64"),  # same /64: same visitor
        ("unknown", "unknown"),
    ],
)
def test_visitors_are_counted_per_ipv4_address_and_per_ipv6_64(address, network):
    assert llm.visitor_network(address) == network


def test_one_ipv6_network_is_one_visitor():
    assert llm.visitor_id("2001:db8:aaaa:bbbb::1", "s") == llm.visitor_id("2001:db8:aaaa:bbbb::2", "s")
    assert llm.visitor_id("2001:db8:aaaa:bbbb::1", "s") != llm.visitor_id("2001:db8:aaaa:cccc::1", "s")


@pytest.mark.parametrize("failure", [TimeoutError("read timed out"), urllib.error.URLError(TimeoutError("timed out"))])
def test_a_slow_gemini_answer_is_a_timeout(monkeypatch, failure):
    seen = []

    def urlopen(req, timeout):
        seen.append(timeout)
        raise failure

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    with pytest.raises(llm.TimedOut):
        GeminiProvider(api_key="k").generate_json("p", {}, timeout=7)
    assert 6 < seen[0] <= 7  # the call's own limit, not the provider's 60 s


def test_no_retry_after_an_overload_when_the_deadline_leaves_no_room(monkeypatch):
    calls = []

    def fail(req, timeout):
        calls.append(timeout)
        raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, io.BytesIO(b"busy"))

    monkeypatch.setattr(llm.urllib.request, "urlopen", fail)
    with pytest.raises(LLMError, match="503"):
        GeminiProvider(api_key="k").generate_json("p", {}, timeout=3)  # less than 2 x the 2 s retry delay
    assert len(calls) == 1
    with pytest.raises(llm.TimedOut):
        GeminiProvider(api_key="k").generate_json("p", {}, timeout=-1)  # already out of time: no call
    assert len(calls) == 1
