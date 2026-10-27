import io
import json
import urllib.error
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from test_cli import seed

from startsit.db import Database
from startsit.llm import visitor_id
from startsit.web import app as web
from startsit.web import diag

NOW = datetime(2026, 10, 24, 18, 0, tzinfo=UTC)


def listing(*titles):
    children = [
        {"data": {"title": t, "permalink": f"/r/fantasyfootball/comments/{i}/x/"}} for i, t in enumerate(titles)
    ]
    return json.dumps({"data": {"children": children}}).encode()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(diag.time, "sleep", lambda s: None)


def test_reddit_report_summarizes_each_probe_and_reads_a_thread():
    def fetch(url):
        if "/comments/" in url:
            return json.dumps([json.loads(listing("Post")), {"data": {"children": [{}, {}]}}]).encode()
        return listing("WDIS Week 8", "Injury thread")

    report = diag.reddit_report(fetch)
    assert report["ok"] is True and len(report["probes"]) == 4
    assert report["probes"][0] | {"ms": 0} == {
        "url": diag.REDDIT_PROBES[0],
        "status": 200,
        "bytes": len(listing("WDIS Week 8", "Injury thread")),
        "posts": 2,
        "first_title": "WDIS Week 8",
        "permalink": "/r/fantasyfootball/comments/0/x/",
        "ms": 0,
    }
    assert report["probes"][3]["url"] == "https://www.reddit.com/r/fantasyfootball/comments/0/x.json?limit=10"
    assert report["probes"][3]["comments"] == 2


def test_reddit_report_reports_blocks_and_garbage_without_raising():
    def fetch(url):
        if "old." in url:
            return b"<html>not json"
        raise urllib.error.HTTPError(url, 403, "Blocked", {}, io.BytesIO(b"whoa there, pardner"))  # type: ignore[arg-type]

    report = diag.reddit_report(fetch)
    assert report["ok"] is False and len(report["probes"]) == 3  # no permalink, so no thread probe
    blocked, _, garbage = report["probes"]
    assert (blocked["status"], blocked["error"], blocked["body_start"]) == (403, "Blocked", "whoa there, pardner")
    assert garbage["error"].startswith("JSONDecodeError")


@pytest.mark.parametrize(
    "address,masked",
    [("203.0.113.7", "203.0.x.x"), ("2001:db8:1:2::7", "2001:db8:…"), ("unknown", "?")],
)
def test_mask(address, masked):
    assert diag.mask(address) == masked


def test_limits_report_counts_like_the_limits_do(tmp_path):
    visitor = visitor_id("203.0.113.7")
    with Database(str(tmp_path / "d.db")) as db:
        db.record_llm_call("decide", "m", True, NOW - timedelta(seconds=30), visitor)
        db.record_llm_call("decide", "m", True, NOW - timedelta(hours=2), visitor)
        db.record_llm_call("decide", "m", True, NOW - timedelta(minutes=5), "someone-else")
        db.record_llm_call("news", "m", True, NOW - timedelta(seconds=10))
        report = diag.limits_report(db, {"x-forwarded-for": "203.0.113.7, 10.0.0.1"}, "203.0.113.7", NOW)
    assert report["address"] == "203.0.x.x" and report["visitor"] == visitor
    assert report["address_headers"] == {"x-forwarded-for": ["203.0.x.x", "10.0.x.x"]}
    assert [(v["window"], v["used"]) for v in report["visitor_limits"]] == [("10 minutes", 1), ("day", 2)]
    assert report["decide_today"]["used"] == 3
    assert report["all_calls_last_minute"] == {"limit": 12, "used": 2}


def test_diagnostics_only_on_preview_deployments(tmp_path, monkeypatch):
    path = seed(tmp_path / "web.db")
    monkeypatch.delenv("VERCEL_ENV", raising=False)
    assert TestClient(web.create_app(path)).get("/diag/limits").status_code == 404
    monkeypatch.setenv("VERCEL_ENV", "production")
    assert TestClient(web.create_app(path)).get("/diag/reddit").status_code == 404

    monkeypatch.setenv("VERCEL_ENV", "preview")
    monkeypatch.setattr(diag, "get_bytes", lambda url: listing())
    client = TestClient(web.create_app(path, clock=lambda: NOW))
    assert client.get("/diag/reddit").json()["probes"][0]["posts"] == 0
    r = client.get("/diag/limits", headers={"x-forwarded-for": "198.51.100.4"})
    assert r.json()["address"] == "198.51.x.x" and r.json()["visitor"] == visitor_id("198.51.100.4")
