from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from test_cli import seed

from startsit.db import Database
from startsit.llm import visitor_id
from startsit.web import app as web
from startsit.web import diag

NOW = datetime(2026, 10, 24, 18, 0, tzinfo=UTC)


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
    assert TestClient(web.create_app(path)).get("/diag/limits").status_code == 404

    monkeypatch.setenv("VERCEL_ENV", "preview")
    client = TestClient(web.create_app(path, clock=lambda: NOW))
    r = client.get("/diag/limits", headers={"x-forwarded-for": "198.51.100.4"})
    assert r.json()["address"] == "198.51.x.x" and r.json()["visitor"] == visitor_id("198.51.100.4")
