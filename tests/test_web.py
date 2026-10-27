import json
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from test_cli import seed

from startsit import decide as decide_module
from startsit.db import Database
from startsit.llm import BudgetedLLM, FakeProvider, VisitorLimit
from startsit.web import app as web

ANSWER = {
    "pick": "P1",
    "confidence": "medium",
    "factors": [
        {"text": "Alpha Back projects higher.", "evidence_ids": ["E2"]},
        {"text": "His team is favored.", "evidence_ids": ["E3"]},
    ],
    "summary": "Alpha Back projects higher [E2] and his team is favored [E3].",
    "baseline_disagreement": "",
}


@pytest.fixture
def path(tmp_path):
    return seed(tmp_path / "web.db")


def client_for(path, provider=None, limits=()):
    def llm_factory(db, visitor):
        if provider is None:
            return None
        return BudgetedLLM(provider, db, "decide", 10, visitor=visitor, visitor_limits=limits)

    return TestClient(web.create_app(path, llm_factory))


def ask(client, *players, scoring="half", hx=False, ip="203.0.113.7"):
    headers = {"x-forwarded-for": f"{ip}, 10.0.0.1"} | ({"HX-Request": "true"} if hx else {})
    return client.post("/ask", data={"players": list(players), "scoring": scoring}, headers=headers)


def test_home_page_shows_the_week_and_data_freshness(path):
    with Database(path) as db:
        db.record_run("nflverse:player_weeks", 2026, 7, 1)
    r = client_for(path).get("/")
    assert r.status_code == 200
    assert "Answering for <strong>Week 7, 2026</strong>" in r.text
    assert "Rankings as of 2026-10-24" in r.text and "News checked never" in r.text and "Stats updated 20" in r.text
    assert "FantasyPros" in r.text  # attribution
    assert r.headers["content-security-policy"].startswith("default-src 'self'")
    assert client_for(path).get("/static/htmx-2.0.4.min.js").status_code == 200


def test_ask_returns_the_ai_pick_with_cited_evidence_and_logs_it(path):
    r = ask(client_for(path, FakeProvider(responses=[ANSWER])), "Alpha Back", "bravo back")
    assert r.status_code == 200 and "<html" in r.text  # no JavaScript: a full page
    assert "Start <strong>Alpha Back</strong>" in r.text
    assert "Medium confidence" in r.text and "AI pick" in r.text
    assert 'href="#E2"' in r.text and 'id="E2"' in r.text
    # The summary's [E2]-style citations become links; the explainer matches an AI pick.
    assert 'Alpha Back projects higher <a class="cite" href="#E2">E2</a>' in r.text
    assert "a clear lean" in r.text and "doubtful is never rated High" in r.text
    assert 'value="Alpha Back"' in r.text  # the form keeps what was asked
    with Database(path) as db:
        [row] = db.query("select * from recommendations")
    assert (row["pick"], row["source"], row["scoring"], row["week"]) == ("A", "llm", "half", 7)
    assert json.loads(row["player_ids"]) == ["A", "B"]
    assert "E2 [projection]" in row["packet_text"]
    assert json.loads(row["packet_json"])["players"][0]["items"][1]["id"] == "E2"
    assert row["visitor"] and "203.0.113.7" not in row["visitor"]  # hashed, never the raw IP


def test_htmx_requests_get_only_the_result_fragment(path):
    r = ask(client_for(path), "Alpha Back", "Bravo Back", hx=True)
    assert "<html" not in r.text
    assert "Start <strong>Alpha Back</strong>" in r.text and "Baseline pick" in r.text
    assert "AI unavailable; showing the baseline pick." in r.text
    assert "High</strong> 5+ points" in r.text  # the baseline explainer gives the projection-gap rule


@pytest.mark.parametrize(
    "players,message",
    [
        (["Alpha Back", " "], "Enter at least two players."),
        (["Alpha Back", "Nobody Here"], "no player matches &#39;Nobody Here&#39;"),
        (["Alpha Back", "Alpha Back"], "need at least two different players"),
    ],
)
def test_bad_questions_show_a_message(path, players, message):
    r = ask(client_for(path), *players, hx=True)
    assert r.status_code == 200 and message in r.text and 'role="alert"' in r.text


def test_the_page_ships_every_player_for_suggestions_and_only_the_button_submits(path):
    client = client_for(path)
    r = client.get("/")
    assert '<option value="Alpha Back (PHI)">RB, PHI</option>' in r.text
    assert '<option value="Bravo Back (DAL)">RB, DAL</option>' in r.text
    assert "hx-get" not in r.text  # nothing is fetched while typing
    assert "/static/app.js" in r.text
    assert "event.preventDefault()" in client.get("/static/app.js").text  # Enter doesn't submit
    assert client.get("/players", params={"players": "alp"}).status_code == 404  # no search endpoint


def test_a_suggested_value_resolves(path):
    r = ask(client_for(path), "Alpha Back (PHI)", "Bravo Back (DAL)", hx=True)
    assert "Start <strong>Alpha Back</strong>" in r.text


def test_each_visitor_has_their_own_ai_allowance(path):
    limits = [VisitorLimit(1, timedelta(minutes=10), "10 minutes")]
    client = client_for(path, FakeProvider(respond=lambda prompt: ANSWER), limits)
    assert "AI pick" in ask(client, "Alpha Back", "Bravo Back", hx=True).text
    capped = ask(client, "Alpha Back", "Bravo Back", scoring="std", hx=True).text
    assert "You&#39;ve reached the limit of 1 new AI answers per 10 minutes" in capped
    other = ask(client, "Alpha Back", "Bravo Back", scoring="std", hx=True, ip="198.51.100.4").text
    assert "AI pick" in other


def test_a_game_already_under_way_is_flagged(path):
    with Database(path) as db:
        db.execute("UPDATE games SET gameday = '2026-10-22', gametime = '20:15'")
    r = ask(client_for(path), "Alpha Back", "Bravo Back", hx=True)
    assert "game already kicked off (Thu 8:15 PM ET)" in r.text


def test_a_failed_log_write_still_shows_the_answer(path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(web, "record_recommendation", broken)
    assert "Start <strong>Alpha Back</strong>" in ask(client_for(path), "Alpha Back", "Bravo Back", hx=True).text


def test_empty_database_says_so(tmp_path):
    client = client_for(str(tmp_path / "empty.db"))
    assert "Answering for" not in client.get("/").text
    assert "No data yet" in ask(client, "Alpha Back", "Bravo Back", hx=True).text


def test_health_check():
    assert TestClient(web.create_app(":memory:")).get("/healthz").json()["ok"] is True


def test_record_recommendation_on_both_backends(db):
    from test_decide import make_packet

    packet = make_packet()
    decision = decide_module.baseline_decision(packet, "test")
    rec_id = decide_module.record_recommendation(db, packet, decision, "v1")
    [row] = db.query("select rec_id, pick, source, visitor from recommendations")
    assert row == {"rec_id": rec_id, "pick": "NACUA", "source": "baseline", "visitor": "v1"}


def test_the_page_compares_two_players_and_ignores_extras(path):
    client = client_for(path)
    assert client.get("/").text.count('name="players"') == 2
    r = ask(client, "Alpha Back", "Bravo Back", "Nobody Here", hx=True)
    assert "Start <strong>Alpha Back</strong>" in r.text  # a third name is ignored, not looked up


def test_citation_links_escape_text_and_skip_unknown_ids():
    from startsit.web.app import link_citations

    html = str(link_citations("<b>x</b> [E1] and E7, not E99", {"E1", "E7"}))
    assert (
        html == '&lt;b&gt;x&lt;/b&gt; <a class="cite" href="#E1">E1</a> and <a class="cite" href="#E7">E7</a>, not E99'
    )
