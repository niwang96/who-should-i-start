import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from test_cli import _week as cli_week
from test_cli import seed

from startsit import decide as decide_module
from startsit import evaluate
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

    # Saturday before the week-7 Sunday games in the seeded schedule.
    return TestClient(web.create_app(path, llm_factory, clock=lambda: datetime(2026, 10, 24, 18, 0, tzinfo=UTC)))


def ask(client, *players, scoring="half", hx=False, ip="203.0.113.7"):
    headers = {"x-forwarded-for": f"{ip}, 10.0.0.1"} | ({"HX-Request": "true"} if hx else {})
    return client.post("/ask", data={"players": list(players), "scoring": scoring}, headers=headers)


def test_home_page_shows_the_week_and_data_freshness(path):
    with Database(path) as db:
        db.record_run("nflverse:player_weeks", 2026, 7, 1)
    r = client_for(path).get("/")
    assert r.status_code == 200
    assert "Answering for <strong>Week 7, 2026</strong> · games Sun Oct 25 \N{EN DASH} Sun Oct 25" in r.text
    assert "Rankings 2026-10-24" in r.text and "News never" in r.text and "Stats 20" in r.text
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


def test_no_ai_call_once_a_game_has_kicked_off(path):
    with Database(path) as db:
        db.execute("UPDATE games SET gameday = '2026-10-22', gametime = '20:15'")
    provider = FakeProvider()  # has no answers: any call would fail
    r = ask(client_for(path, provider), "Alpha Back", "Bravo Back", hx=True)
    assert "No pick needed" in r.text and "Alpha Back's game already kicked off (Thu 8:15 PM ET)" in r.text
    assert "Week 7 runs through Thu Oct 22" in r.text and 'id="E1"' in r.text  # evidence still shown
    assert provider.prompts == []
    with Database(path) as db:
        assert db.query("select count(*) as n from recommendations")[0]["n"] == 0


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


def test_evidence_is_a_side_by_side_comparison(path):
    r = ask(client_for(path, FakeProvider(responses=[ANSWER])), "Alpha Back", "Bravo Back", hx=True)
    # One row per kind of evidence (only kinds either player has), one column per player.
    assert r.text.count('class="kind"') == 3  # consensus, projection, matchup
    assert '<div class="kind">Expert consensus</div>' in r.text and "Injury report" not in r.text
    assert r.text.count('class="cell picked"') == 3 and r.text.count('class="cell"') == 3
    assert r.text.index('id="E1"') < r.text.index('id="E4"') < r.text.index('id="E2"')  # read across, row by row


def test_scoring_is_a_toggle_of_real_radio_buttons(path):
    r = client_for(path).get("/")
    assert r.text.count('type="radio" name="scoring"') == 3 and 'class="segmented"' in r.text
    assert 'value="ppr" checked' in r.text  # PPR is the default
    assert "Same name on two teams" not in r.text  # the ambiguity error explains it when it happens


def test_track_record_page(path, monkeypatch):
    from test_evaluate import rec

    client = client_for(path)
    r = client.get("/record")
    assert r.status_code == 200 and "Nothing scored yet" in r.text
    assert '<a href="/">Ask a question</a>' in r.text and '<a href="/record">Track record</a>' in client.get("/").text

    with Database(path) as db:
        db.upsert("player_weeks", [_week("A", 7, 20.0), _week("B", 7, 12.0, team="DAL")])
        ab = {"A": "PHI", "B": "DAL"}
        db.upsert(
            "recommendations",
            [
                rec("u1", ab, pick="A", baseline_pick="B", confidence="high", crowd_pick="A"),
                rec("b1", ab, pick="B", visitor="benchmark"),
                rec("f1", ab, pick="A", source="baseline", visitor="benchmark-backfill"),
            ],
        )
        evaluate.score_pending(db)
    r = client_for(path).get("/record")  # a new app: the page is cached for an hour per process
    page = r.text
    assert page.index('id="benchmark"') < page.index('id="backfill"') < page.index('id="users"')
    assert "Small sample: with 1 decided matchups" in page
    assert "AI started <strong>Player A</strong>" in page and "(20.0)</span> over Player B" in page
    assert "Right 100% of the time (1 of 1)" in page  # the users' override
    assert "The AI didn't answer any of these yet" in page  # the backfill had only baseline picks
    assert '<span class="label">Reddit crowd</span>' in page and page.count('<th scope="col">Crowd</th>') == 2
    assert page.count("no lean") == 0 and "start/sit votes from r/fantasyfootball" in page
    assert client.get("/record").text.count("Nothing scored yet") == 1  # the first app still serves its cache


def _week(player_id, week, fp_ppr, team="PHI"):
    return cli_week(player_id, week) | dict(team=team, fp_ppr=fp_ppr, fp_half=fp_ppr, fp_std=fp_ppr)


def test_an_unexpected_error_shows_a_message_instead_of_a_blank_page(path, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("database hiccup")

    monkeypatch.setattr(web, "packet_for_names", broken)
    r = ask(client_for(path), "Alpha Back", "Bravo Back", hx=True)
    assert r.status_code == 200 and "Something went wrong on our side" in r.text and 'role="alert"' in r.text
    assert "answering ['Alpha Back', 'Bravo Back'] failed" in caplog.text


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.rotowire.com/football/player/x", "https://www.rotowire.com/football/player/x"),
        ("http://example.com/a", "http://example.com/a"),
        ("javascript:alert(1)", None),
        (" JavaScript:alert(1)", None),
        ("data:text/html;base64,xx", None),
        ("", None),
        (None, None),
    ],
)
def test_only_web_links_are_rendered(url, expected):
    assert web.safe_href(url) == expected


def test_a_feed_link_that_is_not_a_web_link_is_not_rendered(path):
    with Database(path) as db:
        db.execute("UPDATE rankings SET source_url = 'javascript:alert(1)' WHERE player_id = 'A'")
    r = ask(client_for(path), "Alpha Back", "Bravo Back", hx=True)
    assert "javascript:" not in r.text and "Start <strong>Alpha Back</strong>" in r.text


def test_cross_site_posts_are_refused(path):
    client = client_for(path)
    r = client.post("/ask", data={"players": ["Alpha Back", "Bravo Back"]}, headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403
    same = client.post(
        "/ask", data={"players": ["Alpha Back", "Bravo Back"]}, headers={"Sec-Fetch-Site": "same-origin"}
    )
    assert same.status_code == 200


def test_only_the_first_few_form_fields_are_read():
    assert web.clean_names([" ", "Alpha Back", *["x"] * 10_000]) == ["Alpha Back", "x", "x"]


def test_why_page_shows_the_stored_answer_its_evidence_and_what_happened(path):
    client = client_for(path, FakeProvider(responses=[ANSWER]))
    ask(client, "Alpha Back", "Bravo Back", hx=True)
    with Database(path) as db:
        [rec] = db.query("SELECT rec_id FROM recommendations")
        rec_id = rec["rec_id"]
        r = client.get(f"/record/{rec_id}")
        assert r.status_code == 200 and "Alpha Back vs Bravo Back" in r.text
        assert "Not scored yet" in r.text and "asked Sat Oct 24, 2:00 PM ET" in r.text
        assert "Start <strong>Alpha Back</strong>" in r.text and 'id="E2"' in r.text  # the answer and evidence
        assert "Questions asked on this site" in r.text

        db.upsert("player_weeks", [_week("A", 7, 20.0), _week("B", 7, 12.0, team="DAL")])
        evaluate.score_pending(db)
    scored = client.get(f"/record/{rec_id}").text
    assert "What happened" in scored and "scored more" in scored and "20.0" in scored
    assert scored.count('class="badge hit"') >= 2  # the AI and the baseline were right
    record = client_for(path).get("/record").text
    assert f'href="/record/{rec_id}"' in record and "All 1 matchups" in record


@pytest.mark.parametrize("rec_id", ["0" * 32, "not-an-id"])
def test_why_page_for_an_unknown_matchup(path, rec_id):
    r = client_for(path).get(f"/record/{rec_id}")
    assert r.status_code == 404 and "That matchup isn't in the track record." in r.text
    assert client_for(path).get("/record/../etc").status_code == 404  # never reaches the route


def test_the_answer_says_when_the_reddit_crowd_disagrees_with_the_experts(path):
    from test_evidence import vote

    with Database(path) as db:
        db.upsert("crowd_votes", [vote(f"c{i}", "B", "start", "2026-10-22T12:00:00+00:00") for i in range(3)])
    r = ask(client_for(path, FakeProvider(responses=[ANSWER])), "Alpha Back", "Bravo Back", hx=True)
    assert "The Reddit crowd disagrees with the experts:</strong>" in r.text
    assert "fans lean Bravo Back, while the expert consensus favors Alpha Back" in r.text
    assert "(Alpha Back 0 start / 0 sit, Bravo Back 3 start / 0 sit)" in r.text
    assert "The AI sided with the experts." in r.text
    assert '<div class="kind">Reddit crowd</div>' in r.text
    with Database(path) as db:
        assert db.query("SELECT crowd_pick FROM recommendations") == [{"crowd_pick": "B"}]

    r = ask(client_for(path), "Alpha Back", "Bravo Back", scoring="ppr", hx=True)  # no AI: the baseline pick
    assert "fans lean Bravo Back" in r.text and "The AI sided" not in r.text


def test_no_crowd_note_when_the_crowd_agrees_or_has_no_lean(path):
    from test_evidence import vote

    with Database(path) as db:
        db.upsert("crowd_votes", [vote(f"c{i}", "A", "start", "2026-10-22T12:00:00+00:00") for i in range(3)])
    r = ask(client_for(path), "Alpha Back", "Bravo Back", hx=True)
    assert "crowd-note" not in r.text and "3 start / 0 sit" in r.text  # the evidence still shows the votes
