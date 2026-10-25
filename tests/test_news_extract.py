from datetime import UTC, datetime

import pytest

from startsit.ingest.news_extract import (
    CHUNK_SIZE,
    PlayerMatcher,
    build_prompt,
    extract_news,
    facts_from_response,
    normalize_name,
)
from startsit.llm import BudgetedLLM, FakeProvider, LLMError, QuotaExceeded

NOW = datetime(2026, 10, 24, 20, 0, tzinfo=UTC)


def player(player_id, name, team, position="WR"):
    return dict(player_id=player_id, name=name, position=position, team=team, pfr_id=None)


PLAYERS = [
    player("NACUA", "Puka Nacua", "LA"),
    player("WALKER", "Kenneth Walker III", "SEA", "RB"),
    player("MILLER_NE", "Ryan Miller", "NE"),
    player("MILLER_DAL", "Ryan Miller", "DAL"),
]


def item(n, published_at="2026-10-24T10:00:00+00:00", summary="Details."):
    return dict(
        item_id=f"RotoWire:{n}",
        url=f"https://example.com/{n}",
        source="RotoWire",
        title=f"Headline {n}",
        summary=summary,
        published_at=published_at,
        fetched_at="2026-10-24T20:00:00+00:00",
    )


def fact(
    item_id, name="Puka Nacua", team="LAR", event="injury", severity="medium", summary="Questionable.", status="none"
):
    return dict(
        item_id=item_id, player=name, team=team, event=event, severity=severity, summary=summary, game_status=status
    )


def test_normalize_name_and_matcher():
    assert normalize_name("Kenneth Walker III") == normalize_name("kenneth walker") == "kenneth walker"
    assert normalize_name("Amon-Ra St. Brown") == "amon ra st brown"
    m = PlayerMatcher(PLAYERS)
    assert m.match("Kenneth Walker") == "WALKER"
    assert m.match("Ryan Miller") is None  # ambiguous without a team
    assert m.match("Ryan Miller", "ne") == "MILLER_NE"
    assert m.match("Puka Nacua", "LAR") == "NACUA"  # alias not needed for a unique name
    assert m.match("Nobody") is None


def test_facts_from_response_attaches_code_owned_fields_and_drops_bad_rows():
    items = [item(1), item(2)]
    response = {
        "facts": [
            fact("N1"),
            fact("N1", summary="Duplicate for the same player and item: last one wins.", status="questionable"),
            fact("N2", name="Ryan Miller", team="NE", event="transaction", severity="low", summary="Signed."),
            fact("N9"),  # unknown item id
            fact("N2", name="Somebody Else"),  # unmatched player
            fact("N2", event="vibes"),  # invalid event
            fact("N2", summary=" "),  # empty summary
            fact("N2", event="other", severity="low"),  # color, not a playing-time fact
        ]
    }
    facts = facts_from_response(response, items, PlayerMatcher(PLAYERS))
    assert facts == [
        dict(
            item_id="RotoWire:1",
            player_id="NACUA",
            url="https://example.com/1",
            source="RotoWire",
            published_at="2026-10-24T10:00:00+00:00",
            event="injury",
            severity="medium",
            summary="Duplicate for the same player and item: last one wins.",
            game_status="questionable",
        ),
        dict(
            item_id="RotoWire:2",
            player_id="MILLER_NE",
            url="https://example.com/2",
            source="RotoWire",
            published_at="2026-10-24T10:00:00+00:00",
            event="transaction",
            severity="low",
            summary="Signed.",
            game_status=None,  # "none" from the model is stored as no status
        ),
    ]
    assert facts_from_response(None, items, PlayerMatcher(PLAYERS)) == []


def test_build_prompt_numbers_items():
    assert build_prompt([item(1), item(2, summary=None)]) == (
        "News items:\n[N1] Headline 1 | Details.\n[N2] Headline 2"
    )


@pytest.fixture
def news_db(db):
    db.upsert("players", PLAYERS)
    return db


def budgeted(db, provider, budget=10):
    return BudgetedLLM(provider, db, "news", budget)


def test_extract_news_caches_by_url_and_never_resends(news_db):
    news_db.upsert("news_items", [item(1), item(2), item(3, published_at="2026-10-10T10:00:00+00:00")])
    provider = FakeProvider(responses=[{"facts": [fact("N1")]}])
    result = extract_news(news_db, budgeted(news_db, provider), NOW)
    assert (result.items_processed, result.facts, result.calls, result.stopped_early) == (2, 1, 1, None)
    assert "Headline 3" not in provider.prompts[0]  # older than the lookback window
    assert news_db.query("select item_id, fact_count from news_extractions order by item_id") == [
        {"item_id": "RotoWire:1", "fact_count": 1},
        {"item_id": "RotoWire:2", "fact_count": 0},
    ]
    # A second run has nothing new to send.
    again = extract_news(news_db, budgeted(news_db, provider), NOW)
    assert (again.items_processed, again.calls) == (0, 0)
    assert news_db.llm_calls_since("2000-01-01 00:00:00", "news") == 1


def test_extract_news_chunks_and_stops_at_the_budget(news_db):
    news_db.upsert("news_items", [item(n) for n in range(CHUNK_SIZE + 5)])
    provider = FakeProvider(respond=lambda prompt: {"facts": []})
    result = extract_news(news_db, budgeted(news_db, provider, budget=1), NOW)
    assert (result.items_processed, result.calls) == (CHUNK_SIZE, 1)
    assert "budget of 1" in result.stopped_early
    assert len(news_db.unextracted_news("2000-01-01", 100)) == 5


@pytest.mark.parametrize("error", [LLMError("boom"), QuotaExceeded("Gemini rate limit")])
def test_extract_news_leaves_items_for_the_next_run_when_the_llm_fails(news_db, error):
    news_db.upsert("news_items", [item(1)])
    result = extract_news(news_db, budgeted(news_db, FakeProvider(responses=[error])), NOW)
    assert (result.items_processed, result.calls) == (0, 1)
    assert result.stopped_early == str(error)
    assert len(news_db.unextracted_news("2000-01-01", 10)) == 1
    assert news_db.query("select ok from llm_calls") == [{"ok": 0}]
