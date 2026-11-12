from datetime import UTC, datetime

import pytest
from test_evidence import seeded  # noqa: F401 (fixture)

from startsit import experts
from startsit.evidence import build_packet
from startsit.ingest.projections import FantasyProsQuotaExceeded

NOW = datetime(2026, 10, 23, 15, tzinfo=UTC)
ANSWER_AT = datetime(2026, 10, 23, 16, tzinfo=UTC)


def ranks(*pairs):
    return [{"expert_id": e, "rank": str(r)} for e, r in pairs]


def response(a="fp-SMITH", b="fp-NACUA"):
    """Experts 1-3 rank Smith ahead in PPR; expert 4 ranked only Smith; HALF is a 1-1 tie with one tie."""
    return {
        "rankings": {
            "PPR": {a: ranks(("1", 10), ("2", 12), ("3", 9), ("4", 5)), b: ranks(("1", 20), ("2", 15), ("3", 30))},
            "HALF": {a: ranks(("1", 10), ("2", 20), ("3", 7)), b: ranks(("1", 12), ("2", 15), ("3", 7))},
            "STD": {a: []},  # only one player: no split for the format
        }
    }


class FakeClient:
    def __init__(self, result=None):
        self.result, self.calls = result if result is not None else response(), []

    def get(self, path, params):
        self.calls.append((path, params))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_count_split_counts_each_experts_preference_per_format():
    split = experts.count_split(response()["rankings"])
    assert split == {
        "ppr": {"fp-SMITH": 3, "fp-NACUA": 0, "experts": 3},  # expert 4 ranked only one of them
        "half": {"fp-SMITH": 1, "fp-NACUA": 1, "experts": 3},  # expert 3 ranked them the same
    }


@pytest.mark.parametrize(
    "positions,expected",
    [(["WR", "WR"], "WR"), (["QB", "QB"], "QB"), (["RB", "WR"], "FLX"), (["QB", "WR"], None), (["K", "K"], None)],
)
def test_api_position(positions, expected):
    assert experts.api_position(positions) == expected


def test_ensure_fetches_once_a_day_and_the_packet_shows_the_split(seeded):  # noqa: F811
    client = FakeClient()
    assert experts.ensure(seeded, client, 2026, 7, ["SMITH", "NACUA"], NOW)
    assert client.calls == [("compare-players", {"players": "fp-NACUA:fp-SMITH", "position": "WR", "week": 7})]
    assert not experts.ensure(seeded, client, 2026, 7, ["NACUA", "SMITH"], NOW)  # cached today
    assert len(client.calls) == 1

    packet = build_packet(seeded, ["SMITH", "NACUA"], "ppr", 2026, 7, as_of=ANSWER_AT)
    smith, nacua = packet.players
    [item] = smith.items_of("experts")
    assert item.text == (
        "FantasyPros experts head to head (PPR, ranks as of 2026-10-23): 3 of 3 rank DeVonta Smith ahead of Puka Nacua"
    )
    assert item.source_url == experts_url() and nacua.items_of("experts")[0].text.startswith(
        "FantasyPros experts head to head (PPR, ranks as of 2026-10-23): 0 of 3 rank Puka Nacua ahead"
    )
    assert packet.experts_pick == "SMITH"
    assert build_packet(seeded, ["SMITH", "NACUA"], "half", 2026, 7, as_of=ANSWER_AT).experts_pick is None  # 1-1
    assert (
        build_packet(seeded, ["SMITH", "NACUA"], "std", 2026, 7, as_of=ANSWER_AT).players[0].items_of("experts") == []
    )
    # A packet asked before the split was fetched doesn't see it (no peeking).
    early = build_packet(seeded, ["SMITH", "NACUA"], "ppr", 2026, 7, as_of=datetime(2026, 10, 23, 14, tzinfo=UTC))
    assert early.experts_pick is None and early.players[0].experts_ahead is None


def experts_url():
    from startsit.evidence import FANTASYPROS_URL

    return FANTASYPROS_URL


def test_ensure_stops_at_the_daily_cap(seeded, monkeypatch):  # noqa: F811
    monkeypatch.setattr(experts, "DAILY_CALLS", 1)
    seeded.upsert(
        "expert_splits",
        [dict(season=2026, week=7, pair="x:y", fetched_day="2026-10-23", fetched_at=NOW.isoformat(), splits_json="{}")],
    )
    client = FakeClient()
    assert not experts.ensure(seeded, client, 2026, 7, ["SMITH", "NACUA"], NOW) and client.calls == []


@pytest.mark.parametrize(
    "client,player_ids",
    [
        (None, ["SMITH", "NACUA"]),  # no key
        (FakeClient(), ["SMITH"]),  # not a two-player question
        (FakeClient(), ["SMITH", "BYE"]),  # not ranked this week
        (FakeClient(FantasyProsQuotaExceeded("quota")), ["SMITH", "NACUA"]),
        (FakeClient(OSError("timed out")), ["SMITH", "NACUA"]),
        (FakeClient({"rankings": {}}), ["SMITH", "NACUA"]),  # nobody ranked both
    ],
)
def test_ensure_never_fails_the_question(seeded, client, player_ids):  # noqa: F811
    assert not experts.ensure(seeded, client, 2026, 7, player_ids, NOW)
    assert seeded.query("SELECT count(*) AS n FROM expert_splits")[0]["n"] == 0


def test_mixed_positions_use_the_flex_list_and_a_qb_pair_with_others_is_skipped(seeded):  # noqa: F811
    from test_evidence import ranking

    seeded.upsert("rankings", [ranking("HALL", 14.0, position="RB"), ranking("GENO", 18.0, position="QB")])
    client = FakeClient(response("fp-HALL", "fp-SMITH"))
    assert experts.ensure(seeded, client, 2026, 7, ["HALL", "SMITH"], NOW)
    assert client.calls[0][1]["position"] == "FLX"
    assert not experts.ensure(seeded, client, 2026, 7, ["GENO", "SMITH"], NOW) and len(client.calls) == 1
