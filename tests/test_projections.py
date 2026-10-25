import pytest

from startsit.ingest import projections
from startsit.ingest.__main__ import run_projections


def ranking(fp_id, position, ecr, player_id=None):
    return dict(
        season=2026,
        week=7,
        scrape_date="2026-10-24",
        fantasypros_id=fp_id,
        player_id=player_id or f"G{fp_id}",
        player_name=fp_id,
        position=position,
        team="PHI",
        opponent=None,
        ecr=ecr,
        ecr_sd=None,
        ecr_best=None,
        ecr_worst=None,
        proj_ppr=None,
        pos_rank=None,
        start_sit_grade=None,
        game_status=None,
        source_url=None,
    )


def api_player(fp_id, std=10.0, half=12.0, ppr=14.0, rec=4.0):
    return {
        "fpid": int(fp_id),
        "name": fp_id,
        "stats": {"points": std, "points_half": half, "points_ppr": ppr, "rec_rec": rec},
    }


class FakeClient:
    def __init__(self, fail_after=None):
        self.calls, self.fail_after = [], fail_after

    def get(self, path, params):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise projections.FantasyProsQuotaExceeded("FantasyPros daily quota reached")
        self.calls.append((path, params))
        return {"players": [api_player(i) for i in params["players"].split(":")]}


def test_targets_take_the_top_ranked_players_per_position(monkeypatch):
    monkeypatch.setitem(projections.POSITION_DEPTH, "WR", 2)
    rows = [
        ranking("1", "WR", 3.0),
        ranking("2", "WR", 1.0),
        ranking("3", "WR", 2.0),
        ranking("4", "K", 1.0),
        ranking("5", "QB", None),
    ]
    assert projections.targets(rows) == {"WR": ["2", "3"]}


def test_fetch_chunks_by_ten_with_colon_separated_ids_and_stops_at_the_quota():
    wanted = {"WR": [str(i) for i in range(25)]}
    client = FakeClient()
    rows = projections.fetch(client, 2026, 7, wanted)
    assert len(rows) == 25 and len(client.calls) == 3
    assert client.calls[0] == (
        "2026/projections",
        {"position": "WR", "week": 7, "players": ":".join(map(str, range(10)))},
    )
    partial = projections.fetch(FakeClient(fail_after=1), 2026, 7, wanted)
    assert len(partial) == 10  # keeps what it fetched before the quota ran out


def test_build_maps_ids_and_formats():
    rows = projections.build(
        [api_player("1"), {"fpid": 2, "stats": {}}], [ranking("1", "WR", 1.0, "G1")], 2026, 7, "2026-10-25"
    )
    assert rows == [
        dict(
            season=2026,
            week=7,
            fetched_date="2026-10-25",
            fantasypros_id="1",
            player_id="G1",
            points_ppr=14.0,
            points_half=12.0,
            points_std=10.0,
            receptions=4.0,
        )
    ]


@pytest.mark.parametrize("has_rankings", [True, False])
def test_run_projections(db, has_rankings):
    if has_rankings:
        db.upsert("rankings", [ranking("1", "WR", 1.0, "G1"), ranking("2", "RB", 1.0, "G2")])
    n = run_projections(db, FakeClient(), 2026)
    assert n == (2 if has_rankings else 0)
    if has_rankings:
        assert db.latest_projection("G1", 2026, 7)["receptions"] == 4.0


def test_client_spaces_calls_and_retries_a_throttle_once(monkeypatch):
    import io
    import urllib.error

    sleeps, responses = [], []

    def fake_urlopen(req, timeout):
        r = responses.pop(0)
        if r == 429:
            raise urllib.error.HTTPError(req.full_url, 429, "Limit Exceeded", {}, io.BytesIO(b""))
        return io.BytesIO(b'{"players": []}')

    monkeypatch.setattr(projections.urllib.request, "urlopen", fake_urlopen)
    client = projections.FantasyProsClient("k", sleep=sleeps.append)
    responses[:] = [429, "ok", "ok"]
    assert client.get("p", {}) == {"players": []}  # throttled, waited, retried
    assert client.get("p", {}) == {"players": []}
    assert projections.FantasyProsClient("k").throttle_wait in sleeps
    assert any(0 < s <= client.min_interval for s in sleeps)  # spacing before the next call
    responses[:] = [429, 429]
    with pytest.raises(projections.FantasyProsQuotaExceeded):
        client.get("p", {})
