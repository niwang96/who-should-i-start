from datetime import UTC, datetime

import pytest

from startsit.evidence import (
    PlayerLookupError,
    build_packet,
    current_week,
    find_player,
    parse_player_arg,
)


def player(player_id, name, position, team):
    return dict(player_id=player_id, name=name, position=position, team=team, pfr_id=None)


def usage(player_id, through_week, last_game_week=None, **kw):
    row = dict(
        player_id=player_id,
        season=2026,
        through_week=through_week,
        games=6,
        last_game_week=last_game_week or through_week,
        snap_pct_l3=0.85,
        snap_pct_trend=-0.1,
        target_share_l3=0.34,
        opportunities_l3=11.7,
        fp_std_l3=12.0,
        fp_half_l3=15.0,
        fp_ppr_l3=18.0,
        xfp_std_l3=10.0,
        xfp_half_l3=12.0,
        xfp_ppr_l3=14.0,
        xfp_ppr_season=14.0,
        fp_minus_xfp_ppr_l3=4.0,
    )
    return row | kw


def ranking(player_id, proj_ppr, scrape_date="2026-10-23", **kw):
    row = dict(
        season=2026,
        week=7,
        scrape_date=scrape_date,
        fantasypros_id=f"fp-{player_id}",
        player_id=player_id,
        player_name=player_id,
        position="WR",
        team="PHI",
        opponent="vs. DAL",
        ecr=16.2,
        ecr_sd=2.53,
        ecr_best=9.0,
        ecr_worst=25.0,
        proj_ppr=proj_ppr,
        pos_rank="WR16",
        start_sit_grade="A-",
        game_status="scheduled",
        source_url=f"https://www.fantasypros.com/nfl/players/{player_id}.php",
    )
    return row | kw


def game(week, home, away, spread_line=None, total_line=None, gameday=None, gametime=None):
    return dict(
        game_id=f"2026_{week:02d}_{away}_{home}",
        season=2026,
        week=week,
        gameday=gameday,
        gametime=gametime,
        home_team=home,
        away_team=away,
        spread_line=spread_line,
        total_line=total_line,
    )


def roster(player_id, week, team, status="ACT"):
    return dict(player_id=player_id, season=2026, week=week, team=team, status=status)


def injury(player_id, report_status, practice_status, injury_name):
    return dict(
        player_id=player_id,
        season=2026,
        week=7,
        team=None,
        report_status=report_status,
        practice_status=practice_status,
        injury=injury_name,
    )


MATCHUPS = [("PHI", "DAL"), ("LV", "LA"), ("NYJ", "MIA"), ("ATL", "SF"), ("GB", "NYG")]


@pytest.fixture
def seeded(db):
    db.upsert(
        "players",
        [
            player("SMITH", "DeVonta Smith", "WR", "PHI"),
            player("NACUA", "Puka Nacua", "WR", "LA"),
            player("HALL", "Breece Hall", "RB", "NYJ"),
            player("BYE", "Bye Guy", "WR", "DET"),
            player("GENO", "Geno Smith", "QB", "NYJ"),
            player("DALTON", "Andy Dalton", "QB", "ATL"),
            player("JACOBS", "Josh Jacobs", "RB", "GB"),
            player("CUT", "Najee Harris", "RB", "NYG"),
            player("IR", "Christian Kirk", "WR", "SF"),
            player("MILLER1", "Ryan Miller", "WR", "NE"),
            player("MILLER2", "Ryan Miller", "WR", "DAL"),
            player("BROWN", "A.J. Brown", "WR", "PHI"),
        ],
    )
    db.upsert(
        "games",
        [game(w, home, away) for w in range(1, 8) for home, away in MATCHUPS]
        + [game(w, "DET", "CHI") for w in range(1, 7)],  # DET and CHI on bye in week 7
    )
    db.upsert(
        "roster_weeks",
        [roster("DALTON", w, "PHI") for w in range(1, 6)]
        + [roster("DALTON", w, "ATL") for w in (6, 7)]
        + [roster("JACOBS", 7, "GB", "EXE"), roster("CUT", 7, "NYG", "CUT"), roster("IR", 7, "SF", "RES")],
    )
    db.upsert(
        "usage_features",
        [
            usage("SMITH", 5),
            usage("SMITH", 6),
            usage("NACUA", 6, xfp_ppr_l3=20.0, xfp_std_l3=10.0),
            usage("HALL", 6, last_game_week=3),
            usage("BYE", 6),
            usage("DALTON", 6, last_game_week=3),
            usage("SMITH", 7, snap_pct_l3=0.1),  # after the week asked about: must be ignored
        ],
    )
    db.upsert(
        "rankings",
        [
            ranking("SMITH", 14.0, scrape_date="2026-10-21"),
            ranking("SMITH", 15.1),
            ranking("NACUA", 24.5, pos_rank="WR1", opponent="at LV"),
            ranking("HALL", 9.0, position="RB", pos_rank="RB30"),
            ranking("JACOBS", 12.0, position="RB", pos_rank="RB20"),
            ranking("CUT", 3.0, position="RB", pos_rank="RB60"),
            ranking("IR", 6.7, pos_rank="WR61"),
        ],
    )
    db.upsert(
        "injuries",
        [
            injury("NACUA", None, "Limited Participation in Practice", "Calf"),
            injury("HALL", "Out", "Did Not Participate In Practice", "Quadricep"),
        ],
    )
    return db


def texts(packet, player_index):
    return [i.text for i in packet.players[player_index].items]


def test_packet_converts_scoring_and_picks_by_consensus(seeded):
    p = build_packet(seeded, ["SMITH", "NACUA"], "half", 2026, 7)
    smith, nacua = p.players
    assert smith.score.points == 13.1  # 15.1 PPR - 0.5 x 4 expected catches, from the latest scrape
    assert nacua.score.points == 19.5  # 24.5 - 0.5 x 10
    assert (p.baseline_pick, p.baseline_margin) == ("NACUA", 6.4)
    assert [i.id for i in smith.items + nacua.items] == [f"E{n}" for n in range(1, 8)]
    assert p.item_ids == {f"E{n}" for n in range(1, 8)}
    assert smith.items[0].source_url == "https://www.fantasypros.com/nfl/players/SMITH.php"

    text = p.to_text()
    assert text.startswith(
        "Week 7 2026, half-PPR scoring. Baseline (100% FantasyPros consensus) favors Puka Nacua by 6.4 pts."
    )
    assert "E1 [consensus] FantasyPros consensus WR16 vs. DAL (avg rank 16.2, sd 2.53, range 9-25" in text
    assert "about 13.1 half-PPR (minus 0.5 x 4.0 expected catches/game from recent usage)" in text
    assert "85% of snaps (10 percentage points down vs prior 3)" in text
    assert "12.0 expected vs 15.0 actual half-PPR fantasy pts/game" in text
    assert "10%" not in text  # week-7 usage (10% snaps) not leaked
    assert "E7 [injury] Week 7 injury report: no game status yet (Calf), practice: Limited" in text


def test_out_bye_and_unranked_players_score_zero(seeded):
    p = build_packet(seeded, ["HALL", "BYE", "GENO"], "ppr", 2026, 7)
    hall, bye, geno = p.players
    assert (hall.score.points, hall.score.note) == (0.0, "ruled Out")
    assert any("has missed 3 team games since last playing in week 3" in t for t in texts(p, 0))
    assert (bye.score.points, bye.score.note) == (0.0, "on bye")
    assert texts(p, 1)[0] == "DET is on bye in week 7"
    assert not any("has missed" in t for t in texts(p, 1))
    assert geno.score.note == "not in FantasyPros weekly rankings"
    assert "has Breece Hall and Bye Guy and Geno Smith tied at 0.0 pts" in p.to_text()


def test_traded_player_uses_roster_history(seeded):
    p = build_packet(seeded, ["DALTON", "SMITH"], "ppr", 2026, 7)
    dalton = p.players[0]
    assert dalton.team == "ATL"
    # Weeks 4-5 with PHI and week 6 with ATL: 3 team games missed, without naming the old team.
    assert any("has missed 3 team games since last playing in week 3" in t for t in texts(p, 0))
    assert not any("PHI" in t for t in texts(p, 0))


def test_only_active_roster_players_can_score(seeded):
    p = build_packet(seeded, ["JACOBS", "CUT", "IR", "SMITH"], "ppr", 2026, 7)
    jacobs, cut, ir, smith = p.players
    # A player without a 2026 game is still queryable, but exempt/reserve/released all score zero.
    assert (jacobs.score.points, jacobs.score.note) == (0.0, "on the exempt list, not on the active roster")
    assert "Roster status: on the exempt list (not on the active roster)" in texts(p, 0)
    assert "FantasyPros consensus RB20" in texts(p, 0)[1]  # the experts' view is still shown
    assert (cut.score.points, cut.score.note) == (0.0, "released, not on the active roster")
    assert cut.team is None and "Najee Harris (RB, FA)" in p.to_text()
    assert ir.score.points == 0.0
    assert texts(p, 2)[0] == "Roster status: on a reserve list (IR/PUP/NFI) (not on the active roster)"
    assert smith.score.points == 15.1  # no roster row: left to the consensus
    assert "favors DeVonta Smith by 15.1 pts" in p.to_text()


def test_packet_rejects_bad_input(seeded):
    with pytest.raises(ValueError):
        build_packet(seeded, ["SMITH", "SMITH"], "ppr", 2026, 7)
    with pytest.raises(ValueError):
        build_packet(seeded, ["SMITH", "NACUA"], "superflex", 2026, 7)  # type: ignore[arg-type]
    with pytest.raises(PlayerLookupError):
        build_packet(seeded, ["SMITH", "NOBODY"], "ppr", 2026, 7)


def test_find_player_and_current_week(seeded):
    assert find_player(seeded, "puka nacua")["player_id"] == "NACUA"
    assert find_player(seeded, "Breece")["player_id"] == "HALL"
    assert find_player(seeded, "Ryan Miller", "ne")["player_id"] == "MILLER1"
    assert find_player(seeded, "AJ Brown")["player_id"] == find_player(seeded, "a.j. brown")["player_id"] == "BROWN"
    with pytest.raises(PlayerLookupError, match=r"ambiguous: Ryan Miller \(DAL\), Ryan Miller \(NE\)"):
        find_player(seeded, "Ryan Miller")
    with pytest.raises(PlayerLookupError, match="no player"):
        find_player(seeded, "Zzz")
    assert current_week(seeded, 2026) == 7
    assert current_week(seeded, 2025) == 1


@pytest.mark.parametrize(
    "arg,expected",
    [
        ("Ryan Miller (ne)", ("Ryan Miller", "NE")),
        (" Puka Nacua ", ("Puka Nacua", None)),
        ("A.J. Brown", ("A.J. Brown", None)),
    ],
)
def test_parse_player_arg(arg, expected):
    assert parse_player_arg(arg) == expected


def news_fact(player_id, published_at, summary, url_suffix, event="injury", severity="medium", game_status=None):
    return dict(
        game_status=game_status,
        item_id=f"RotoWire:{url_suffix}",
        player_id=player_id,
        url=f"https://example.com/news/{url_suffix}",
        source="RotoWire",
        published_at=published_at,
        event=event,
        severity=severity,
        summary=summary,
    )


def test_news_facts_are_recent_capped_and_cut_off_at_as_of(seeded):
    seeded.upsert(
        "news_facts",
        [
            news_fact("NACUA", "2026-10-10T12:00:00+00:00", "Too old.", 1),
            news_fact("NACUA", "2026-10-21T12:00:00+00:00", "Limited Wednesday.", 2, event="practice", severity="low"),
            news_fact("NACUA", "2026-10-22T12:00:00+00:00", "Limited Thursday.", 3, event="practice", severity="low"),
            news_fact("NACUA", "2026-10-23T12:00:00+00:00", "Did not practice Friday.", 4, severity="high"),
            news_fact("NACUA", "2026-10-24T12:00:00+00:00", "Questionable for Sunday.", 5),
            news_fact("NACUA", "2026-10-26T12:00:00+00:00", "After the question was asked.", 6),
        ],
    )
    as_of = datetime(2026, 10, 25, 0, 0, tzinfo=UTC)
    p = build_packet(seeded, ["NACUA", "SMITH"], "ppr", 2026, 7, as_of=as_of)
    news = [i for i in p.players[0].items if i.kind == "news"]
    assert [i.text for i in news] == [
        "2026-10-24 RotoWire: Questionable for Sunday. (injury, medium impact)",
        "2026-10-23 RotoWire: Did not practice Friday. (injury, high impact)",
        "2026-10-22 RotoWire: Limited Thursday. (practice, low impact)",
    ]
    assert news[0].source_url == "https://example.com/news/5"
    assert not [i for i in p.players[1].items if i.kind == "news"]


def test_matchup_fact_uses_betting_lines_from_each_teams_side(seeded):
    # PHI hosts DAL, favored by 3.5 (total 49.5); LA plays at LV, a 2.5-point underdog (total 44).
    seeded.upsert("games", [game(7, "PHI", "DAL", 3.5, 49.5), game(7, "LV", "LA", 2.5, 44.0)])
    p = build_packet(seeded, ["SMITH", "NACUA"], "ppr", 2026, 7)
    matchups = {pl.name: [i.text for i in pl.items if i.kind == "matchup"] for pl in p.players}
    assert matchups == {
        "DeVonta Smith": ["PHI vs DAL: betting lines imply 26.5 team points (favored by 3.5, game total 49.5)"],
        "Puka Nacua": ["LA at LV: betting lines imply 20.8 team points (underdog by 2.5, game total 44)"],
    }


def test_no_matchup_fact_without_lines(seeded):
    p = build_packet(seeded, ["SMITH", "NACUA"], "ppr", 2026, 7)
    assert not [i for pl in p.players for i in pl.items if i.kind == "matchup"]


def projection(player_id, receptions, fetched_date="2026-10-24"):
    return dict(
        season=2026,
        week=7,
        fetched_date=fetched_date,
        fantasypros_id=f"fp-{player_id}",
        player_id=player_id,
        points_ppr=None,
        points_half=None,
        points_std=None,
        receptions=receptions,
    )


def test_fantasypros_projected_catches_drive_the_conversion(seeded):
    seeded.upsert("projections", [projection("SMITH", 9.0, "2026-10-23"), projection("SMITH", 5.0)])
    p = build_packet(seeded, ["SMITH", "NACUA"], "half", 2026, 7)
    assert p.players[0].score.points == 12.6  # 15.1 - 0.5 x 5.0 (latest fetch), not usage's 4.0
    assert "minus 0.5 x 5.0 FantasyPros projected catches" in p.to_text()


def test_this_weeks_news_status_beats_stale_roster_and_last_weeks_news(seeded):
    as_of = datetime(2026, 10, 25, 0, 0, tzinfo=UTC)
    seeded.upsert(
        "news_facts",
        [
            news_fact(
                "JACOBS",
                "2026-10-24T20:09:00+00:00",
                "Exemption removed, expected to play.",
                7,
                event="return",
                game_status="active",
            ),
            news_fact("NACUA", "2026-10-24T18:00:00+00:00", "Ruled out for Sunday.", 8, game_status="out"),
        ],
    )
    p = build_packet(seeded, ["JACOBS", "NACUA", "SMITH"], "ppr", 2026, 7, as_of=as_of)
    jacobs, nacua, smith = p.players
    assert jacobs.score.points == 12.0 and jacobs.score.note is None  # news "active" beats the exempt roster
    assert (nacua.score.points, nacua.score.note) == (0.0, "ruled out (news)")
    assert nacua.game_status == "out" and smith.game_status is None


def test_news_status_from_before_the_teams_last_game_is_ignored(seeded):
    seeded.upsert("games", [dict(game(6, "LV", "LA"), gameday="2026-10-18")])
    seeded.upsert(
        "news_facts", [news_fact("NACUA", "2026-10-18T12:00:00+00:00", "Out for Sunday.", 9, game_status="out")]
    )
    p = build_packet(seeded, ["NACUA", "SMITH"], "ppr", 2026, 7, as_of=datetime(2026, 10, 23, 12, tzinfo=UTC))
    assert p.players[0].score.points == 24.5  # last week's "out" doesn't count this week


def test_matchup_shows_kickoff_and_flags_games_already_started(seeded):
    seeded.upsert(
        "games",
        [
            game(7, "PHI", "DAL", 3.5, 49.5, gameday="2026-10-22", gametime="20:15"),  # Thursday night
            game(7, "LV", "LA", 2.5, 44.0, gameday="2026-10-25", gametime="16:05"),
        ],
    )
    p = build_packet(seeded, ["SMITH", "NACUA"], "ppr", 2026, 7, as_of=datetime(2026, 10, 24, tzinfo=UTC))
    smith, nacua = p.players
    assert "PHI vs DAL, kickoff Thu 8:15 PM ET: betting lines" in p.to_text()
    assert "LA at LV, kickoff Sun 4:05 PM ET" in p.to_text()
    assert smith.kickoff == datetime(2026, 10, 23, 0, 15, tzinfo=UTC) and smith.game_started
    assert not nacua.game_started


def test_in_game_news_from_last_weeks_late_game_is_ignored(seeded):
    # LA played a Monday night game (week 6: 8:15 PM ET = 00:15 UTC Tuesday).
    seeded.upsert("games", [game(6, "LV", "LA", gameday="2026-10-19", gametime="20:15")])
    seeded.upsert(
        "news_facts",
        [
            news_fact("NACUA", "2026-10-20T02:30:00+00:00", "Won't return Monday.", 10, game_status="out"),
            news_fact("SMITH", "2026-10-20T02:30:00+00:00", "Unrelated team, same time.", 11, game_status="out"),
        ],
    )
    p = build_packet(seeded, ["NACUA", "SMITH"], "ppr", 2026, 7, as_of=datetime(2026, 10, 23, 12, tzinfo=UTC))
    assert p.players[0].score.points == 24.5  # during LA's game: not a status for week 7
    assert p.players[1].score.points == 0.0  # PHI's week-6 game (no date in the fixture) doesn't cover it
    seeded.upsert("news_facts", [news_fact("NACUA", "2026-10-20T15:00:00+00:00", "Ruled out.", 12, game_status="out")])
    p = build_packet(seeded, ["NACUA", "SMITH"], "ppr", 2026, 7, as_of=datetime(2026, 10, 23, 12, tzinfo=UTC))
    assert p.players[0].score.points == 0.0  # the next day: counts


def test_a_past_question_sees_only_rankings_and_projections_from_then(seeded):
    seeded.upsert("rankings", [ranking("NACUA", 10.0, scrape_date="2026-10-25")])  # a later scrape
    seeded.upsert("projections", [projection("NACUA", 9.0, fetched_date="2026-10-25")])
    then = build_packet(seeded, ["NACUA", "SMITH"], "half", 2026, 7, as_of=datetime(2026, 10, 24, tzinfo=UTC))
    now = build_packet(seeded, ["NACUA", "SMITH"], "ppr", 2026, 7)
    # Then: the Oct 23 projection (24.5 PPR) converted with usage-based catches (10.0), since the
    # FantasyPros catches (9.0) were fetched later. Now: the Oct 25 scrape.
    assert then.players[0].score.points == 19.5 and now.players[0].score.points == 10.0
    assert "rankings as of 2026-10-23" in then.to_text() and "rankings as of 2026-10-25" in now.to_text()
    assert "expected catches/game from recent usage" in then.to_text()


def test_a_stored_packet_reads_back_exactly(seeded):
    from startsit.evidence import EvidencePacket

    packet = build_packet(seeded, ["NACUA", "SMITH"], "half", 2026, 7, as_of=datetime(2026, 10, 24, tzinfo=UTC))
    again = EvidencePacket.from_json(packet.to_json())
    assert again == packet and again.to_text() == packet.to_text()


def vote(comment_id, player_id, verdict, published_at, reason=None):
    url = f"https://www.reddit.com/r/fantasyfootball/comments/t/x/{comment_id}/"
    return dict(
        comment_id=comment_id, player_id=player_id, verdict=verdict, reason=reason, published_at=published_at, url=url
    )


def test_crowd_votes_from_this_weeks_threads_become_one_item_and_a_crowd_pick(seeded):
    from startsit.evidence import EvidencePacket

    seeded.upsert("games", [dict(game(6, "LV", "LA"), gameday="2026-10-18")])  # NACUA's last game
    seeded.upsert(
        "crowd_votes",
        [
            vote("c0", "NACUA", "sit", "2026-10-18T12:00:00+00:00"),  # before last week's game: last week's thread
            vote("c1", "NACUA", "start", "2026-10-21T12:00:00+00:00", "safe floor"),
            vote("c2", "NACUA", "start", "2026-10-22T12:00:00+00:00"),
            vote("c3", "NACUA", "sit", "2026-10-22T13:00:00+00:00", "tough matchup"),
            vote("c4", "NACUA", "start", "2026-10-23T09:00:00+00:00", "safe floor"),
            vote("c5", "NACUA", "start", "2026-10-23T11:00:00+00:00"),
            vote("c6", "NACUA", "sit", "2026-10-24T09:00:00+00:00"),  # after the question
            vote("c7", "SMITH", "start", "2026-10-22T12:00:00+00:00"),
        ],
    )
    as_of = datetime(2026, 10, 23, 12, tzinfo=UTC)
    p = build_packet(seeded, ["NACUA", "SMITH"], "ppr", 2026, 7, as_of=as_of)
    nacua, smith = p.players
    [item] = nacua.items_of("crowd")
    assert item.text == (
        "Reddit r/fantasyfootball Who Do I Start threads this week: 4 start / 1 sit (fans' opinions, one vote "
        'per comment); reasons given: "safe floor" (start), "tough matchup" (sit)'
    )
    assert item.source_url.endswith("/c5/")  # the newest vote
    assert (nacua.crowd_start, nacua.crowd_sit, smith.crowd_start) == (4, 1, 1)
    assert p.crowd_pick == "NACUA"  # net 3 vs 1: a lead of CROWD_MIN_LEAD
    assert EvidencePacket.from_json(p.to_json()).crowd_pick == "NACUA"

    seeded.upsert("crowd_votes", [vote("c8", "SMITH", "start", "2026-10-22T14:00:00+00:00")])
    assert build_packet(seeded, ["NACUA", "SMITH"], "ppr", 2026, 7, as_of=as_of).crowd_pick is None  # 3 vs 2
    assert build_packet(seeded, ["HALL", "GENO"], "ppr", 2026, 7, as_of=as_of).crowd_pick is None  # no votes
