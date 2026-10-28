from startsit.names import PlayerMatcher, normalize_name, normalize_team


def player(player_id, name, team, position="WR"):
    return dict(player_id=player_id, name=name, position=position, team=team, pfr_id=None)


PLAYERS = [
    player("NACUA", "Puka Nacua", "LA"),
    player("WALKER", "Kenneth Walker III", "SEA", "RB"),
    player("BROWN", "A.J. Brown", "PHI"),
    player("STBROWN", "Amon-Ra St. Brown", "DET"),
    player("MILLER_NE", "Ryan Miller", "NE"),
    player("MILLER_DAL", "Ryan Miller", "DAL"),
]


def test_normalize():
    assert normalize_name("Kenneth Walker III") == normalize_name("kenneth walker") == "kenneth walker"
    assert normalize_name("Amon-Ra St. Brown") == "amon ra st brown"
    assert normalize_name("A.J. Brown") == normalize_name("AJ Brown") == "aj brown"
    assert normalize_team(" lar ") == "LA" and normalize_team("ne") == "NE"


def test_match_is_exact_and_uses_the_team_only_to_break_ties():
    m = PlayerMatcher(PLAYERS)
    assert m.match("Kenneth Walker") == "WALKER"
    assert m.match("Ryan Miller") is None  # ambiguous without a team
    assert m.match("Ryan Miller", "ne") == "MILLER_NE"
    assert m.match("Puka Nacua", "BUF") == "NACUA"  # a unique name ignores a stale team
    assert m.match("Brown") is None  # no partial matches for news
    assert m.match("Nobody") is None


def test_candidates_for_user_lookups():
    m = PlayerMatcher(PLAYERS)
    ids = lambda rows: [r["player_id"] for r in rows]  # noqa: E731
    assert ids(m.candidates("aj brown")) == ["BROWN"]  # exact beats the "st brown" substring
    assert ids(m.candidates("Brown")) == ["BROWN", "STBROWN"]
    assert ids(m.candidates("Puka", "LAR")) == ["NACUA"]  # FantasyPros abbreviation
    assert ids(m.candidates("Puka", "NE")) == []
    assert ids(m.candidates("Ryan Miller")) == ["MILLER_DAL", "MILLER_NE"]
    assert m.candidates("  ") == [] and m.candidates("...") == []


def test_match_one_word_finds_a_unique_first_or_last_name():
    m = PlayerMatcher(PLAYERS)
    assert m.match_one_word("Nacua") == m.match_one_word("puka") == "NACUA"
    assert m.match_one_word("Walker") == "WALKER"  # the suffix isn't the last name
    assert m.match_one_word("Brown") is None  # A.J. Brown and Amon-Ra St. Brown
    assert m.match_one_word("Brown", "DET") == "STBROWN"
    assert m.match_one_word("Miller") is None and m.match_one_word("Miller", "DAL") == "MILLER_DAL"
    assert m.match_one_word("Puka Nacua") is None  # full names go through match()
    assert m.match_one_word("") is None and m.match_one_word("Ra") is None  # only first and last names
