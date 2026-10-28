"""Player-name matching, shared by user lookups and news extraction.

Names are compared normalized (lowercase, no punctuation or generational suffix), so
"AJ Brown", "A.J. Brown" and "a.j. brown" all find the same player, and "Kenneth Walker"
finds "Kenneth Walker III".
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence

from startsit.db import PlayerRow

# Abbreviations news sources and FantasyPros use that differ from nflverse's.
TEAM_ALIASES = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU"}
_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def normalize_name(name: str) -> str:
    """Lowercase, no punctuation or generational suffix: "Kenneth Walker III" -> "kenneth walker"."""
    name = re.sub(r"[^a-z ]", "", name.lower().replace("-", " "))
    return re.sub(r"\s+", " ", _SUFFIX.sub("", name)).strip()


def normalize_team(team: str) -> str:
    return TEAM_ALIASES.get(team.strip().upper(), team.strip().upper())


class PlayerMatcher:
    """Finds players by name, optionally narrowed by team."""

    def __init__(self, players: Sequence[PlayerRow]):
        self._by_name: dict[str, list[PlayerRow]] = defaultdict(list)
        for p in players:
            self._by_name[normalize_name(p["name"])].append(p)

    @property
    def players(self) -> list[PlayerRow]:
        """Every player, by name then team (e.g. for a suggestions list)."""
        return sorted((p for ps in self._by_name.values() for p in ps), key=lambda p: (p["name"], p["team"] or ""))

    def candidates(self, name: str, team: str | None = None) -> list[PlayerRow]:
        """Exact normalized-name matches, else names containing the query; only on `team`, if given."""
        query = normalize_name(name)
        if not query:
            return []
        found = self._by_name.get(query) or [p for key, ps in self._by_name.items() if query in key for p in ps]
        if team:
            found = [p for p in found if (p["team"] or "").upper() == normalize_team(team)]
        return sorted(found, key=lambda p: (p["name"], p["team"] or ""))

    def match(self, name: str, team: str = "") -> str | None:
        """The one player with exactly this name (team breaks ties only), else None.

        For news: the team isn't required to match a single candidate, because news can
        name a player's old team the day of a trade.
        """
        exact = self._by_name.get(normalize_name(name), [])
        return self._one(exact, team)

    def match_one_word(self, name: str, team: str = "") -> str | None:
        """The one player whose first or last name is this single word ("Egbuka", "Puka"), else None.

        For forum comments, which name players by one name. Build the matcher on a small pool
        (e.g. the week's ranked players), or common names match nobody.
        """
        word = normalize_name(name)
        if not word or " " in word:
            return None
        found = [p for key, ps in self._by_name.items() if word in (key.split()[0], key.split()[-1]) for p in ps]
        return self._one(found, team)

    @staticmethod
    def _one(found: list[PlayerRow], team: str) -> str | None:
        if len(found) > 1 and team:
            found = [p for p in found if (p["team"] or "").upper() == normalize_team(team)]
        return found[0]["player_id"] if len(found) == 1 else None
