"""Shared fixtures.

`db` runs each test against in-memory SQLite and, when STARTSIT_TEST_DATABASE_URL points
at a disposable Postgres database (CI runs one), against Postgres too. The Postgres
database is wiped before every test.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from startsit.db import ALL_TABLE_NAMES, Database

PG_URL = os.environ.get("STARTSIT_TEST_DATABASE_URL")


@pytest.fixture(params=["sqlite", "postgres"])
def db(request: pytest.FixtureRequest) -> Iterator[Database]:
    if request.param == "sqlite":
        target = ":memory:"
    elif PG_URL:
        target = PG_URL
        with Database(PG_URL) as wipe:
            wipe.execute(f"DROP TABLE IF EXISTS {', '.join(ALL_TABLE_NAMES)}, keys_only")
    else:
        pytest.skip("STARTSIT_TEST_DATABASE_URL not set")
    with Database(target) as database:
        yield database
