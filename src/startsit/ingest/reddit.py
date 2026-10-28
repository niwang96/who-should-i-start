"""Reddit crowd chatter (step 6b): comments from r/fantasyfootball's daily "Who Do I Start" threads.

Read through RSS, the one route that works from GitHub Actions (the .json endpoints answer 403
there), and politely: Reddit's unauthenticated RSS is tightly rate-limited per IP, so requests
are a minute apart, a 429 waits out Reddit's own reset once, and anything else that fails just
means no Reddit data this run (CLAUDE.md: optional, degrades gracefully).

Each run reads the newest day's WDIS threads (one per position: QB, RB, WR, Flex, K/TE/DEF),
found with the subreddit's search feed, and stores their comments. The LLM step that turns
comments into start/sit votes is `startsit.ingest.reddit_extract`.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.error
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from defusedxml.ElementTree import ParseError, fromstring

from startsit.db import Database, RedditCommentRow, utc_iso
from startsit.http import get_bytes
from startsit.ingest.news import ATOM, clean_text, parse_date

log = logging.getLogger("startsit.ingest.reddit")

SEARCH_URL = "https://www.reddit.com/r/fantasyfootball/search.rss?q=title%3AWDIS&restrict_sr=1&sort=new&limit=25"
# "Official: [WDIS RB] - Tue 10/27/2026"; the date is the thread's day (US Eastern).
THREAD_TITLE = re.compile(r"^Official: \[WDIS (?P<slot>[^\]]+)\] - \w{3} (?P<day>\d{1,2}/\d{1,2}/\d{4})$")
COMMENTS_LIMIT = 100  # comments per thread feed
SPACING = 60.0  # seconds between Reddit requests (12 s got a 429 in testing; 60 s worked)
MAX_RESET_WAIT = 120.0  # wait out a 429 only if Reddit says it resets within this many seconds
COMMENT_MAX_CHARS = 500
SKIP_AUTHORS = {"automoderator", "[deleted]"}
FETCH_EVERY_HOURS = 6  # at most one Reddit fetch per this many hours (each takes ~5 minutes)
RUN_SOURCE = "reddit:comments"  # ingest_runs source of a fetch


@dataclass(frozen=True)
class Thread:
    thread_id: str  # "t3_..."
    title: str
    url: str
    day: str  # MM/DD/YYYY from the title


def reddit_get(url: str, sleep: Callable[[float], None] = time.sleep) -> bytes | None:
    """The feed body, or None (logged) if Reddit refuses or fails; a 429 is retried once after its reset."""
    for attempt in range(2):
        try:
            return get_bytes(url)
        except urllib.error.HTTPError as e:
            reset = float(e.headers.get("x-ratelimit-reset") or e.headers.get("retry-after") or MAX_RESET_WAIT + 1)
            if e.code == 429 and attempt == 0 and reset <= MAX_RESET_WAIT:
                sleep(reset + 2)
                continue
            log.warning("reddit %s: HTTP %s", url, e.code)
            return None
        except OSError as e:
            log.warning("reddit %s: %s", url, e)
            return None
    return None  # pragma: no cover (the loop always returns)


def _entries(body: bytes) -> list[dict[str, str]]:
    """Atom entries as {id, title, link, author, content, published}; [] for a body that isn't a feed."""
    try:
        root = fromstring(body)
    except ParseError:
        return []
    out = []
    for e in root.iter(f"{ATOM}entry"):
        link = e.find(f"{ATOM}link")
        out.append(
            {
                "id": e.findtext(f"{ATOM}id", "").strip(),
                "title": e.findtext(f"{ATOM}title", "").strip(),
                "link": (link.get("href") or "") if link is not None else "",
                "author": e.findtext(f"{ATOM}author/{ATOM}name", "").strip(),
                "content": e.findtext(f"{ATOM}content", ""),
                "published": e.findtext(f"{ATOM}published") or e.findtext(f"{ATOM}updated") or "",
            }
        )
    return out


def latest_threads(search_body: bytes) -> list[Thread]:
    """The newest day's WDIS threads from the search feed (one per position slot)."""
    threads = [
        Thread(e["id"], e["title"], e["link"], m["day"])
        for e in _entries(search_body)
        if (m := THREAD_TITLE.match(e["title"])) and e["link"]
    ]
    if not threads:
        return []
    newest = max(threads, key=lambda t: datetime.strptime(t.day, "%m/%d/%Y")).day
    return list({t.thread_id: t for t in threads if t.day == newest}.values())  # deduped, in feed order


def comments(body: bytes, thread: Thread, now: datetime) -> list[RedditCommentRow]:
    """The thread's comments from its feed, minus the post itself, bots, deleted and empty ones."""
    rows: list[RedditCommentRow] = []
    for e in _entries(body):
        author = e["author"].removeprefix("/u/")
        text = clean_text(e["content"], COMMENT_MAX_CHARS)
        published = parse_date(e["published"])
        if not e["id"].startswith("t1_") or author.lower() in SKIP_AUTHORS or not text or published is None:
            continue  # the post itself is the t3_ entry
        rows.append(
            RedditCommentRow(
                comment_id=e["id"],
                thread_id=thread.thread_id,
                thread_title=thread.title,
                author=author or None,
                published_at=utc_iso(min(published, now)),
                text=text,
                url=e["link"],
                fetched_at=utc_iso(now),
            )
        )
    return rows


def comments_url(thread: Thread) -> str:
    return f"{thread.url.rstrip('/')}/.rss?limit={COMMENTS_LIMIT}"


def due(db: Database, now: datetime) -> bool:
    """Whether a fetch is due: none in the last FETCH_EVERY_HOURS hours."""
    last = db.last_run_at(RUN_SOURCE)
    if last is None:
        return True
    elapsed = now - datetime.fromisoformat(last).replace(tzinfo=UTC)
    return elapsed >= timedelta(hours=FETCH_EVERY_HOURS)


def fetch_comments(
    db: Database,
    season: int,
    now: datetime,
    get: Callable[[str], bytes | None] = reddit_get,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Store the comments of the newest day's WDIS threads; returns how many were stored."""
    body = get(SEARCH_URL)
    threads = latest_threads(body) if body else []
    stored = 0
    for thread in threads:
        sleep(SPACING)
        feed = get(comments_url(thread))
        if feed is None:
            break  # Reddit is refusing us: stop for this run rather than keep knocking
        stored += db.upsert("reddit_comments", comments(feed, thread, now))
    db.record_run(RUN_SOURCE, season, None, stored)
    log.info("reddit: %s threads, %s comments", len(threads), stored)
    return stored
