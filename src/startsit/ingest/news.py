"""RSS/Atom news ingest: headline, summary and link per item (CLAUDE.md: link out, never copy articles).

Fetching is separate from parsing so parsing is tested with saved feed text and no network.
A feed that fails (blocked, down, malformed) is logged and skipped; the others still load.
"""

from __future__ import annotations

import html
import logging
import re
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from xml.etree.ElementTree import Element

from defusedxml.ElementTree import fromstring

from startsit.db import NewsItemRow, utc_iso

log = logging.getLogger("startsit.ingest.news")

USER_AGENT = "who-should-i-start/0.1 (+https://github.com/niwang96/who-should-i-start)"
SUMMARY_MAX_CHARS = 400
ATOM = "{http://www.w3.org/2005/Atom}"


@dataclass(frozen=True)
class Feed:
    source: str  # short display name, shown with each fact
    url: str


FEEDS = [
    Feed("RotoWire", "https://www.rotowire.com/rss/news.php?sport=NFL"),
    Feed("ProFootballTalk", "https://profootballtalk.nbcsports.com/feed/"),
    Feed("ESPN", "https://www.espn.com/espn/rss/nfl/news"),
    Feed("CBS Sports", "https://www.cbssports.com/rss/headlines/nfl/"),
]


def fetch(feed: Feed, timeout: float = 30) -> bytes:
    req = urllib.request.Request(feed.url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body: bytes = resp.read()
    return body


def fetch_all(feeds: Iterable[Feed] = FEEDS) -> dict[Feed, bytes]:
    """Each feed's raw body; feeds that fail are logged and left out."""
    bodies = {}
    for feed in feeds:
        try:
            bodies[feed] = fetch(feed)
        except Exception as e:  # any failure just skips this feed
            log.warning("news feed %s failed: %s", feed.source, e)
    return bodies


def clean_text(raw: str | None, max_chars: int | None = None) -> str:
    """Strip HTML tags and entities, collapse whitespace, and optionally truncate on a word boundary."""
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw or ""))).strip()
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)  # "Friday ." left by a removed tag
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return text


def parse_date(raw: str | None) -> datetime | None:
    """RFC 822 (RSS) or ISO 8601 (Atom) dates, as UTC."""
    if not raw:
        return None
    raw = raw.strip()
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    return parsed.replace(tzinfo=parsed.tzinfo or UTC).astimezone(UTC)


def _text(el: Element, tag: str) -> str | None:
    child = el.find(tag)
    return child.text if child is not None else None


def _entries(root: Element) -> Iterable[tuple[str | None, str | None, str | None, str | None]]:
    """(title, link, summary, date) for each RSS item or Atom entry."""
    for item in root.iter("item"):
        yield _text(item, "title"), _text(item, "link"), _text(item, "description"), _text(item, "pubDate")
    for entry in root.iter(f"{ATOM}entry"):
        link_el = entry.find(f"{ATOM}link")
        link = link_el.get("href") if link_el is not None else None
        summary = _text(entry, f"{ATOM}summary") or _text(entry, f"{ATOM}content")
        date = _text(entry, f"{ATOM}published") or _text(entry, f"{ATOM}updated")
        yield _text(entry, f"{ATOM}title"), link, summary, date


def parse_feed(feed: Feed, body: bytes, fetched_at: datetime) -> list[NewsItemRow]:
    """News items from one feed body; items without a link, title or parseable date are dropped."""
    root = fromstring(body)
    fetched = utc_iso(fetched_at)
    items: dict[str, NewsItemRow] = {}
    for title, link, summary, date in _entries(root):
        published = parse_date(date)
        if not (link and title and published):
            continue
        url = link.strip()
        items[url] = NewsItemRow(
            url=url,
            source=feed.source,
            title=clean_text(title),
            summary=clean_text(summary, SUMMARY_MAX_CHARS) or None,
            published_at=utc_iso(published),
            fetched_at=fetched,
        )
    return list(items.values())


def build_news_items(bodies: dict[Feed, bytes], fetched_at: datetime) -> list[NewsItemRow]:
    items: list[NewsItemRow] = []
    for feed, body in bodies.items():
        try:
            parsed = parse_feed(feed, body, fetched_at)
        except Exception as e:  # malformed XML: skip this feed, keep the rest
            log.warning("news feed %s could not be parsed: %s", feed.source, e)
            continue
        log.info("news feed %s: %s items", feed.source, len(parsed))
        items.extend(parsed)
    return items
