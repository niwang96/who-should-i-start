"""RSS/Atom news ingest: headline, summary and link per item (CLAUDE.md: link out, never copy articles).

Fetching is separate from parsing so parsing is tested with saved feed text and no network.
A feed that fails (blocked, down, malformed) is logged and skipped; the others still load.
"""

from __future__ import annotations

import html
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from xml.etree.ElementTree import Element
from zoneinfo import ZoneInfo

from defusedxml.ElementTree import fromstring

from startsit.db import NewsItemRow, utc_iso
from startsit.http import get_bytes

log = logging.getLogger("startsit.ingest.news")

SUMMARY_MAX_CHARS = 400
ATOM = "{http://www.w3.org/2005/Atom}"


@dataclass(frozen=True)
class Feed:
    source: str  # short display name, shown with each fact
    url: str


# Verified Oct 24 2026. RotoWire's public feed only ever lists its latest 5 player blurbs (no
# paging or filters), so it's polled every 15 minutes at peak times. ESPN's feed was dropped:
# it answers GitHub Actions (datacenter IPs) with an empty bot-check response.
FEEDS = [
    Feed("RotoWire", "https://www.rotowire.com/rss/news.php?sport=NFL"),
    Feed("ProFootballTalk", "https://www.nbcsports.com/profootballtalk.rss"),
    Feed("CBS Sports", "https://www.cbssports.com/rss/headlines/nfl/"),
]

# US zone abbreviations, read as local time in that region: ESPN labels times "EST" even
# during daylight time, so the S/D letter can't be trusted.
US_ZONES = {
    **dict.fromkeys(("EST", "EDT", "ET"), "America/New_York"),
    **dict.fromkeys(("CST", "CDT", "CT"), "America/Chicago"),
    **dict.fromkeys(("MST", "MDT", "MT"), "America/Denver"),
    **dict.fromkeys(("PST", "PDT", "PT"), "America/Los_Angeles"),
}
# "Sat, 24 Oct 2026 16:26:40 EST" or RotoWire's 12-hour "Sat, 24 Oct 2026 1:47:00 PM PDT".
_US_LOCAL = re.compile(r"^(?:\w{3}, )?(\d{1,2} \w{3} \d{4} \d{1,2}:\d{2}:\d{2})( [AP]M)? ([A-Z]{2,3})$")


def fetch(feed: Feed, timeout: float = 30) -> bytes:
    body = get_bytes(feed.url, timeout=timeout)
    if not body.strip():
        # ESPN answered GitHub Actions (datacenter IPs) with an empty 200 response; fail loudly if another does.
        raise ValueError("empty response")
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
    """RFC 822 (RSS), US-local ("... 1:47:00 PM PDT", "... 16:26:40 EST") or ISO 8601 (Atom) dates, as UTC."""
    if not raw:
        return None
    raw = raw.strip()
    # Checked first: email.utils misreads "1:47:00 PM PDT" as 01:47 UTC and takes "EST" literally.
    m = _US_LOCAL.match(raw)
    if m and m[3] in US_ZONES:
        fmt = "%d %b %Y %I:%M:%S %p" if m[2] else "%d %b %Y %H:%M:%S"
        local = datetime.strptime(m[1] + (m[2] or ""), fmt)
        return local.replace(tzinfo=ZoneInfo(US_ZONES[m[3]])).astimezone(UTC)
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


@dataclass(frozen=True)
class _Entry:
    guid: str | None
    title: str | None
    link: str | None
    summary: str | None
    date: str | None


def _entries(root: Element) -> Iterable[_Entry]:
    """Each RSS item or Atom entry, with its fields as raw text."""
    for item in root.iter("item"):
        yield _Entry(
            _text(item, "guid"),
            _text(item, "title"),
            _text(item, "link"),
            _text(item, "description"),
            _text(item, "pubDate"),
        )
    for entry in root.iter(f"{ATOM}entry"):
        link_el = entry.find(f"{ATOM}link")
        yield _Entry(
            _text(entry, f"{ATOM}id"),
            _text(entry, f"{ATOM}title"),
            link_el.get("href") if link_el is not None else None,
            _text(entry, f"{ATOM}summary") or _text(entry, f"{ATOM}content"),
            _text(entry, f"{ATOM}published") or _text(entry, f"{ATOM}updated"),
        )


def parse_feed(feed: Feed, body: bytes, fetched_at: datetime) -> list[NewsItemRow]:
    """News items from one feed body; items without a link, title or parseable date are dropped."""
    root = fromstring(body)
    fetched = utc_iso(fetched_at)
    items: dict[str, NewsItemRow] = {}  # by item_id; a repeated id keeps the last copy
    for e in _entries(root):
        published = parse_date(e.date)
        if not (e.link and e.title and published):
            continue
        published = min(published, fetched_at.astimezone(UTC))  # a feed's clock can't be ahead of ours
        url = e.link.strip()
        guid = (e.guid or "").strip()
        item_id = f"{feed.source}:{guid}" if guid else url
        items[item_id] = NewsItemRow(
            item_id=item_id,
            url=url,
            source=feed.source,
            title=clean_text(e.title),
            summary=clean_text(e.summary, SUMMARY_MAX_CHARS) or None,
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
