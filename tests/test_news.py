from datetime import UTC, datetime

import pytest

from startsit.ingest import news

FETCHED = datetime(2026, 10, 24, 20, 0, tzinfo=UTC)
FEED = news.Feed("RotoWire", "https://example.com/rss")

RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>NFL News</title>
<item>
  <title>Puka Nacua (calf) questionable for Sunday</title>
  <link> https://example.com/news/1 </link>
  <description><![CDATA[<p>Nacua didn&#39;t practice <b>Friday</b>.</p>]]></description>
  <pubDate>Fri, 23 Oct 2026 21:32:15 -0400</pubDate>
</item>
<item>
  <title>No date, dropped</title>
  <link>https://example.com/news/2</link>
</item>
<item>
  <title>Duplicate link keeps the last copy</title>
  <link>https://example.com/news/1</link>
  <pubDate>Sat, 24 Oct 2026 10:00:00 GMT</pubDate>
</item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>PFT</title>
<entry>
  <title>Breece Hall returns to practice</title>
  <link href="https://example.com/pft/9"/>
  <summary>Hall was limited.</summary>
  <published>2026-10-22T15:00:00Z</published>
</entry>
</feed>"""


def test_parse_rss_cleans_text_converts_dates_and_drops_bad_items():
    items = news.parse_feed(FEED, RSS, FETCHED)
    assert [i["url"] for i in items] == ["https://example.com/news/1"]
    item = items[0]
    assert item["title"] == "Duplicate link keeps the last copy"
    assert item["published_at"] == "2026-10-24T10:00:00+00:00"
    assert item["fetched_at"] == "2026-10-24T20:00:00+00:00"
    assert item["source"] == "RotoWire" and item["summary"] is None


def test_parse_rss_strips_html_and_converts_timezones():
    first = news.parse_feed(
        FEED, RSS.replace(b"news/1</link>\n  <pubDate>Sat", b"news/3</link>\n  <pubDate>Sat"), FETCHED
    )
    nacua = next(i for i in first if i["url"] == "https://example.com/news/1")
    assert nacua["summary"] == "Nacua didn't practice Friday."
    assert nacua["published_at"] == "2026-10-24T01:32:15+00:00"  # -0400 converted to UTC


def test_parse_atom():
    [item] = news.parse_feed(news.Feed("PFT", "https://example.com/atom"), ATOM, FETCHED)
    assert item == {
        "item_id": "https://example.com/pft/9",  # no id in the entry: the link
        "url": "https://example.com/pft/9",
        "source": "PFT",
        "title": "Breece Hall returns to practice",
        "summary": "Hall was limited.",
        "published_at": "2026-10-22T15:00:00+00:00",
        "fetched_at": "2026-10-24T20:00:00+00:00",
    }


def test_clean_text_truncates_on_a_word_boundary():
    assert news.clean_text("one two three four", max_chars=9) == "one two…"
    assert news.clean_text(None) == ""


def test_build_news_items_skips_a_malformed_feed(caplog):
    bodies = {FEED: RSS, news.Feed("Broken", "https://example.com/broken"): b"<rss><channel>"}
    items = news.build_news_items(bodies, FETCHED)
    assert len(items) == 1
    assert "news feed Broken could not be parsed" in caplog.text


def test_entity_expansion_is_refused():
    bomb = (
        b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]>'
        b"<rss><channel><item><title>&a;</title></item></channel></rss>"
    )
    assert news.build_news_items({FEED: bomb}, FETCHED) == []


ROTOWIRE = b"""<rss version="2.0"><channel>
<item><guid>nfl644147</guid><title>Christian Kirk: Activated from injured reserve</title>
  <link>https://www.rotowire.com/football/player/christian-kirk-12508</link>
  <pubDate>Sat, 24 Oct 2026 1:47:00 PM PDT</pubDate></item>
<item><guid>nfl644001</guid><title>Christian Kirk: Questionable for Sunday</title>
  <link>https://www.rotowire.com/football/player/christian-kirk-12508</link>
  <pubDate>Fri, 23 Oct 2026 11:05:00 AM PDT</pubDate></item>
</channel></rss>"""


def test_rotowire_items_sharing_a_player_link_are_kept_apart_by_guid():
    later = datetime(2026, 10, 24, 23, 0, tzinfo=UTC)
    items = news.parse_feed(news.Feed("RotoWire", "https://www.rotowire.com/rss"), ROTOWIRE, later)
    assert [(i["item_id"], i["published_at"]) for i in items] == [
        ("RotoWire:nfl644147", "2026-10-24T20:47:00+00:00"),
        ("RotoWire:nfl644001", "2026-10-23T18:05:00+00:00"),
    ]
    assert {i["url"] for i in items} == {"https://www.rotowire.com/football/player/christian-kirk-12508"}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Sat, 24 Oct 2026 1:47:00 PM PDT", "2026-10-24T20:47:00+00:00"),
        ("Sat, 24 Oct 2026 12:05:00 AM EDT", "2026-10-24T04:05:00+00:00"),
        ("Sat, 24 Oct 2026 16:26:40 EST", "2026-10-24T20:26:40+00:00"),  # ESPN's "EST" in daylight time
        ("Sat, 05 Dec 2026 16:26:40 EST", "2026-12-05T21:26:40+00:00"),
        ("Sat, 24 Oct 2026 20:19:56 +0000", "2026-10-24T20:19:56+00:00"),
        ("2026-10-22T15:00:00Z", "2026-10-22T15:00:00+00:00"),
    ],
)
def test_parse_date_formats(raw, expected):
    assert news.parse_date(raw).isoformat() == expected


def test_parse_date_rejects_garbage():
    assert news.parse_date("next Tuesday") is None and news.parse_date(None) is None


def test_published_time_is_capped_at_fetch_time():
    future = RSS.replace(b"Sat, 24 Oct 2026 10:00:00 GMT", b"Sun, 25 Oct 2026 10:00:00 GMT")
    [item] = news.parse_feed(FEED, future, FETCHED)
    assert item["published_at"] == "2026-10-24T20:00:00+00:00"
