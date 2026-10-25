"""Plain HTTP GETs for the ingest jobs, with one identifying User-Agent (be polite to free sources)."""

from __future__ import annotations

import urllib.request
from collections.abc import Mapping

USER_AGENT = "who-should-i-start/0.1 (+https://github.com/niwang96/who-should-i-start)"


def get_bytes(url: str, headers: Mapping[str, str] | None = None, timeout: float = 30) -> bytes:
    """The response body; HTTP errors raise urllib.error.HTTPError (callers handle the codes they expect)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body: bytes = resp.read()
    return body
