"""Probe what a FantasyPros API key returns. Never prints the key.

Usage: FANTASYPROS_API_KEY=... python scripts/probe_fantasypros.py [--season 2026] [--week N]
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://api.fantasypros.com/public/v2/json/nfl"


def probe(key, label, path, params):
    url = f"{BASE}/{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"x-api-key": key, "User-Agent": "startsit-probe/0.1"})
    print(f"\n== {label}: {path} {params}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status, headers, body = resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as e:
        status, headers, body = e.code, e.headers, e.read()
    except urllib.error.URLError as e:
        print(f"   network error: {e.reason}")
        return
    print(f"   status {status}, {len(body)} bytes")
    for h, v in headers.items():
        if "limit" in h.lower() or "quota" in h.lower() or h.lower() == "retry-after":
            print(f"   header {h}: {v}")
    try:
        data = json.loads(body)
    except ValueError:
        print(f"   non-JSON body: {body[:200]!r}")
        return
    if not isinstance(data, dict):
        print(f"   top-level {type(data).__name__}, len {len(data)}")
        return
    print(f"   keys: {sorted(data)}")
    for k, v in data.items():
        if isinstance(v, list):
            print(f"   {k}: {len(v)} items")
            if v and isinstance(v[0], dict):
                print(f"     first item keys: {sorted(v[0])}")
                print(f"     first item: {json.dumps(v[0])[:400]}")
        elif not isinstance(v, dict):
            print(f"   {k}: {v!r}")
        else:
            print(f"   {k}: {json.dumps(v)[:300]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2026")
    ap.add_argument("--week", help="omit to let the API pick its default week")
    args = ap.parse_args()
    key = os.environ.get("FANTASYPROS_API_KEY")
    if not key:
        sys.exit("FANTASYPROS_API_KEY is not set")
    week = {"week": args.week} if args.week else {}
    s = args.season

    for scoring in ("PPR", "HALF", "STD"):
        probe(key, f"weekly ECR {scoring}", f"{s}/consensus-rankings",
              {"position": "ALL", "type": "weekly", "scoring": scoring, **week})
    probe(key, "weekly ECR RB PPR", f"{s}/consensus-rankings",
          {"position": "RB", "type": "weekly", "scoring": "PPR", **week})
    probe(key, "weekly projections", f"{s}/projections", {"position": "ALL", **week})
    probe(key, "news", "news", {"limit": 10})
    probe(key, "injuries", "injuries", {})
    probe(key, "players", "players", {})


if __name__ == "__main__":
    main()
