# Fantasy Football Start/Sit Assistant

## What this is
A "who should I start?" tool for fantasy football. A user enters 2+ players and their league's scoring format; the app combines expert consensus, usage data, recent news, and Reddit chatter, and an LLM returns a pick with a confidence level, the deciding factors, and links to the evidence it used.

Purpose: portfolio/resume project. It should have real users beyond the author (e.g., friends' leagues, fantasy subreddits).

## Hard constraints
1. **Free tiers only** for every service (LLM, hosting, DB, scheduling, data).
2. **An LLM must be a core part of the solution.**
3. **Usable by people other than the author.**

## Scope decisions
- **MVP is league-agnostic:** manual player entry + scoring format (PPR / half-PPR / standard). Works for any platform (Sleeper, ESPN, Yahoo).
- **Sleeper is dropped for now.** Possible v2: import a user's Sleeper league (roster, opponent, scoring, roster slots) and use Sleeper trending adds/drops as a crowd signal.
- **Not using** more expert ranking sites (ESPN/CBS/Yahoo): FantasyPros consensus already aggregates 100+ analysts, so more sites mostly re-count the same experts. Prefer *different kinds* of signals over more of the same kind.
- **Not scraping** JS-rendered pages like FantasyPros start/sit pages or reverse-engineering private APIs (e.g., Sleeper comments).

## Data sources

### FantasyPros API — expert consensus (+ news)
- Free public API for **personal, non-commercial use**; request a key at secure.fantasypros.com/api-keys/request with a description of intended use.
- Terms: must not build a product that directly competes with FantasyPros; must cite FantasyPros as the source; player images (Sportradar-licensed) can't be used.
- Free keys return at most 10 rows per call (tested Oct 24 2026; details under Open to-dos).
- In the access request, describe it as a free, non-commercial portfolio project that uses their consensus as one input alongside news/usage analysis.

### nflverse — usage and stats
- Free, open NFL data published as GitHub release assets (CSV/Parquet); Python loader: `nflreadpy`.
- Datasets of interest: weekly player stats, snap counts, depth charts, injuries, schedules, **FF Opportunity** (expected fantasy points from usage — a model-based signal independent of experts), **FF Rankings** (possible fallback for consensus; check contents/update cadence first).
- Maintained by community volunteers — no SLA. Batch data that updates after games, **not** a source for breaking news.

### RSS feeds — latest news
Candidate feeds (**verify each URL still resolves before building on it**):
- RotoWire latest NFL news — player-level fantasy blurbs; most useful
- ProFootballTalk — fast breaking news
- ESPN NFL, CBS Sports NFL — broader coverage/confirmation
- FantasyPros NFL — optional, may overlap with API
Store headline + summary + link only; link out to the source rather than fetching full articles.

### Reddit — crowd chatter (optional, must degrade gracefully)
- Unauthenticated `.json` endpoints (`/r/{sub}/new.json`, `/r/{sub}/top.json?t=week`, `/r/{sub}/comments/{id}.json`). Author has confirmed these work locally as of Oct 2026.
- Official Data API access is approval-gated and closing to new requests, so do not depend on it.
- Reference implementation pattern: https://github.com/niwang96/award-travel-updates (`RedditAccessor`, `RChurningSummaryAgent`) — find a recurring thread by title, pull top comments, filter out bots, return an empty list on any failure.
- For fantasy: find r/fantasyfootball's weekly start/sit and injury threads, pull top comments, filter for the queried player names.
- **Test from the deployed environment early** — datacenter IPs (GitHub Actions, Vercel, etc.) may be treated differently from a laptop.
- Be polite: descriptive User-Agent, caching, spaced-out comment fetches.

## Architecture

### Stage 1 — Ingest (scheduled job)
- Numbers go straight to the DB: FantasyPros rankings, nflverse usage. Code computes features (e.g., 3-week snap share trend, consensus rank by scoring format, expected vs. actual points).
- Text gets structured **once, at ingest time** by a cheap LLM call, then cached:
  - news → `{player, event, severity, date, source_url}`
  - Reddit comment → `{player, sentiment, claim, upvotes, source_url}`

### Stage 2 — Evidence packet (code, no LLM)
On a query, code assembles a compact per-player summary: consensus rank, usage trend, expected vs. actual points, latest structured news items, Reddit sentiment counts. Target a few hundred tokens, not raw articles.

### Stage 3 — Decide and explain (one LLM call)
- Code first computes a **simple baseline score** (e.g., consensus rank adjusted by usage trend).
- The LLM receives the evidence packet + baseline and returns structured output: pick, confidence, 2–3 deciding factors, and the IDs of evidence items used (for source links).
- If the LLM disagrees with the baseline, it must state why (e.g., "baseline favors B, but B was DNP Thursday").
- Never let the LLM state facts that aren't in the evidence packet.

### Stage 4 — Evaluate (after each week's games)
Log every recommendation; score against actual fantasy points. Report hit rates for: LLM pick vs. code baseline vs. raw FantasyPros consensus. This shows whether the LLM layer adds value.

## Caching (protects free-tier limits)
Follow the two-level pattern from the reference repo:
- **L1:** per-player evidence/summary cache with a TTL of a few hours.
- **L2:** permanent per-item LLM extraction cache keyed by source URL.
Many users will ask about the same players, so cache per-player results.

## Stack (decided)
- **Language:** Python everywhere (author is primarily a Java dev; Python chosen for the nflverse/LLM ecosystem and resume range).
- **Ingest:** Python jobs (`nflreadpy`) run by GitHub Actions cron, writing to Supabase.
- **Web app:** FastAPI on Vercel's free Python functions; simple HTMX/Jinja frontend (no JS framework for MVP).
- **DB:** Supabase (Postgres) via `DATABASE_URL` (session pooler URI; the direct connection is IPv6-only, which Actions/Vercel can't reach; the Claude Code cloud env can't reach Postgres at all). Local SQLite when unset. All SQL lives in `startsit/db.py` and is written once for both backends; tests run on both.
- **Code quality:** ruff (lint + format) and mypy run in CI alongside pytest.
- **LLM:** Gemini free tier, behind a provider interface so the model can be swapped. Unit tests use a fake provider and never call a real LLM.

## Secrets
- Read keys only from environment variables: `GEMINI_API_KEY`, `FANTASYPROS_API_KEY`. Never commit them, print them, or log them; keep `.env` gitignored and provide `.env.example`.
- Claude Code cloud env: keys are plain env vars (the network-secret types — Bearer/Body param/Basic/SigV4/GCP token — don't fit Gemini's `x-goog-api-key` header). The Gemini key there is a dev-only key with billing off.
- GitHub Actions: repository secrets. Vercel: project environment variables.

## Build order
1. Skeleton + nflverse ingest + usage features + Actions cron
2. ~~Baseline score + evidence packet~~ (done Oct 24 2026): FF Rankings as the consensus; baseline is 100% consensus for now (decided: add usage weighting only if the evaluation shows it helps; FantasyPros API add-on deferred)
3. RSS news + cached LLM extraction (L2 cache keyed by URL)
4. Stage 3 decide call with structured output, validated against packet evidence IDs
5. Web UI + recommendation logging
6. Weekly evaluation, then Reddit as an optional signal

## Open to-dos
- [x] Request FantasyPros API key (approved; key received)
- [x] Test what the free FantasyPros key returns (Oct 24 2026, from the Claude Code cloud env after allowlisting `api.fantasypros.com`; re-run with `scripts/probe_fantasypros.py`, which never prints the key):
  - Auth: `x-api-key` header. Base URL: `https://api.fantasypros.com/public/v2/json/nfl`. Every response includes `tier: "free"` and `limit: 10`.
  - **The free tier returns at most 10 rows per call.** `limit`, `offset`, `start` and `page` are ignored, so there's no paging. `count` still reports the full total (e.g., 378 for weekly FLX ECR).
  - `GET /2026/consensus-rankings?position=ALL|QB|RB|WR|TE|K|DST&type=weekly&scoring=PPR|HALF|STD&week=N`: all three scoring values work and are echoed back. An unknown scoring value silently falls back to `STD`. With no `week`, the API returns the current week (7 on Oct 24); `week=8` returns 0 players. Each row has `rank_ecr`, `pos_rank`, `rank_ave/min/max/std`, opponent, game status and ownership. Single-position calls also include `start_sit_grade` (e.g., "A+"), `tag` ("start"), and `r2p_pts` (projected points). No player-ID filter works here, so **ECR only covers the top 10 at each position**, which isn't enough for real start/sit questions.
  - `GET /2026/projections?position=WR&players=<fpid,...>`: the `players` filter works (with a single position, not `ALL`) and reaches beyond the top 10 (e.g., DeVonta Smith). Each row has `points`, `points_ppr` and `points_half` regardless of the `scoring` param. Some players came back empty (e.g., Rashee Rice, Emeka Egbuka); the cause is unknown, so handle a missing projection. This is the best usable per-player FantasyPros signal; `fpid` = `player_id`.
  - `GET /news`, `/injuries`, `/players`: 10 newest/top rows only, and no player or team filter worked. Injuries include `probability_of_playing` and Wed/Thu/Fri practice status (e.g., Limited/DNP), which is useful but limited to 10 players. News rows include title, description, `player_id` and a link.
  - **Rate limit:** no rate-limit headers are sent, but after ~75–80 calls in one session (Oct 24 2026) every endpoint returned `429 {"message":"Limit Exceeded"}`. That's an AWS API Gateway usage-plan quota (likely daily). Budget calls: ingest on a schedule and cache, never call per user query.
  - `GET /compare-players?players=<id>:<id>[:<id>]&position=WR|RB|...` (no season in the path; colon-delimited IDs): returns, for each scoring format (STD/PPR/HALF), **every expert's individual rank for each player**. This is the data behind FantasyPros' "Who Should I Start" tool, so code can compute "X of N experts start A over B". `position=ALL` returns empty lists. With PPR/HALF/STD, a non-top-10 player and a mixed-position pair still untested (hit the quota first).
  - Implication: nflverse FF Rankings stays the primary consensus source. Use FantasyPros per-player projections (and top-10 ECR/start-sit grade when available) as a supplementary, cited signal.
- [x] Verify `GEMINI_API_KEY` works from the Claude Code cloud env (Oct 24 2026: `gemini-3.5-flash` OK; `gemini-2.5-flash` returns 404 "no longer available to new users")
- [ ] Verify RSS feed URLs
- [ ] Test Reddit `.json` access from the deployed environment
- [x] FF Opportunity: 2026 data through the latest week; expected points are PPR-scored (derive half/std via `receptions_exp`)
- [x] FF Rankings weekly (`fp_latest_weekly.csv`, Oct 24 2026): `load_ff_rankings('week')` 403s here because the proxy blocks `github.com/.../raw/...`, but the same file loads from `https://raw.githubusercontent.com/dynastyprocess/data/master/files/fp_latest_weekly.csv`. It's a scrape of FantasyPros' full weekly ECR for **every** ranked player (1006 rows, `scrape_date` 2026-10-23), with `ecr`, `sd`, `best`, `worst`, `pos_rank`, `start_sit_grade`, `r2p_pts`, opponent and game status. Pages: `qb`, `ppr-rb`, `ppr-wr`, `ppr-te`, `k`, `dst` (+ IDP). **PPR only** for skill positions, so no half/std ECR. Keyed by `fantasypros_id` (same as the API `player_id`). It's FantasyPros data, so cite FantasyPros. Ingested by `startsit.ingest.rankings` (archived per `scrape_date`; week inferred from kickoff dates vs. the schedule, since the file has no week column). Still to check: update cadence across the week.
- [ ] nflverse injuries lag: on Oct 24 (Sat) week 7 rows had practice status but no game status yet (e.g., Puka Nacua "Limited" in nflverse while the FantasyPros API showed Friday DNP + Questionable). Status zeroing in the baseline only fires once nflverse posts "Out". Consider the FantasyPros `/injuries` endpoint (top 10 only) or RSS news (step 3) for fresher status.
- [ ] Deferred FantasyPros API add-on: per-player projections in all three formats (fills the PPR-only ECR gap) + `compare-players` expert head-to-head; needs a call budget under the ~75-call quota.
- [x] Gemini free-tier limits (author's AI Studio, Oct 24 2026): **1 request/minute, 20 requests/day**, 250k input tokens/minute. Requests, not tokens, are the constraint: 20/day can't cover one LLM call per news item or per user question. Design implications:
  - Step 3: batch news extraction: one call carries all new headlines (fits easily in 250k tokens), so ingest uses ~1–3 calls/day.
  - Step 4: cache decisions per (players, scoring, week) and fall back to the code baseline with a clear "LLM quota reached" note when the daily budget is spent.
  - Still to check: whether a lighter model (e.g. a flash-lite variant) on the same key has higher limits; the provider interface allows using a different model for extraction than for decisions.
- [x] Draft evidence packet schema (`startsit.evidence`: numbered items E1..En with kind/text/source_url; `to_text()` is what the LLM sees)
- [ ] Draft Stage 3 prompt
- Decided Oct 24 2026: the baseline zeroes every player whose weekly roster status isn't Active (reserve/IR, practice squad, exempt, inactive, released, retired), not just cut/retired. nflverse can lag an activation by a day or two; the packet's roster-status item shows the LLM why.
- Decided Oct 24 2026: the "current week" is the latest week with rankings, so after Monday night the app keeps answering for the finished week until the new week's rankings arrive (normally Tuesday). Step 5's UI should say which week it's answering for.
- [ ] Step 5: store the full evidence packet (text + item ids) with each logged recommendation. Injury reports and roster status are overwritten as they update, so the packet is the only record of what the user and LLM saw.
- [ ] Confirm the Supabase connection via a manual run of the ingest workflow (needs `ingest.yml` on `main`). The Claude Code cloud env **can't** reach Postgres (Oct 24 2026): its proxy carries only HTTP/HTTPS, and raw-TCP databases are unsupported, so allowlisting doesn't help. Development and tests there use SQLite and a local Postgres; Supabase is reached only from GitHub Actions, Vercel, or a laptop. `DATABASE_URL` must percent-encode special characters in the password (e.g. `#` → `%23`), or the value gets cut off when saved.

## Timing
The NFL regular season runs through early January. Shipping an MVP within ~2–3 weeks leaves time to collect real usage and several weeks of evaluation data.
