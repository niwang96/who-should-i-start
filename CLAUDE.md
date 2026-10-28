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
- Unauthenticated `.json` endpoints (`/r/{sub}/new.json`, `/r/{sub}/top.json?t=week`, `/r/{sub}/comments/{id}.json`). Author has confirmed these work locally as of Oct 2026. **From GitHub Actions they're blocked (403); RSS works there** (tested Oct 27 2026, see Open to-dos), so step 6 uses RSS.
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
  - Reddit comment → `{player, sentiment, claim, source_url}` (RSS has no upvotes)

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
- **Ingest:** Python jobs (`nflreadpy`) run by GitHub Actions cron, writing to Supabase. The repo is **public** (since Oct 25 2026), so Actions minutes are free; GitHub disables scheduled workflows in a public repo after 60 days without repo activity, so re-enable `ingest.yml` if that happens in the off-season.
- **Web app:** FastAPI on Vercel's free Python functions; simple HTMX/Jinja frontend (no JS framework for MVP).
- **DB:** New columns are added to existing tables automatically on connect (`Database._add_missing_columns`; they must be nullable). Supabase (Postgres) via `DATABASE_URL` (session pooler URI; the direct connection is IPv6-only, which Actions/Vercel can't reach; the Claude Code cloud env can't reach Postgres at all). Local SQLite when unset. All SQL lives in `startsit/db.py` and is written once for both backends; tests run on both.
- **Code quality:** ruff (lint + format) and mypy run in CI alongside pytest; CI fails below **95% branch coverage** (`pyproject.toml`). Live integration tests (`tests/test_live.py`, marker `live`, deselected by default) hit the real services with a minimal quota (2–4 Gemini calls, 1 FantasyPros call) via `integration.yml`: only on merges to `main` touching source adapters, and by hand (Actions → integration → Run workflow, any branch). No schedule: the ingest jobs already call every service.
- **Web security** (review Oct 27 2026):
  - All SQL values are parameterized; f-strings only splice in table/column names from code.
  - Templates autoescape. Source links are rendered only if http(s) (`web.app.safe_href`), behind a strict CSP.
  - Cross-site form posts are refused (`Sec-Fetch-Site: cross-site` → 403), so another page can't make its visitors spend the AI budget.
  - Visitors are counted per IPv4 address and per IPv6 /64 (`llm.visitor_network`), since one home gets a whole /64.
  - Only the first few form fields are read.
  - Workflow inputs reach shells only through env vars; the ingest job's token is `contents: read`.
  - Accepted risk: news text from feeds reaches the LLM prompts (prompt injection). Its effect is bounded: picks must be one of the asked players, every number must be in the packet, and the global daily budget caps abuse.
- **Access:** the CLIs (`python -m startsit.ingest`, `python -m startsit.decide` for debugging) aren't network endpoints; they reach production only with the secrets. Secrets live only in GitHub Actions secrets, Vercel env vars and the Claude Code env. On Postgres all tables have RLS enabled with no policies (set on connect), which blocks Supabase's REST API; the app connects as the tables' owner.
- **LLM:** Gemini free tier, model **`gemini-3.5-flash-lite`** (decided Oct 24 2026; 500 requests/day vs 20 for `gemini-3.5-flash`). Read the model id from `GEMINI_MODEL` (default `gemini-3.5-flash-lite`) behind a provider interface so it can be swapped. Newer Flash versions listed on the key (3.6/3.7/3.8) aren't usable for this project. Unit tests use a fake provider and never call a real LLM.

## Secrets
- Read keys only from environment variables: `GEMINI_API_KEY`, `FANTASYPROS_API_KEY`. Never commit them, print them, or log them; keep `.env` gitignored and provide `.env.example`.
- Claude Code cloud env: keys are plain env vars (the network-secret types — Bearer/Body param/Basic/SigV4/GCP token — don't fit Gemini's `x-goog-api-key` header). The Gemini key there is a dev-only key with billing off.
- GitHub Actions: repository secrets. Vercel: project environment variables.

## Build order
1. Skeleton + nflverse ingest + usage features + Actions cron
2. ~~Baseline score + evidence packet~~ (done Oct 24 2026): FF Rankings as the consensus; baseline is 100% consensus for now (decided: add usage weighting only if the evaluation shows it helps; FantasyPros API add-on deferred)
3. ~~RSS news + cached LLM extraction~~ (built Oct 24 2026): `startsit/llm.py` (Gemini REST provider, fake provider, `BudgetedLLM` with per-purpose daily budgets logged in `llm_calls`), `startsit/ingest/news.py` (RSS/Atom via defusedxml), `startsit/ingest/news_extract.py` (batches of 40, name matching, cache in `news_extractions` keyed by item id = feed guid, else link), and `news` items in the evidence packet (last 7 days, up to 3 per player, cut off at the question's `as_of`). News budget 40 calls/day; runs hourly in `ingest.yml`.
4. ~~Stage 3 decide call~~ (built Oct 24 2026): `startsit/decide.py`. One Gemini call per question (players labeled P1/P2...), JSON schema with pick, confidence, 2–3 factors citing evidence ids and a required `baseline_disagreement` when the pick differs. Code validates the pick, every cited id and the disagreement, retries once with the problems, then falls back to the baseline with a note (also on quota/LLM errors). Validated answers are cached in `decision_cache` by a hash of model + `PROMPT_VERSION` + packet text. Decide budget: 400 calls/day. Betting lines (spread, total → implied team points) from the nflverse schedule were added as `matchup` evidence.
4b. ~~Hardening before the web app~~ (built Oct 25 2026):
   - FantasyPros projections daily (`startsit/ingest/projections.py`): the top QB/RB/WR/TE by consensus rank (36/72/96/36 → ~26 calls, 2 s apart; ~240 players). The baseline converts PPR to half/standard with FantasyPros' projected receptions, falling back to usage-based expected catches.
   - News facts carry a `game_status` (out/doubtful/questionable/active). This week's newest status (after the team's previous game) zeroes a player ruled out, and "active" overrides a stale non-active roster status.
   - A pick that's questionable/doubtful (news or injury report) is capped at medium confidence in code, with a note.
   - News runs hourly and re-checks the rankings file. (Superseded Oct 25 2026, see 5b: the 15-minute fetch-only game-day polling was replaced by targeted pre-kickoff news runs.) ESPN was dropped from the feeds (bot check from Actions).
4c. ~~Data audit + review fixes~~ (Oct 25 2026). Audit: all 441 stored rankings match the source file; usage features recomputed from raw nflverse match; implied points check out; 12 live Gemini decisions had 0 numbers not in the packet. Without FantasyPros projections, half-PPR picks flip in 0.2% of clear (1+ pt) matchups and standard in 3.3% (usage-based catches are noisier: Puka Nacua 9.9 vs FantasyPros 5.9). Fixes:
   - Player lookup uses normalized names (`startsit/names.py`, shared with news matching), so "AJ Brown" finds "A.J. Brown".
   - The schedule's kickoff time (`games.gametime`, US Eastern) is in the matchup fact, and `PlayerEvidence.game_started` flags games already kicked off at `as_of` (the UI must say so; e.g. Thursday games on Sunday).
   - A news game status only counts once the team's previous game is over (kickoff + 4 h), so an in-game "won't return" from Monday night doesn't zero next week.
   - News extraction drops role speculation ("may soon start") via the prompt plus a code filter on may/might/could for role facts without a game status.
   - FantasyPros projections are fetched in rank order across positions, so a quota cut-off loses the lowest-ranked players everywhere instead of a whole position (the first post-4b run got 179 of ~240 before a 429).
   - Decisions require 2–3 factors (matching the prompt); shared HTTP GET with one User-Agent (`startsit/http.py`).
4d. ~~Free-tier protections~~ (Oct 25 2026), so step 5 is only the web app. `BudgetedLLM` checks, before every Gemini call, by counting `llm_calls` (now with `called_at` written by code and a nullable `visitor` column): the purpose's daily budget (decide 400, news 40; news raised to 60 in 5b), a **shared 12 calls/minute** cap across ingest and web (Gemini allows 15), and per-visitor limits for decisions (`decide.VISITOR_LIMITS`: 6 new AI answers per 10 min, 25/day; cached answers are free). Visitors are a salted hash of the IP (`llm.visitor_id`, `VISITOR_SALT`). Each limit raises its own error (`QuotaExceeded` daily, `RateLimited` per minute, `VisitorLimited`) and the baseline fallback says which; a Gemini 429 naming a per-day quota maps to daily, anything else to per-minute. The web app builds one `decide.decide_llm(db, visitor)` per request. Same PR: 95% coverage gate, live integration tests, `startsit.evidence` CLI removed (`decide --no-llm` prints the packet), RLS on every Postgres table. Count-then-call can overshoot slightly under concurrency, hence the headroom.
5. ~~Web UI + recommendation logging~~ (built Oct 25 2026): `startsit/web/` (FastAPI + Jinja + htmx 2.0.4 vendored in `static/`, strict same-origin CSP, works without JS), `api/index.py` + `vercel.json` for Vercel (no rewrites: Vercel's FastAPI support routes every path to the app; the first deploy's rewrite to `/api/index` made every page a 404). Shows the week answered for and data freshness, player suggestions (the full list in a `<datalist>`, filtered by the browser: no requests while typing, and `static/app.js` stops Enter from submitting, so only the button asks), pick/confidence/factors with evidence links, all evidence with source links, and a "game already kicked off" warning. Each request builds `decide_llm(db, visitor_id(ip))`; every answer goes to `recommendations` with the packet text and JSON (`decide.record_recommendation`). Deployment: base install only (polars/nflreadpy moved to the `ingest` extra; web install ~57 MB), connections use `prepare_threshold=None` for the transaction pooler, and the web app creates tables only on its first connection per process. Supabase is in AWS us-west-2, so the Vercel function region is `pdx1` (set in `vercel.json`).
5b. ~~Reliable ingest schedule~~ (Oct 25 2026). GitHub delayed or dropped most scheduled runs on Sunday Oct 25 (7 runs in 20 h instead of ~40; the 10:17 daily run never ran). Fixes: every scheduled run is a `news` run, which also does the daily stats refresh when it's due (`ingest.__main__.stats_due`: none since the latest 10:00 UTC), so the first run to get through after 10:00 UTC does it. The 15-minute fetch-only game-day polling (mostly after kickoff, extraction delayed to the next hourly run, and mostly dropped) was replaced by full news runs when news matters for lineups: Sunday every 30 min ~8am–1:30pm ET plus twice before the 4pm games, and twice before Thursday/Monday night games (UTC crons cover both EDT and EST; minutes avoid :00). News budget raised to 60 calls/day (400 + 60 of Gemini's 500). If runs still go missing, the fallback is an external trigger (cron-job.org calling `workflow_dispatch` with a repo-scoped token).
   - **Update (Oct 28 2026): the external trigger is in place.** GitHub kept dropping scheduled runs: on Oct 27, only 1 of ~9 ran between 03:00 and 10:50 UTC, and the 10:17 daily refresh was skipped.
   - **cron-job.org** (the author's account) POSTs to `https://api.github.com/repos/niwang96/who-should-i-start/actions/workflows/ingest.yml/dispatches` with body `{"ref":"main","inputs":{"only":"news"}}`.
   - **Headers:** `Authorization: Bearer <token>`, `Accept: application/vnd.github+json`, `X-GitHub-Api-Version: 2022-11-28`, `Content-Type: application/json`, `User-Agent`.
   - **Token:** a fine-grained token limited to this repo, with **Actions: Read and write** only. The first test got a 403 until that permission was set. It expires after the season; renew it then.
   - **Jobs (America/New_York):** hourly at :52, and Sundays 8–13h at :22 and :52.
   - **GitHub's own schedule stays as a backup.** Duplicate runs are harmless: the `ingest` concurrency group queues them, and work that's done (news extraction, the daily refresh, a benchmark set) isn't repeated.
   - First trigger: Oct 28 00:28 UTC, response 204.
5c. ~~Clearer answers~~ (Oct 27 2026): the decision adds a 3–5 sentence `summary` citing ids like [E2] (`PROMPT_VERSION` 3), shown above the factor bullets with the ids linked. Validation now also rejects any number in the summary, factors or disagreement that isn't in the packet (rounded values, differences between two packet numbers and counts up to 10 are allowed) and swapped player names ("Tyler Kelce"); an uncited summary was rejected too until 5d. Live check (Oct 26, 9 matchups): every answer validated, one after a retry for a missing citation. The page explains the confidence and "AI pick"/"Baseline pick" tags (`<details>`), and compares exactly 2 players (the engine still takes more). The header shows the week's game days (e.g. "games Thu Oct 22 – Mon Oct 26"; NFL weeks run Thursday–Monday, the fantasy week Tuesday–Monday). If either player's game has already kicked off, the answer is "No pick needed" (locked in most leagues): no AI call and no `recommendations` row, with the evidence still shown. The web app takes an injectable `clock`, so tests don't depend on today's date. Known: the model expands team abbreviations ("Baltimore" from BAL), and an activated player shown as both "exempt list" (stale roster) and "activated" (news) can confuse it.
5d. ~~Wider layout~~ (Oct 27 2026): up to ~1150 px wide; header on one row (title left, week/dates/freshness right); the two player inputs side by side with scoring + button on one row; the answer card in two columns (verdict, tags and explainer left; summary and reasons right, capped at a readable width); evidence as a comparison grid (one row per kind, one column per player, picked player shaded; `PlayerEvidence.items_of`); "How it works" collapsed; small citation chips. Phones stack everything, with each evidence row labeled by player. Validation got more forgiving after a live near-tie failed twice on format: citations that aren't ids (e.g. the packet's header line) are dropped, and a summary without citations is accepted (the number check still applies); `PROMPT_VERSION` 4. Re-check: 8/8 real matchups answered by the AI on the first call. Follow-up: scoring is a segmented toggle (still radio buttons), the button sits at the right edge (full width on phones), and the same-name hint is gone (the ambiguity error gives the example).
5e. Limit and Reddit checks (Oct 27 2026): `/diag/limits` (`startsit/web/diag.py`) shows the masked IP, which headers carried it, the visitor hash, and usage of each AI limit (no AI call). It's registered only when `VERCEL_ENV=preview`, and preview deployments sit behind Vercel's login, so production never serves it. Reddit gets read by the ingest jobs, not the web app, so it's tested from GitHub Actions: `test_live.py::test_reddit_is_readable_from_this_runner` (no API quota; Actions → integration → Run workflow).
6. **Weekly evaluation, then Reddit as a crowd signal** (planned Oct 27 2026; 6a first, since it measures everything else).
   - **6a. Evaluation** (built Oct 27 2026: scoring, benchmark, backfill and the track record page):
     - **Built:**
       - `startsit/evaluate.py` scores into `evaluations`: points per player (rounded to 2 decimals), the top scorer, and hits as 1/0/None (a tie counts for neither side) for the shown pick, the baseline and the raw consensus.
       - A matchup waits until every player's team game is in `player_weeks`. Checked on real week 7 data: Monday night's DAL@PHI matchups stayed pending.
       - Consensus pick: the rankings scrape on or before the day asked; the better ECR within a position, else the higher PPR projection; an unranked player loses.
       - The report (`python -m startsit.evaluate`) compares AI, baseline and consensus on the AI-answered matchups, and adds AI overrides, confidence buckets and baseline/consensus over all matchups.
       - The daily stats refresh scores new results.
       - `startsit/benchmark.py`: 30 pairs a week (QB 4, RB 9, WR 12, TE 5; raised from 20 on Oct 27) of ECR neighbors starting at QB10/RB15/WR20/TE6. Who's listed first alternates, and the scoring formats cycle.
       - **Live set:** asked by news runs in the window 3 h before the first Sunday kickoff at noon ET or later (10am–1pm ET for the usual 1pm slate; morning international games don't count), skipping players whose game has started, 15 s between AI calls (≤4 of the shared 12/minute, ~8 minutes a set). Moved from 9am and 10 s on Oct 27 to catch more Sunday-morning news and leave room for users. A window missed entirely is left to the backfill.
       - **Budgets:** `benchmark` 45/day (30 + retries); decide lowered to 390, so 390 + 60 + 45 = 495 of Gemini's 500.
       - **Interruptions** (code review, Oct 27):
         - A set is complete only when every matchup was asked; completion is recorded in `ingest_runs` as `benchmark:<tag>` (`benchmark.done`).
         - It stops without logging when the AI never answered (`Decision.fallback` in `NOT_ANSWERED`: busy, over a limit, unreachable), so baseline picks are never logged as the AI's.
         - Later runs resume it, skipping questions already asked under either tag. The backfill needs only `calls_needed` budget to resume.
         - An answer that fails validation (`fallback = "invalid"`) is logged: the AI answered badly, which counts against it.
         - Pauses are 15 s per AI call actually made (a retry counts), before the next call, never after the last.
       - **No peeking:** packets use only rankings and projections archived on or before the day asked (`through_day`); the consensus pick uses the same cutoff.
       - **Backfill** (`ingest.__main__.run_backfill`, tag `benchmark-backfill`, reported as its own group):
         - Not on Sundays in UTC (Saturday 8pm to Sunday 8pm Eastern; budgets reset at midnight UTC, so that day's budget is the live set's). Needs room for a whole set in today's budget, so in practice one week a day, newest first, weeks 7 → 2.
         - Asks as of that week's live time. It reads the rankings scrape from just before then from the file's git history (`rankings.snapshot_before` via the GitHub API with the workflow token, then `fetch_snapshot`) and archives it.
         - Week 1 can't be backfilled: its scrapes lack kickoff times, so the file's week can't be inferred (`BACKFILL_FROM_WEEK` = 2).
       - **Backfill caveats** (why it's reported apart from the live benchmark):
         - News only exists from Oct 24, so weeks 2–6 packets have no news, and the AI can't use the late news that's its main edge.
         - There are no FantasyPros projections before Oct 25 (usage-based catches convert the formats).
         - nflverse's weekly roster status for a past week may reflect later changes.
         - If the model's training data covered these games, its answers could be contaminated; unlikely for a model released before the season, but not provable.
         - Backfilling week 7 overwrites the archived Oct 25 rankings scrape with the 3am ET version.
       - **Track record page** (`/record`, linked from the header):
         - One card per group, in this order: weekly benchmark, backfill, questions asked on the site.
         - Each card has: AI / consensus / baseline hit-rate tiles (AI with a 95% margin of error); a small-sample note under 100 decided matchups; a by-week table; the AI-vs-baseline overrides with the list of matchups (pick, points, right/wrong); and hit rate by confidence.
         - The report is cached per process for an hour (`RECORD_CACHE_SECONDS`); it changes once a day.
         - The page frame is shared via `templates/_layout.html`.
       - **Dry run on real data** (baseline only, no AI, Oct 27): 180 matchups over weeks 2–7, each mapped to the right week's Sunday ~3am ET scrape. Of 176 scored, the baseline hit 89/175 (51%) and the raw consensus 83/172 (48%): close calls are coin flips for the experts, the bar the AI has to clear. (The earlier 20-pair week 7 dry run: 6/17 each.)
     - **Actual points:** each player's PPR/half/standard points from the nflverse weekly stats the daily job already loads. A player who didn't play scores 0.
     - **Picks scored per matchup:** a pick hits when the chosen player outscores the other in the asked scoring. Four picks are scored side by side:
       - AI pick (what the user saw);
       - baseline pick (projection);
       - consensus pick (better FantasyPros rank; today nearly always equal to the baseline, tracked so they can diverge);
       - crowd pick (from 6b).
     - **Weekly benchmark set:** real questions are too few for meaningful hit rates, so each week code runs ~20–30 close matchups through the AI (same position, small projection gap: the real dilemmas). That's ~30 Gemini calls/week within the decide budget. They're kept apart from user questions in every report (e.g. `recommendations.visitor = 'benchmark'` or a source column).
     - **Schedule:** after Monday night's stats land (normally Tuesday), the daily run scores the finished week into a results table.
     - **Public "Track record" page:** hit rates per week and for the season, by pick type. It also shows the matchups where the AI overrode the baseline and who was right, and calibration (does High hit more often than Low?). Decided to make it public: it's the portfolio story, bad weeks included.
   - **6b. Reddit** (access findings under Open to-dos: JSON is blocked from Actions, RSS works but is rate-limited):
     - **Access checked (Oct 28):** the search RSS finds the daily per-position WDIS threads, and their comments RSS works at ~60 s spacing. Comments are mostly questions; answers are short replies; there are no upvotes (details under Open to-dos).
     - **Ingest:** a few game-day runs (e.g. Saturday evening, Sunday morning) read those threads' comment feeds, ~60 s apart. A 403/429 means no Reddit data that run, never a failure.
     - **Extraction:** batched and cached per comment, like news: `{player, start/sit, reason, link}` (RSS carries no upvotes). Bots and memes are filtered out. Own daily budget (~20 calls; 400 + 60 + 20 stays under Gemini's 500).
     - **Evidence:** a per-player crowd item (e.g. "Reddit: 8 start / 2 sit"), plus items for comments that carry new information (practice report, role change). The prompt treats new information differently from plain preference.
     - **Crowd pick:** computed in code from the counts, logged with every recommendation, and scored by 6a. Crowd opinion can beat expert consensus, especially on late news the rankings haven't absorbed; the evaluation decides how much weight it earns.
     - **UI:** when the crowd and the experts disagree, the answer says so plainly.

## Open to-dos
- [x] Request FantasyPros API key (approved; key received)
- [x] Test what the free FantasyPros key returns (Oct 24 2026, from the Claude Code cloud env after allowlisting `api.fantasypros.com`; re-run with `scripts/probe_fantasypros.py`, which never prints the key):
  - Auth: `x-api-key` header. Base URL: `https://api.fantasypros.com/public/v2/json/nfl`. Every response includes `tier: "free"` and `limit: 10`.
  - **The free tier returns at most 10 rows per call.** `limit`, `offset`, `start` and `page` are ignored, so there's no paging. `count` still reports the full total (e.g., 378 for weekly FLX ECR).
  - `GET /2026/consensus-rankings?position=ALL|QB|RB|WR|TE|K|DST&type=weekly&scoring=PPR|HALF|STD&week=N`: all three scoring values work and are echoed back. An unknown scoring value silently falls back to `STD`. With no `week`, the API returns the current week (7 on Oct 24); `week=8` returns 0 players. Each row has `rank_ecr`, `pos_rank`, `rank_ave/min/max/std`, opponent, game status and ownership. Single-position calls also include `start_sit_grade` (e.g., "A+"), `tag` ("start"), and `r2p_pts` (projected points). No player-ID filter works here, so **ECR only covers the top 10 at each position**, which isn't enough for real start/sit questions.
  - `GET /2026/projections?position=WR&week=N&players=<fpid>:<fpid>:...`: the `players` filter works with a single position (not `ALL`) and reaches beyond the top 10. **IDs must be colon-separated**: with commas the API silently ignores the filter and returns the top 10 (found Oct 25 2026). Each row has `points`, `points_ppr` and `points_half` regardless of the `scoring` param. Some players came back empty (e.g., Rashee Rice, Emeka Egbuka); the cause is unknown, so handle a missing projection. This is the best usable per-player FantasyPros signal; `fpid` = `player_id`.
  - `GET /news`, `/injuries`, `/players`: 10 newest/top rows only, and no player or team filter worked. Injuries include `probability_of_playing` and Wed/Thu/Fri practice status (e.g., Limited/DNP), which is useful but limited to 10 players. News rows include title, description, `player_id` and a link.
  - **Quota recovered (Oct 27 2026):** the 10:52 UTC daily run fetched week 8 projections for all 204 ranked players. The Oct 25 lockout lifted within two days, so the limit is probably a daily or rolling quota that the ~100 test calls had used up. Keep the daily fetch as is and watch for 429s in the logs.
  - **Quota is smaller than ~75/day (found Oct 25 2026):** a full run at 20:51 UTC Oct 25 got a 429 on its first call, after only ~2 calls that Pacific day and ~30 that UTC day, so the limit isn't a simple daily reset; it may be weekly or monthly, and the ~100 test calls on Oct 24–25 likely used it up. Projections then fall back to usage-based catches (small effect, see 4c). **To do:** watch the daily run logs ("projections week N: X players") for when calls succeed again, then size the daily fetch to the real quota (e.g. only the top 10–20 per position, 4–8 calls).
  - **Rate limit:** no rate-limit headers are sent. After ~75–80 calls in one session (Oct 24 2026) every endpoint returned `429 {"message":"Limit Exceeded"}` (a daily usage-plan quota, likely). There's also a **per-second throttle**: ~6 back-to-back calls trip the same 429 (Oct 25 2026), so the client spaces calls 2 s apart and retries once after 15 s. Budget calls: ingest on a schedule and cache, never call per user query.
  - `GET /compare-players?players=<id>:<id>[:<id>]&position=WR|RB|...` (no season in the path; colon-delimited IDs): returns, for each scoring format (STD/PPR/HALF), **every expert's individual rank for each player**. This is the data behind FantasyPros' "Who Should I Start" tool, so code can compute "X of N experts start A over B". `position=ALL` returns empty lists. With PPR/HALF/STD, a non-top-10 player and a mixed-position pair still untested (hit the quota first).
  - Implication: nflverse FF Rankings stays the primary consensus source. Use FantasyPros per-player projections (and top-10 ECR/start-sit grade when available) as a supplementary, cited signal.
- [x] Verify `GEMINI_API_KEY` works from the Claude Code cloud env (Oct 24 2026: `gemini-3.5-flash` OK; `gemini-2.5-flash` returns 404 "no longer available to new users")
- [x] RSS feeds verified Oct 24 2026 (Claude Code env, after allowlisting): RotoWire, ESPN and CBS Sports work. Findings:
  - **RotoWire's public feed only ever lists its latest 5 blurbs** (no paging or filters), and links every item to the player's page, so items are keyed by `<guid>`, not URL. Fetched hourly, and every 30 minutes on Sunday mornings (step 5b); busy stretches can exceed 5 per run, so some RotoWire blurbs are missed (PFT and CBS list 30+ items).
  - RotoWire writes 12-hour local times ("1:47:00 PM PDT"), which `email.utils` misreads; ESPN labels times "EST" even during daylight time. US zone abbreviations are therefore read as local time in that region, and published times are capped at fetch time.
  - ProFootballTalk moved to `https://www.nbcsports.com/profootballtalk.rss` (needs `www.nbcsports.com` allowlisted here; unverified). FantasyPros has no working public RSS URL (404s); its news is in the API.
  - From GitHub Actions (Oct 24 2026, first news run): RotoWire, ProFootballTalk (30 items) and CBS work; **ESPN returns an empty 200 response to datacenter IPs**, so ESPN is effectively local-only. 71 items → 28 facts in 2 calls, written to Supabase.
- Extraction quality (Oct 24 2026, 66 live headlines, 2 calls): after tightening the prompt to skip opinions, predictions and picks (plus a code filter for `other`/`low` facts), 23 facts, all real availability/role news; all names matched. The first prompt produced ~40% noise (picks, quotes).
- [x] `GEMINI_API_KEY` added as a GitHub Actions repository secret.
- [x] Reddit from GitHub Actions (Oct 27 2026, `test_reddit_is_readable_from_this_runner`, Actions → integration → Run workflow with `only` = `reddit`; no API quota):
  - `.json` is **blocked** (403 "Blocked") on both `www.reddit.com` and `old.reddit.com`, so the JSON plan above doesn't work from Actions.
  - **RSS works:** `https://www.reddit.com/r/fantasyfootball/new/.rss` returned 200 with real posts (Atom).
  - The sub-wide comments feed (`/r/fantasyfootball/comments/.rss`) got **429** as the 4th request in ~6 s (2 s apart), so unauthenticated RSS is tightly rate-limited.
  - **Thread check (Oct 28 2026, `test_reddit_threads_and_comments_via_rss`, `only` = `reddit_threads`):**
    - **Finding threads works.** The search feed (`/r/fantasyfootball/search.rss?q=title%3AWDIS&restrict_sr=1&sort=new`) returned the daily threads `Official: [WDIS QB|RB|WR|Flex|K/TE/DEF] - Tue 10/27/2026`: five a day, one per position.
    - **Reading comments works at 60 s spacing.** A thread's comments feed (`<thread link>/.rss?limit=50`) got **429** 12 s after the search request, but **200 with 49 entries** at 60 s. Plan on ~60 s between Reddit requests (~5 minutes for the five threads).
    - **Most comments are questions**, e.g. "PPR Pick 3 AJB vs Miami, Moore vs Baltimore, …". The crowd signal is in the replies. The feed is flat (no reply structure is visible in the entries), so extraction has to work from the reply text (names a player) and any quoted context.
    - **No upvote counts in RSS.** Weighting by upvotes isn't possible this way; count one vote per reply.
  - Step 6 should read Reddit via RSS (posts, then a few thread comment feeds, spaced out) and degrade to nothing on 403/429.
- [x] FF Opportunity: 2026 data through the latest week; expected points are PPR-scored (derive half/std via `receptions_exp`)
- [x] FF Rankings weekly (`fp_latest_weekly.csv`, Oct 24 2026): `load_ff_rankings('week')` 403s here because the proxy blocks `github.com/.../raw/...`, but the same file loads from `https://raw.githubusercontent.com/dynastyprocess/data/master/files/fp_latest_weekly.csv`. It's a scrape of FantasyPros' full weekly ECR for **every** ranked player (1006 rows, `scrape_date` 2026-10-23), with `ecr`, `sd`, `best`, `worst`, `pos_rank`, `start_sit_grade`, `r2p_pts`, opponent and game status. Pages: `qb`, `ppr-rb`, `ppr-wr`, `ppr-te`, `k`, `dst` (+ IDP). **PPR only** for skill positions, so no half/std ECR. Keyed by `fantasypros_id` (same as the API `player_id`). It's FantasyPros data, so cite FantasyPros. Ingested by `startsit.ingest.rankings` (archived per `scrape_date`; week inferred from kickoff dates vs. the schedule, since the file has no week column). Re-checked hourly since Oct 25 2026. **Update cadence** (read from the file's git history, Oct 27 2026):
    - Scraped about twice a day (~07:00 and ~18:00 UTC).
    - The new week first appears on **Tuesday** (week 5 Oct 6 08:31 UTC, week 6 Oct 13 20:03, week 7 Oct 20 18:18, week 8 by Oct 27 10:51), once on a Wednesday (week 4, Sep 30).
    - The first scrape of a new week is partial: week 8's had 204 players vs ~400 later in the week. The hourly re-check picks up fuller scrapes.
- [x] nflverse injuries lag (Oct 24: practice status but no game status yet). Addressed by news `game_status` (step 4b): news "out" zeroes the baseline, news "active" beats a stale roster status.
- [x] FantasyPros per-format projections (step 4b, ~26 calls/day). Still deferred: `compare-players` expert head-to-head.
- [x] Gemini free-tier limits (author's AI Studio, Oct 24 2026), per model:
  - `gemini-3.5-flash-lite` (**chosen**): 15 requests/minute, 500 requests/day, 250k input tokens/minute.
  - `gemini-3.5-flash`: 1 request/minute, 20 requests/day, 250k input tokens/minute; too few requests for real users.
  - Requests, not tokens, are the constraint. Step 3: batch news extraction (one call carries all new headlines), so ingest uses ~1–5 calls/day. Step 4: cache decisions per (players, scoring, week); when the daily budget is spent, fall back to the code baseline with a clear "daily AI limit reached" note.
  - Step 4 check: run ~10 matchups through both Lite and Flash (~20 requests) and compare reasoning quality before finalizing the decision prompt.
- [x] Draft evidence packet schema (`startsit.evidence`: numbered items E1..En with kind/text/source_url; `to_text()` is what the LLM sees)
- [x] Stage 3 prompt (`decide.SYSTEM_PROMPT`, version 2). Override rule: only definite facts the projection hasn't caught (ruled out, inactive, activated, released, suspended, clear role change); a questionable/limited tag may lower confidence but isn't a reason to override (v1 overrode Puka Nacua for being questionable).
- [x] Lite vs Flash (Oct 24 2026, 8 real week-7 matchups): `gemini-3.5-flash-lite` answered 8/8 with valid, grounded output in ~1.1 s each; `gemini-3.5-flash` returned 503 "high demand" on 5 and then hit its 1/minute limit, answering 2/8 (same picks, ~5 s). Lite stays. Lite correctly overrode the baseline using news on every definite-fact case (Josh Jacobs activated, Lamar Jackson and DeVonta Smith ruled out, Jordan Mason activated). The provider retries 5xx once after 2 s.
- [x] High confidence on questionable picks: now capped at medium in code (step 4b).
- Decided Oct 24 2026: the baseline zeroes every player whose weekly roster status isn't Active (reserve/IR, practice squad, exempt, inactive, released, retired), not just cut/retired. nflverse can lag an activation by a day or two; the packet's roster-status item shows the LLM why.
- Decided Oct 24 2026: the "current week" is the latest week with rankings, so after Monday night the app keeps answering for the finished week until the new week's rankings arrive (normally Tuesday). Step 5's UI should say which week it's answering for.
- [x] Step 5: "game already started" warning and kickoff time; `names.PlayerMatcher` autocomplete.
- [x] Step 5: the full evidence packet (text + JSON with item ids) is stored with each logged recommendation. Injury reports and roster status are overwritten as they update, so the packet is the only record of what the user and LLM saw.
- [x] Deployed to Vercel (Oct 26 2026): https://who-should-i-start.vercel.app. Env vars: `DATABASE_URL` (transaction pooler URI), `GEMINI_API_KEY`, `VISITOR_SALT`; region `pdx1` from `vercel.json`. Visitor limits checked on a preview (`/diag/limits`). Reddit is read by the ingest jobs, so it's tested from Actions instead (see the Reddit to-do).
- [x] Supabase connection confirmed (Oct 24 2026): the first manual ingest run on `main` wrote the full season (954 players, 2,408 player-games, 441 week-7 rankings) to Supabase. The Claude Code cloud env **can't** reach Postgres (its proxy carries only HTTP/HTTPS), so development there uses SQLite and a local Postgres; Supabase is reached only from GitHub Actions, Vercel, or a laptop. `DATABASE_URL` must percent-encode special characters in the password (e.g. `#` → `%23`), or the value gets cut off when saved.

## Timing
The NFL regular season runs through early January. Shipping an MVP within ~2–3 weeks leaves time to collect real usage and several weeks of evaluation data.
