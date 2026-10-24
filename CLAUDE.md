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
- Free keys may return only a sample slice of some datasets (e.g., full projections). **Test what the key actually returns before designing around it.**
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

## Free-tier stack candidates (confirm current limits before committing)
- **LLM:** Gemini API free tier, Groq, or free models on OpenRouter. Use a provider abstraction so the model can be swapped.
- **Hosting:** Vercel or Cloudflare (Pages/Workers).
- **DB/auth:** Supabase (Postgres; pgvector if needed).
- **Scheduled jobs:** GitHub Actions cron.

## Open to-dos
- [ ] Request FantasyPros API key (approval may take time) and test what a free key returns
- [ ] Verify RSS feed URLs
- [ ] Test Reddit `.json` access from the deployed environment
- [ ] Check nflverse FF Rankings / FF Opportunity contents and update cadence
- [ ] Choose LLM provider and confirm free-tier limits
- [ ] Draft evidence packet schema and Stage 3 prompt

## Timing
The NFL regular season runs through early January. Shipping an MVP within ~2–3 weeks leaves time to collect real usage and several weeks of evaluation data.
