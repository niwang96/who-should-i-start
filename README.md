# who-should-i-start
Fantasy Football helper for deciding which player to start.

## Status
Build step 6 of 6 (see `CLAUDE.md`): ingest and usage features, consensus rankings, baseline and
evidence packet, news extraction, the AI decision, the web app, and the weekly evaluation (6a).
Next: the public track record page, then Reddit as a crowd signal (6b).

## Setup
```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev,ingest]'
cp .env.example .env   # fill in keys; never commit .env
ruff check . && ruff format --check . && mypy && pytest --cov
```
Three kinds of tests:
- **Unit tests:** pure logic with fakes; no network and never a real LLM.
- **Database tests:** every query and ingest step, against in-memory SQLite and, when
  `STARTSIT_TEST_DATABASE_URL` points at a disposable database, Postgres too (CI starts one).
  These two kinds run on every pull request, and CI fails below 95% branch coverage.
- **Live integration tests** (`tests/test_live.py`, `pytest -m live`): the real Gemini,
  FantasyPros, feeds, rankings file and nflverse, end to end through our code. They spend 2-4
  Gemini calls and 1 FantasyPros call per run, so `integration.yml` runs them only on merges to
  `main` that change code talking to an outside service, and when started by hand. Without a key, that
  service's test skips.

## Web app (Stage 5)
```bash
pip install -e '.[dev]'
uvicorn startsit.web.app:app --reload   # http://localhost:8000, uses $DATABASE_URL or the SQLite file
```
FastAPI with Jinja templates and htmx (`src/startsit/web/`). htmx is vendored as a pinned file
in `static/`, so pages load nothing from other sites and the content-security policy allows only
the app's own origin. Every page works without JavaScript; with it, htmx swaps in just the
result. The page says which week it answers for and how fresh the data is, suggests players as
you type (the full player list ships with the page, so typing sends no requests, and only the
button submits; Enter doesn't), shows the pick, confidence (with an explainer), a short summary and deciding factors with links
to the numbered evidence, every evidence item with its source link, and, when either player's game has
already kicked off, "No pick needed" instead of an AI call (the spot is locked in most leagues). Every answer is logged to `recommendations` with the full packet (text and
structure) for the weekly evaluation; visitors are recorded only as a salted hash.

Deployed on Vercel (free Hobby plan): `api/index.py` exposes the app, which Vercel's FastAPI
support serves for every path; `vercel.json` sets the region (`pdx1`, next to Supabase) and must
not rewrite paths (a rewrite to `/api/index` made every page a 404). The base install (no `ingest` extra) is all the web app needs, so polars and
nflreadpy stay out of the deployment. Vercel environment variables: `DATABASE_URL` (Supabase
**transaction pooler**, port 6543), `GEMINI_API_KEY`, `VISITOR_SALT`.

## Database
`startsit.db.Database` is the only module with SQL. Each table is defined once in `db.py`
(a TypedDict of its columns plus its key); the CREATE TABLE statement and the polars schemas are
generated from it, so adding a column is a one-line change. Scoring-format rules (PPR / half /
standard) live only in `startsit/scoring.py`. It talks to Postgres (Supabase) when given
a `postgres://` URL or when `DATABASE_URL` is set, and to a local SQLite file otherwise
(`STARTSIT_DB_PATH`, default `data/startsit.db`).

On Postgres every table has row-level security enabled with no policies, so Supabase's REST API
(the anon and authenticated roles) can't read or write them; only the database password can.

## Ingest (Stage 1)
```bash
python -m startsit.ingest              # current season → $DATABASE_URL or data/startsit.db
python -m startsit.ingest --season 2025 --db /tmp/test.db
```
Each run re-ingests the full season from nflverse and upserts, so it is safe to re-run.
GitHub Actions runs it (`.github/workflows/ingest.yml`), writing to Supabase when the
`DATABASE_URL` repository secret is set (otherwise it uploads a SQLite file as an artifact).
Every scheduled run is a news run (below) that also does this daily refresh when it's due (none
since the latest 10:00 UTC), because GitHub delays or drops scheduled runs when it's busy: the
first run to get through after 10:00 UTC does it.

Tables (QB/RB/WR/TE, regular season only):
- `players` — GSIS id, name, position, current roster team; includes rostered players who
  haven't played yet (rookies, returns from injury)
- `roster_weeks` — each player's team and roster status per week (active, reserve/IR,
  practice squad, cut, retired, ...); follows trades and releases
- `player_weeks` — per game: snap %, targets, carries, target share, WOPR, actual and expected
  fantasy points in standard / half-PPR / PPR
- `injuries` — weekly official injury report status
- `usage_features` — computed as of every week (`through_week`) so evaluation can use
  point-in-time data: last-3-game snap %, snap % trend vs. the prior 3 games, target share,
  opportunities, actual vs. expected points per format, and `last_game_week` (staleness)
- `games` — regular-season schedule (also used to detect byes)
- `rankings` — FantasyPros weekly expert consensus (ECR) for every ranked QB/RB/WR/TE, one
  archived copy per daily scrape: average/best/worst rank, spread, start/sit grade, and
  FantasyPros' projected PPR points. Read from the DynastyProcess copy of FantasyPros' rankings.
- `projections` — FantasyPros per-format projections and projected receptions for ~240 top players,
  one archived copy per daily fetch (needs `FANTASYPROS_API_KEY`)
- `ingest_runs` — row counts per run

Expected points come from nflverse FF Opportunity, which is PPR-scored; half/standard are
derived by subtracting 0.5 / 1.0 × expected receptions.

## News (Stage 1, text)
```bash
python -m startsit.ingest --only news        # fetch RSS, extract player facts with Gemini
python -m startsit.ingest --only news-fetch  # fetch RSS only (no LLM)
```
Feeds (`startsit/ingest/news.py`) are stored as headline + summary + link only. Items not yet
processed are sent to Gemini in batches of ~40 (`startsit/ingest/news_extract.py`); the model
returns, per player, an event type, a severity and a short restatement, and code attaches the
URL, date and player id (names it can't match to a player are dropped). Every processed URL is
cached in `news_extractions` (keyed by the feed's guid, else the link), so no item is sent twice. Facts also record a game
status (out/doubtful/questionable/active) when the item states one. News uses at most 60 LLM calls a
day (`llm_calls` logs every call). GitHub Actions runs the news step hourly, every 30 minutes on
Sunday mornings (about 8am to 1:30pm US Eastern, when inactives and game-time decisions come out and
lineups are set) and twice before the 4pm games, and twice before Thursday and Monday night games.

Tables: `news_items`, `news_extractions` (the per-URL LLM cache), `news_facts`, `llm_calls`.

## Decision (Stage 3)
```bash
python -m startsit.decide "Josh Jacobs" "Saquon Barkley" --scoring ppr
```
prints the evidence packet and the AI's pick: confidence, 2–3 deciding factors with the evidence
ids they rely on, and, when it overrides the baseline, why. Code checks every cited id against the
packet, retries once with the problems, and otherwise falls back to the baseline pick with a note
(also when a limit below is hit or Gemini is down). Answers are cached by the exact packet text,
so a repeat question costs nothing until the evidence changes.

Free-tier protections (`startsit/llm.py`, enforced before every call by counting `llm_calls`):
- daily budgets per purpose: 390 decisions, 60 news calls, 45 benchmark calls (Gemini allows 500/day);
- one shared cap of 12 calls/minute across ingest and web requests (Gemini allows 15);
- per visitor (web app): 6 new AI answers per 10 minutes and 25 per day, keyed by a salted hash
  of the IP address (`VISITOR_SALT`); cached answers don't count.

Each limit has its own note on the baseline fallback ("daily AI limit reached", "the AI is busy
right now", "you've reached the limit of ..."). A Gemini 429 is read the same way: a per-day
quota error counts as the daily limit, anything else as the per-minute one.

## Evaluation (Stage 4, step 6a)
```bash
python -m startsit.evaluate     # score what's finished, then print the season's hit rates
```
Every logged recommendation is scored once all its players' games are in the nflverse stats
(a player with no stats scores 0 once his team's game is in; until then it waits). A pick hits
when the chosen player outscored the other in the question's format; a tie counts for neither
side. Three picks are scored for each matchup: the one shown (the AI's, or the baseline's when
the AI wasn't used), the code baseline, and the raw FantasyPros consensus on the day asked
(better rank within a position, else the higher PPR projection). The report compares them on
the matchups the AI answered, and also shows how the AI did when it went against the baseline
and how often each confidence level hit. The daily stats refresh scores new results.

Real questions are too few for meaningful hit rates, so each week a benchmark set of 30 close
matchups (same position, neighbors in the consensus ranks, borderline starters) is asked of the
AI three hours before the Sunday main slate, 10am Eastern for the 1pm games (`startsit/benchmark.py`,
run by the ingest job, 15 s between calls). They're logged with `visitor = "benchmark"` and reported separately.

Past weeks (2 onward) are backfilled one a day: the same 30-matchup set, asked as of that
week's Sunday morning with the rankings scrape from just before it (read from the rankings
file's git history). Questions only ever see rankings and projections from on or before the
time asked. Backfilled weeks have no news before Oct 24, 2026, so they're reported apart
(`benchmark-backfill`).

## Baseline and evidence packet (Stage 2)
```bash
python -m startsit.decide "DeVonta Smith" "Puka Nacua" --scoring half --no-llm
```
prints the compact, numbered evidence (E1, E2, ...) the Stage 3 LLM will see, plus the
code-only baseline pick. Add a team to disambiguate: `"Ryan Miller (NE)"`. The baseline is
currently 100% FantasyPros consensus: projected points (PPR) converted to the league's format
by removing points for expected catches (from recent usage), and zero for players on bye,
ruled Out, cut or retired. New evidence kinds are added as one function in
`evidence.ITEM_BUILDERS`. Usage adjustments will be
added only if the weekly evaluation shows they help.

Data: [nflverse](https://github.com/nflverse) via `nflreadpy`; expert consensus rankings from
[FantasyPros](https://www.fantasypros.com/nfl/rankings/), via
[DynastyProcess](https://github.com/dynastyprocess/data).
