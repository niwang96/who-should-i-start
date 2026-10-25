# who-should-i-start
Fantasy Football helper for deciding which player to start.

## Status
Build step 4 of 6 (see `CLAUDE.md`): ingest and usage features, consensus rankings, baseline and
evidence packet, news extraction, and the AI decision. Next: web UI and recommendation logging.

## Setup
```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env   # fill in keys; never commit .env
ruff check . && ruff format --check . && mypy && pytest
```
Tests run against in-memory SQLite, and also against Postgres when
`STARTSIT_TEST_DATABASE_URL` points at a disposable database (CI starts one).

## Database
`startsit.db.Database` is the only module with SQL. Each table is defined once in `db.py`
(a TypedDict of its columns plus its key); the CREATE TABLE statement and the polars schemas are
generated from it, so adding a column is a one-line change. Scoring-format rules (PPR / half /
standard) live only in `startsit/scoring.py`. It talks to Postgres (Supabase) when given
a `postgres://` URL or when `DATABASE_URL` is set, and to a local SQLite file otherwise
(`STARTSIT_DB_PATH`, default `data/startsit.db`).

## Ingest (Stage 1)
```bash
python -m startsit.ingest              # current season → $DATABASE_URL or data/startsit.db
python -m startsit.ingest --season 2025 --db /tmp/test.db
```
Each run re-ingests the full season from nflverse and upserts, so it is safe to re-run.
GitHub Actions runs it daily (`.github/workflows/ingest.yml`), writing to Supabase when the
`DATABASE_URL` repository secret is set (otherwise it uploads a SQLite file as an artifact).

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
python -m startsit.ingest --only news-fetch  # fetch RSS only (no LLM), for frequent polling
```
Feeds (`startsit/ingest/news.py`) are stored as headline + summary + link only. Items not yet
processed are sent to Gemini in batches of ~40 (`startsit/ingest/news_extract.py`); the model
returns, per player, an event type, a severity and a short restatement, and code attaches the
URL, date and player id (names it can't match to a player are dropped). Every processed URL is
cached in `news_extractions` (keyed by the feed's guid, else the link), so no item is sent twice. Facts also record a game
status (out/doubtful/questionable/active) when the item states one. News uses at most 40 LLM calls a
day (`llm_calls` logs every call). GitHub Actions fetches feeds every 15 minutes at peak times,
runs the news step hourly, and runs the full ingest (including FantasyPros projections) daily.

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
- daily budgets per purpose: 400 decisions, 40 news calls (Gemini allows 500/day);
- one shared cap of 12 calls/minute across ingest and web requests (Gemini allows 15);
- per visitor (web app): 6 new AI answers per 10 minutes and 25 per day, keyed by a salted hash
  of the IP address (`VISITOR_SALT`); cached answers don't count.

Each limit has its own note on the baseline fallback ("daily AI limit reached", "the AI is busy
right now", "you've reached the limit of ..."). A Gemini 429 is read the same way: a per-day
quota error counts as the daily limit, anything else as the per-minute one.

## Baseline and evidence packet (Stage 2)
```bash
python -m startsit.evidence "DeVonta Smith" "Puka Nacua" --scoring half
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
