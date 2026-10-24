# who-should-i-start
Fantasy Football helper for deciding which player to start.

## Status
Build step 2 of 6 (see `CLAUDE.md`): ingest + usage features (step 1), weekly consensus rankings,
baseline score, and evidence packet (step 2).

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
`startsit.db.Database` is the only module with SQL. It talks to Postgres (Supabase) when given
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
- `ingest_runs` — row counts per run

Expected points come from nflverse FF Opportunity, which is PPR-scored; half/standard are
derived by subtracting 0.5 / 1.0 × expected receptions.

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
