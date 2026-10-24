# who-should-i-start
Fantasy Football helper for deciding which player to start.

## Status
Build step 1 of 6 (see `CLAUDE.md`): skeleton, nflverse ingest, usage features, daily cron.

## Setup
```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env   # fill in keys; never commit .env
pytest
```

## Ingest (Stage 1)
```bash
python -m startsit.ingest              # current season → data/startsit.db
python -m startsit.ingest --season 2025 --db /tmp/test.db
```
Each run re-ingests the full season from nflverse and upserts, so it is safe to re-run.
GitHub Actions runs it daily (`.github/workflows/ingest.yml`) and uploads the SQLite file
as an artifact until a Supabase project exists.

Tables (QB/RB/WR/TE, regular season only):
- `players` — GSIS id, name, position, current team
- `player_weeks` — per game: snap %, targets, carries, target share, WOPR, actual and expected
  fantasy points in standard / half-PPR / PPR
- `injuries` — weekly official injury report status
- `usage_features` — computed as of every week (`through_week`) so evaluation can use
  point-in-time data: last-3-game snap %, snap % trend vs. the prior 3 games, target share,
  opportunities, actual vs. expected points per format, and `last_game_week` (staleness)
- `ingest_runs` — row counts per run

Expected points come from nflverse FF Opportunity, which is PPR-scored; half/standard are
derived by subtracting 0.5 / 1.0 × expected receptions.

Data: [nflverse](https://github.com/nflverse) via `nflreadpy`.
