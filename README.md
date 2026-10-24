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
code-only baseline pick. The baseline is currently 100% FantasyPros consensus: projected
points (PPR) converted to the league's format by removing points for expected catches
(from recent usage), and zero for players on bye or ruled Out. Usage adjustments will be
added only if the weekly evaluation shows they help.

Data: [nflverse](https://github.com/nflverse) via `nflreadpy`; expert consensus rankings from
[FantasyPros](https://www.fantasypros.com/nfl/rankings/), via
[DynastyProcess](https://github.com/dynastyprocess/data).
