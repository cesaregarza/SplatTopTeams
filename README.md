# SplatTopTeams

Standalone tournament-team discovery app for the SplatTop cluster.

## What ships in this repo

- FastAPI backend for team similarity and cluster APIs.
- Daily refresh job that builds team embeddings and cluster assignments from Postgres.
- React frontend with Team Search + Cluster Explorer.

## Quick Start

### 1) One-command setup

```bash
cd /root/dev/SplatTopTeams
make setup
```

### 2) Run locally

```bash
make dev
```

`make dev` starts API + frontend together. `Ctrl+C` stops both.
It expects dependencies to already be installed (`make setup` once).

If you prefer split terminals:

```bash
make dev-split
```

### 3) Refresh embeddings

```bash
make refresh
```

`make refresh` reads and writes against the same DB URL.

### Local analytics mode (recommended with read-only remote DB)

If `.env` points at a read-only remote Postgres, build snapshots into local Postgres:

```bash
make compose-up-local
make refresh-local-from-remote
```

This reads source tournament tables from `.env` `RANKINGS_DATABASE_URL`
and writes `team_search_*` snapshots into local `localhost:5432`.

If you prefer to avoid host python tooling for the refresh, use:

```bash
make compose-refresh-local
```

### Mirror source analytics tables locally

To run head-to-head locally, mirror the source tables from the configured remote DB into local compose postgres:

```bash
make mirror-analytics-tables
```

If the API is running in docker-compose and you want to mirror directly into that container, use:

```bash
make compose-mirror-analytics-tables
```

Optional overrides:

```bash
make mirror-analytics-tables MIRROR_TABLES=matches,tournaments,player_appearance_teams,tournament_teams
make mirror-analytics-tables MIRROR_CHUNK_SIZE=10000
make mirror-analytics-tables MIRROR_ALLOW_MISSING=0  # fail if a requested source table is missing
```

`player_rankings` is now part of the default mirror set because tournament tier scoring needs it.

The API reads from the DB at request time, so a restart is not required after mirroring into a running compose DB.

### Useful commands

```bash
make test
make compose-up
make compose-up-local
make local-bootstrap
make compose-logs
make compose-down
make ports
```

### uv workflow

```bash
uv sync --extra dev
uv run pytest -q
uv run uvicorn team_api.app:app --reload --host 0.0.0.0 --port 8000 --app-dir src
```

## Key Environment Variables

- `RANKINGS_DATABASE_URL` (preferred)
- `DATABASE_URL` (fallback)
- `RANKINGS_DATABASE_URL_DOCKER` (optional `docker compose` override; otherwise compose uses `RANKINGS_DATABASE_URL`)
- `compose-up`/`compose-up-remote` use `RANKINGS_DATABASE_URL_DOCKER`; `compose-up-local` always uses the local compose postgres.
- `compose-up-local` only contains `team_search_*` tables unless you restore full match source data, so head-to-head analytics may return zero.
- `RANKINGS_DB_SCHEMA` (default: `comp_rankings`)
- `MIRROR_TABLES` (comma-separated table names, default now includes `player_rankings`)
- `MIRROR_CHUNK_SIZE` (default: `5000`)
- `MIRROR_ALLOW_MISSING` (`1` to skip missing source tables, default `1`)
- `DAYS` (`make` override for refresh window, default: `540`)
- `API_RL_PER_MIN` (default: `120`)
- `CORS_ORIGINS` (default: `*`)

## API

- `GET /api/health`
- `GET /api/team-search?q=<name>&top_n=20&cluster_mode=strict`
- `GET /api/clusters?cluster_mode=explore&limit=50`
- `GET /api/clusters/{cluster_id}?cluster_mode=explore`
- `GET /api/analytics/overview?cluster_mode=explore`
- `GET /api/analytics/matchups?cluster_mode=explore&min_matches=3`
- `GET /api/analytics/team/{team_id}?cluster_mode=explore&neighbors=12`
- `GET /api/analytics/team/{team_id}/blend?cluster_mode=explore&semantic_weight=0.5&neighbors=12`
- `GET /api/analytics/outliers?cluster_mode=explore&limit=30`
- `GET /api/analytics/space?cluster_mode=explore&max_points=900`
- `GET /api/analytics/drift?cluster_mode=explore&top_movers=20`
