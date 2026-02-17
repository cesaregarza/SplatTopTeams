SHELL := /bin/bash

UV ?= uv

NPM ?= npm
FRONTEND_DIR := frontend
DAYS ?= 540
LOCAL_DB_URL ?= postgresql://postgres:postgres@localhost:5432/rankings_db?sslmode=disable
DOCKER_LOCAL_DB_URL ?= postgresql://postgres:postgres@postgres:5432/rankings_db?sslmode=disable
MIRROR_TABLES ?= matches,tournaments,player_appearance_teams,tournament_teams,team_search_refresh_runs,team_search_clusters,team_search_embeddings,players,player_rankings
MIRROR_CHUNK_SIZE ?= 5000
MIRROR_ALLOW_MISSING ?= 1

.PHONY: \
	help \
	install \
	lock \
	frontend-install \
	setup \
	mirror-analytics-tables \
	dev-api \
	dev-frontend \
	dev \
	dev-split \
	ports \
	refresh \
	refresh-local \
	refresh-local-from-remote \
	test \
	compose-up \
	compose-up-local \
	compose-up-remote \
	compose-refresh-local \
	compose-mirror-analytics-tables \
	compose-down \
	compose-logs \
	local-bootstrap \
	clean

help:
	@echo "Local development targets:"
	@echo "  make setup            Install backend+frontend deps with uv/npm, seed .env"
	@echo "  make install          Run uv sync with dev extras"
	@echo "  make lock             Refresh uv.lock"
	@echo "  make dev-api          Run FastAPI dev server on :8000"
	@echo "  make dev-frontend     Run Vite frontend dev server on :3000"
	@echo "  make dev              Run API + frontend together (Ctrl+C stops both)"
	@echo "  make dev-split        Print split-terminal instructions"
	@echo "  make ports            Show local URLs and docker-compose port mappings"
	@echo "  make refresh          Run embedding refresh on current DB (DAYS=$(DAYS))"
	@echo "  make refresh-local    Run refresh against local postgres at localhost:5432"
	@echo "  make refresh-local-from-remote  Read source from .env DB, write snapshots to local postgres"
	@echo "  make mirror-analytics-tables  Mirror source tables into local postgres (reads .env)"
	@echo "  make compose-mirror-analytics-tables  Mirror source tables via API container into local compose postgres"
	@echo "  make test             Run backend tests"
	@echo "  make compose-up       Start docker stack (uses .env DB URL by default)"
	@echo "  make compose-up-local Start docker stack with API pinned to local compose postgres"
	@echo "  make compose-up-remote Start docker stack using .env DB URL"
	@echo "  make compose-refresh-local  Run refresh inside API container into local compose postgres"
	@echo "  make local-bootstrap  Start local stack + build local snapshots from remote source"
	@echo "  make compose-down     Stop docker-compose stack"
	@echo "  make compose-logs     Tail docker-compose logs"
	@echo "  make clean            Remove build cache/venv artifacts"

install:
	$(UV) sync --extra dev

lock:
	$(UV) lock

frontend-install:
	cd $(FRONTEND_DIR) && $(NPM) install

setup: install frontend-install
	@if [ ! -f .env ]; then cp .env.example .env; fi
	@echo "Setup complete."

mirror-analytics-tables:
	@set -euo pipefail; \
	if [ ! -f .env ]; then echo "Missing .env"; exit 1; fi; \
	set -a; source .env; set +a; \
	if [ -z "$${RANKINGS_DATABASE_URL:-}" ]; then echo "RANKINGS_DATABASE_URL is required in .env"; exit 1; fi; \
	$(UV) run python scripts/mirror_analytics_tables.py \
		--source-db-url "$$RANKINGS_DATABASE_URL" \
		--target-db-url "$(LOCAL_DB_URL)" \
		--schema "$${RANKINGS_DB_SCHEMA:-comp_rankings}" \
		--tables "$(MIRROR_TABLES)" \
		--chunk-size "$(MIRROR_CHUNK_SIZE)" \
		$(if $(filter 1,$(MIRROR_ALLOW_MISSING)),--allow-missing,)

dev-api: install
	$(UV) run uvicorn team_api.app:app --reload --host 0.0.0.0 --port 8000 --app-dir src

dev-frontend: frontend-install
	cd $(FRONTEND_DIR) && $(NPM) run dev

dev-split:
	@echo "Run in two terminals:"
	@echo "  1) make dev-api"
	@echo "  2) make dev-frontend"

dev:
	@set -euo pipefail; \
	if [ ! -d .venv ]; then echo "Missing .venv. Run: make setup"; exit 1; fi; \
	if [ ! -d $(FRONTEND_DIR)/node_modules ]; then echo "Missing frontend deps. Run: make setup"; exit 1; fi; \
	($(UV) run uvicorn team_api.app:app --reload --host 0.0.0.0 --port 8000 --app-dir src) & \
	API_PID=$$!; \
	(cd $(FRONTEND_DIR) && $(NPM) run dev -- --host 0.0.0.0 --port 3000) & \
	FRONT_PID=$$!; \
	trap 'kill $$API_PID $$FRONT_PID >/dev/null 2>&1 || true' INT TERM EXIT; \
	wait -n $$API_PID $$FRONT_PID; \
	STATUS=$$?; \
	kill $$API_PID $$FRONT_PID >/dev/null 2>&1 || true; \
	wait $$API_PID $$FRONT_PID >/dev/null 2>&1 || true; \
	exit $$STATUS

ports:
	@echo "Local dev:"
	@echo "  API      http://localhost:8000"
	@echo "  Frontend http://localhost:3000"
	@echo ""
	@echo "Docker compose:"
	@echo "  teams-api      localhost:8000 -> container:8000"
	@echo "  teams-frontend localhost:3000 -> container:8080"
	@echo "  postgres       localhost:5432 -> container:5432"

refresh: install
	$(UV) run python -m team_jobs.refresh_embeddings --days $(DAYS)

refresh-local: install
	$(UV) run python -m team_jobs.refresh_embeddings --db-url "$(LOCAL_DB_URL)" --days $(DAYS)

refresh-local-from-remote: install
	@set -euo pipefail; \
	if [ ! -f .env ]; then echo "Missing .env"; exit 1; fi; \
	set -a; source .env; set +a; \
	if [ -z "$${RANKINGS_DATABASE_URL:-}" ]; then echo "RANKINGS_DATABASE_URL is required in .env"; exit 1; fi; \
	$(UV) run python -m team_jobs.refresh_embeddings \
	  --source-db-url "$$RANKINGS_DATABASE_URL" \
	  --target-db-url "$(LOCAL_DB_URL)" \
	  --schema "$${RANKINGS_DB_SCHEMA:-comp_rankings}" \
	  --days $(DAYS)

test: install
	$(UV) run pytest -q

compose-up: compose-up-remote

compose-up-local:
	RANKINGS_DATABASE_URL_DOCKER="$(DOCKER_LOCAL_DB_URL)" docker compose up -d --build

compose-up-remote:
	RANKINGS_DATABASE_URL_DOCKER="$(RANKINGS_DATABASE_URL)" docker compose up -d --build

compose-refresh-local:
	@set -euo pipefail; \
	if [ ! -f .env ]; then echo "Missing .env"; exit 1; fi; \
	set -a; source .env; set +a; \
	if [ -z "$${RANKINGS_DATABASE_URL:-}" ]; then echo "RANKINGS_DATABASE_URL is required in .env"; exit 1; fi; \
	docker compose exec teams-api python -m team_jobs.refresh_embeddings \
	  --source-db-url "$$RANKINGS_DATABASE_URL" \
	  --target-db-url "$(DOCKER_LOCAL_DB_URL)" \
	  --schema "$${RANKINGS_DB_SCHEMA:-comp_rankings}" \
	  --days $(DAYS)

compose-mirror-analytics-tables:
	@set -euo pipefail; \
	if [ ! -f .env ]; then echo "Missing .env"; exit 1; fi; \
	set -a; source .env; set +a; \
	if [ -z "$${RANKINGS_DATABASE_URL:-}" ]; then echo "RANKINGS_DATABASE_URL is required in .env"; exit 1; fi; \
	container_id=$$(docker compose ps -q teams-api | head -n 1); \
	if [ -z "$$container_id" ]; then \
	  echo \"No running teams-api container found\"; \
	  exit 1; \
	fi; \
	docker cp scripts/mirror_analytics_tables.py "$$container_id:/app/mirror_analytics_tables.py"; \
	docker exec "$$container_id" python /app/mirror_analytics_tables.py \
		--source-db-url "$$RANKINGS_DATABASE_URL" \
		--target-db-url "$(DOCKER_LOCAL_DB_URL)" \
		--schema "$${RANKINGS_DB_SCHEMA:-comp_rankings}" \
		--tables "$(MIRROR_TABLES)" \
		--chunk-size "$(MIRROR_CHUNK_SIZE)" \
		$(if $(filter 1,$(MIRROR_ALLOW_MISSING)),--allow-missing,)

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f --tail=200

local-bootstrap: compose-up-local compose-refresh-local

clean:
	rm -rf .pytest_cache .mypy_cache build dist *.egg-info .venv
	cd $(FRONTEND_DIR) && rm -rf node_modules dist
