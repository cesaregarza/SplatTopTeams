# Repo-Specific Response To That Summary

That zero-context summary gets the broad shape mostly right, but it misses the parts that are actually distinctive in this repo.

## What We Actually Have Here

We already have a repo-specific deterministic entity-resolution upgrade documented in:

- `docs/entity_resolution_design.md`
- `docs/entity_resolution_benchmark.md`
- `scripts/benchmark_entity_resolution.py`

So the most relevant answer is not just "players are hard-resolved and teams are soft-resolved." It is:

1. We precompute deterministic team representations in the refresh job.
2. We search with those representations using whole-set player queries by default.
3. We rerank with exact player and pair overlap signals.
4. We only consolidate near-duplicate teams after retrieval/ranking.
5. We rescore consolidated groups so the merged result is ranked as a group, not just as the first alias we happened to see.

## What In Code Backs That Up

### 1. Refresh-time feature build is real and important

This repo does not jump straight from names to consolidation.

The refresh job builds:

- semantic vectors
- identity vectors
- `player_support`
- `pair_support`
- `lineup_variant_counts`

Relevant files:

- `src/team_jobs/refresh_embeddings.py`
- `src/shared_lib/team_vector_utils.py`

That is a big part of the current system, and the zero-context summary mostly skips it.

### 2. Query-time search uses a deterministic whole-set player profile

The current default is documented as:

- `query_mode="whole_set"`

The old subset-enumeration path is still present, but mainly for benchmarking/regression comparison, not as the primary approach anymore.

Relevant files:

- `docs/entity_resolution_design.md`
- `src/team_api/store.py`
- `src/team_api/search_logic.py`

### 3. Retrieval is not just embedding similarity

The repo-specific upgrade added exact sparse reranking on top of dense retrieval:

- embedding score
- weighted player overlap
- weighted pair overlap

That is one of the main things this codebase now has that a generic summary would miss.

Relevant files:

- `docs/entity_resolution_design.md`
- `src/team_api/search_logic.py`
- `src/team_api/store.py`

### 4. Team consolidation is still roster/lineup-driven

This part of the summary is basically right.

Near-duplicate teams are grouped after ranking when they look like the same roster family, for example:

- exact top-lineup match
- near-match with limited player drift
- strong overlap plus similar lineup-count / stability characteristics

The grouping is deterministic and transitive via union-find.

Relevant files:

- `src/team_api/search_logic.py`
- `tests/test_search_logic.py`

### 5. Name matching is candidate lookup, not the main resolution rule

This is also basically right.

`team_name` matching helps find candidates, including normalized/fuzzy lookup, but the repo does not use a permanent canonical-team table as the main identity layer.

Relevant file:

- `src/team_api/store.py`

## What The Summary Gets Right

- `player_id` is the stable player identity used throughout the pipeline.
- There is no obvious permanent canonical-team table.
- Team names are not the core evidence for final same-team decisions.
- Final same-team grouping is driven mainly by roster/lineup evidence.

## What The Summary Leaves Out

- We already documented this as a deterministic upgrade, not just an ad hoc merge heuristic.
- We persist and use sparse support features (`player_support`, `pair_support`, `lineup_variant_counts`) in addition to embeddings.
- The default player-seeded path is whole-set query construction, not just proxy-team guessing.
- We added post-consolidation rescoring, so grouped aliases are scored as an aggregate profile.
- We benchmarked the upgraded pipeline and kept the old subset baseline for comparison.

## The Most Accurate Short Version For This Repo

We precompute deterministic semantic/identity team vectors plus sparse roster signals, retrieve with a whole-set player query, rerank with exact player/pair overlap, and then consolidate near-duplicate teams at query time using lineup/roster heuristics rather than name-based canonicalization.
