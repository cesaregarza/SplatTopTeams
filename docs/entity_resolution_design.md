# Deterministic Entity Resolution Upgrade

## Audit Baseline
- Query vectors were previously built only from existing team rows via `build_query_similarity_profile()`.
- Larger player-seeded queries fell back to `TeamSearchStore._project_seed_players_to_proxy_teams()`, which enumerated capped 4-player subsets and built a proxy centroid from matching teams.
- `identity_beta` was already configured in the refresh job and stored per snapshot row, but `identity_idf_cap` was not persisted before this change.
- Identity IDF weights were derived from player presence across teams with `log((n_teams + 1) / (team_count + 1))`, capped by `identity_idf_cap`.
- Ranking and consolidation already formed a sensible deterministic pipeline: dense retrieval first, then `consolidate_ranked_results()`.

## What Changed
- Added shared vector helpers in `src/shared_lib/team_vector_utils.py` so refresh-time hashing and query-time hashing use the exact same BLAKE2b index/sign scheme.
- Added direct whole-set query construction for arbitrary player sets:
  - `build_query_semantic_vector()`
  - `build_query_identity_vector()`
  - `build_query_final_vector()`
  - `build_player_query_profile()`
- Extended refresh payloads and snapshot storage with explicit sparse support:
  - `player_support`
  - `pair_support`
- Persisted `identity_idf_cap` in `team_search_refresh_runs` so query-time IDF reconstruction can match the refresh job exactly.
- Extended snapshot cache metadata with:
  - `idf_lookup`
  - `semantic_dim`
  - `identity_dim`
  - `identity_beta`
  - `identity_idf_cap`
- Updated the default player-seeded search path to `query_mode="whole_set"`.
- Kept the old subset projection path behind `query_mode="subset_enum"` for benchmarking and regression comparison only.
- Added exact sparse reranking on the top candidate window:
  - embedding score
  - weighted player overlap
  - weighted pair overlap
- Added post-consolidation group rescoring using aggregate player support, pair support, and a group centroid.

## Internal Defaults
- `query_mode="whole_set"`
- `rerank_candidate_limit=200`
- `w_embed=0.70`
- `w_player=0.15`
- `w_pair=0.15`
- `use_pair_rerank=True`
- `use_cluster_profile_scoring=True`

## Constraints Preserved
- No learned encoder was introduced.
- No match-level co-occurrence rewrite was introduced.
- No new time decay or temporal penalty was added; the existing optional `recency_weight` path was left unchanged.
- Missing player/pair evidence stays neutral.
- Consolidation grouping rules stay the same; only post-group scoring is new.
