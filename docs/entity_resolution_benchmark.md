# Entity Resolution Benchmark

Synthetic fixture benchmark for the deterministic team-resolution pipeline.

## Summary Metrics

| Mode | Recall@1 | Recall@5 | MRR | Latency 4p (ms) | Latency 7p (ms) | Latency 10p (ms) | Latency 15p (ms) | Subset Enums |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| subset_enum_baseline | 0.571 | 0.571 | 0.571 | 0.564 | 2.449 | 3.300 | 4.084 | 96 |
| whole_set_vector_only | 1.000 | 1.000 | 1.000 | 0.874 | 0.874 | 0.920 | 1.038 | 0 |
| whole_set_player_rerank | 1.000 | 1.000 | 1.000 | 0.969 | 0.932 | 0.981 | 1.251 | 0 |
| whole_set_player_pair_rerank | 1.000 | 1.000 | 1.000 | 1.061 | 0.994 | 1.212 | 1.360 | 0 |
| whole_set_player_pair_cluster | 1.000 | 1.000 | 1.000 | 1.231 | 1.455 | 1.431 | 1.897 | 0 |

## Per-Query Top Result

| Mode | Query | Top Team ID | Top Team Name | First Correct Rank |
| --- | --- | ---: | --- | ---: |
| subset_enum_baseline | 4-player exact lineup |  |  |  |
| subset_enum_baseline | Partial 7-player roster | 201 | Roster Seven A | 1 |
| subset_enum_baseline | Generic substitute |  |  |  |
| subset_enum_baseline | Star substitute |  |  |  |
| subset_enum_baseline | Historical reunion | 501 | Historic A | 1 |
| subset_enum_baseline | Large 10-player roster | 702 | Large Ten B | 1 |
| subset_enum_baseline | Large 15-player roster | 801 | Large Fifteen A | 1 |
| whole_set_vector_only | 4-player exact lineup | 101 | Core Exact | 1 |
| whole_set_vector_only | Partial 7-player roster | 201 | Roster Seven A | 1 |
| whole_set_vector_only | Generic substitute | 301 | Generic Sub A | 1 |
| whole_set_vector_only | Star substitute | 402 | Star Sub | 1 |
| whole_set_vector_only | Historical reunion | 501 | Historic A | 1 |
| whole_set_vector_only | Large 10-player roster | 702 | Large Ten B | 1 |
| whole_set_vector_only | Large 15-player roster | 802 | Large Fifteen B | 1 |
| whole_set_player_rerank | 4-player exact lineup | 101 | Core Exact | 1 |
| whole_set_player_rerank | Partial 7-player roster | 201 | Roster Seven A | 1 |
| whole_set_player_rerank | Generic substitute | 301 | Generic Sub A | 1 |
| whole_set_player_rerank | Star substitute | 402 | Star Sub | 1 |
| whole_set_player_rerank | Historical reunion | 501 | Historic A | 1 |
| whole_set_player_rerank | Large 10-player roster | 702 | Large Ten B | 1 |
| whole_set_player_rerank | Large 15-player roster | 802 | Large Fifteen B | 1 |
| whole_set_player_pair_rerank | 4-player exact lineup | 101 | Core Exact | 1 |
| whole_set_player_pair_rerank | Partial 7-player roster | 201 | Roster Seven A | 1 |
| whole_set_player_pair_rerank | Generic substitute | 301 | Generic Sub A | 1 |
| whole_set_player_pair_rerank | Star substitute | 402 | Star Sub | 1 |
| whole_set_player_pair_rerank | Historical reunion | 501 | Historic A | 1 |
| whole_set_player_pair_rerank | Large 10-player roster | 702 | Large Ten B | 1 |
| whole_set_player_pair_rerank | Large 15-player roster | 802 | Large Fifteen B | 1 |
| whole_set_player_pair_cluster | 4-player exact lineup | 101 | Core Exact | 1 |
| whole_set_player_pair_cluster | Partial 7-player roster | 201 | Roster Seven A | 1 |
| whole_set_player_pair_cluster | Generic substitute | 301 | Generic Sub A | 1 |
| whole_set_player_pair_cluster | Star substitute | 402 | Star Sub | 1 |
| whole_set_player_pair_cluster | Historical reunion | 501 | Historic A | 1 |
| whole_set_player_pair_cluster | Large 10-player roster | 702 | Large Ten B | 1 |
| whole_set_player_pair_cluster | Large 15-player roster | 802 | Large Fifteen B | 1 |

## Notes

- `subset_enum_baseline` exercises the old proxy-team projection path and is the only mode with non-zero subset enumeration counts.
- `whole_set_player_pair_cluster` is the new target configuration: whole-set dense retrieval, exact player+pair rerank, then post-consolidation group rescoring.
- Historical reunion queries are evaluated with no temporal penalty.
- No-bridge case qualitative check: `whole_set_player_pair_cluster` top result is `No Bridge A` (team `601`) with score `0.450`; this fixture is retained to verify the pipeline does not force a global merge.
