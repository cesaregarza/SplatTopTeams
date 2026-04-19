# Entity Resolution Evaluation

## Reference families

- Healbook
- BlankZ
- Distinkt
- Reload
- Seapunks
- Tidy Tidings
- Wave Breaker
- Operation Zero

These are encoded in:

- `src/team_api/healbook_reference.py`
- `src/team_api/real_family_reference.py`
- `tests/test_healbook_reference.py`
- `tests/test_real_family_reference.py`

## Current read

The current engine is stronger than the earlier retrofit shape made it look.

- It resolves broad same-family variants well on real teams with roster churn.
- It did not produce cross-family false merges in the reference set.
- It keeps clearly separate branches apart when the overlap is not strong enough:
  - Healbook `devade branch`
  - Seapunks `eider branch`

## What was weak

Two structural issues showed up during evaluation:

1. Consolidation only saw the already-trimmed `top_n` results.
   - That meant alias grouping quality depended on display size.
   - Real effect: true same-family rows could be missing from the consolidation graph.

2. Consolidated rescoring averaged every alias together too aggressively.
   - Widening the consolidation pool improved recall, but raw averaging diluted scores for larger families.

## What changed

### 1. Wider consolidation pool

Search now gathers a larger candidate window for consolidation, then trims back to the requested `top_n` after consolidation and rescoring.

Relevant code:

- `src/team_api/store.py`
- `src/team_api/search_logic.py`

### 2. Partial best-member rescue during group rescoring

Consolidated groups are still scored from aggregate support, but the final embed/player/pair features get a partial pull toward the best matching member instead of a hard average over the full family.

This avoids washing out a good match just because a team has many historical variants.

### 3. Lineup-aware rerank boost

The scorer now uses stored `lineup_variant_counts` during rerank and consolidated rescoring.

- exact lineup matches get a direct boost
- near lineups can still help when they are one-player drifts
- the boost is conservative and only raises a score above the embed/player/pair base when lineup evidence exists

## Current evaluation snapshot

From `scripts/evaluate_real_family_references.py` after the latest changes:

- Healbook core exact: `0.890`
- Healbook core grapeyy: `0.875`
- Healbook bridgeish variant: `0.709`
- Healbook five-player core: `0.847`
- BlankZ core: `0.973`
- Distinkt core: `0.934`
- Reload cinnamon core: `0.931`
- Seapunks rexy core: `0.924`
- Tidy Tidings core: `0.958`
- Tidy Tidings kooooo branch: `0.889`
- Wave Breaker core: `0.908`
- Wave Breaker drag branch: `0.937`
- Operation Zero core: `0.966`
- Operation Zero cherry branch: `0.886`

Branches kept separate:

- Healbook devade branch: standalone
- Seapunks eider branch: standalone

## Bottom line

The engine is not just "kind of working". On these real reference families, it looks materially solid.

The main remaining limitation is that it is still heuristic:

- consolidation is roster/lineup-rule driven
- scores are useful ranking signals, not calibrated probabilities
- family breadth can still change how much a group score compresses

If we push it further, the next worthwhile step is not "switch to embeddings".
It is building a larger labeled reference set and tuning the final same-team score against those cases, especially for:

- broad families with many historical variants
- branch splits that share one or two sticky players
- larger roster queries where lineup subset enumeration is intentionally capped
