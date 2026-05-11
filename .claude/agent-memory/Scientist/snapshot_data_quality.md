---
name: FanFooty snapshot col15/16/39 unreliable for goals/behinds/clangers
description: The FanFooty live-feed snapshot's per-player goals, behinds, and clangers fields are misindexed; cross-check against afltables.com for these three fields specifically
type: project
---

The FanFooty live-feed snapshots in `data/live_snapshots/<gameid>_<date>_<time>_*.json` use a 65-column per-player schema where the `goals`, `behinds`, and `clangers` fields are unreliable. Other fields (kicks, handballs, marks, tackles, hit_outs, AF, SC, Q1-Q4 AF splits, TOG%, DE%) agree exactly with afltables.com.

**Verification source**: Cross-checked the R10 2026 Richmond vs Adelaide snapshot (`9781_20260510_1751_final-siren.json`) against afltables.com for all 46 players. Found 102 field mismatches concentrated in goals, behinds, and clangers (21 mismatches per field on average). Kicks/marks/handballs/tackles/hit_outs had zero mismatches.

**Why:** The schema in `snapshot.schema.column_order` lists these columns at indices 15 (goals), 16 (behinds), 39 (clangers) but the actual data appears to be sourced from a different column position in the upstream FanFooty feed. The full-time verdict for R9 2026 also flagged this: "col15 'goals' field is unreliable (per-player team sums do not match scoreboard)".

**How to apply:**
- For any goals/behinds/clangers value in a post-match analysis, query afltables.com Round NN match URL instead of trusting the snapshot.
- For AF, SC, quarter splits, TOG, DE, and possession types (kicks/handballs/marks/tackles/hit_outs), the snapshot is reliable.
- Reproducibility check: re-compute team goal totals from snapshot player goals and compare to scoreboard total. If they disagree by more than 1-2 (allowing for genuine team-vs-individual discrepancies), the column is misindexed for that snapshot.
- The `update_r10_player_data.py` script in scripts/ shows the working pattern: use afltables for goals/behinds/clangers, use snapshot for AF/SC/quarter splits.
