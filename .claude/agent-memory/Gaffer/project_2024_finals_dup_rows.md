---
name: 2024-finals-dup-rows
description: 2024 player finals rows are DUPLICATED — backfill appended real-date rows without removing the YYYY-03-01 placeholder rows
metadata:
  type: project
---

GENUINE DATA DEFECT found 2026-06-22 (needs Scientist, touches data/ — Gaffer did
NOT fix). 2024 finals rows in player CSVs are DUPLICATED.

**What:** The finals-date fix (commit b47b06088, week-offset mapping) + finals
backfill (commit 5fd605bf5, claimed "2024-2025") APPENDED corrected real-date
finals rows but did NOT remove the original `YYYY-03-01` placeholder finals rows.
So 2024 player files now carry BOTH sets. Example — daniher_joe_04031994 2024:
rows 200-203 = EF/SF/PF/GF all dated 2024-03-01 (placeholder), rows 204-205 =
PF 2024-08-23 + GF 2024-08-30 (real). Both coexist = double-counted finals.

**Scope (era partition of all 27,823 placeholder finals rows across the corpus):**
- All bad rows are EXACTLY `YYYY-03-01` (the placeholder fallback signature).
- 2024: 241 bad rows (DEFECT — should have been cleaned by the backfill).
- 2025: 2 bad rows (effectively clean).
- 2026: 0 (finals not played yet; consistent with [[project_player_finals_data_lag]]).
- Pre-2024: ~27,580 legacy placeholder rows. The recent fix never claimed to
  backfill these; whether they should be cleaned is a separate, larger question.

**Why it matters:** any per-player 2024 finals metric (finals goals, finals
disposal avg, finals appearances) double-counts. Headline season/career totals that
sum all rows are inflated for 2024 finalists.

**How to apply:** before shipping any doc with 2024 finals-level player stats,
flag this to Scientist. The fix is a dedup/replace on 2024 finals rows (keep the
month>=8 real-date row, drop the -03-01 placeholder), NOT an append. The
finals-date scan recipe (month<8 finals = suspicious) is CORRECT but conflates
this defect with legacy placeholders — partition by year>=2024 to isolate the
actionable set. See [[project_audit_url_collision_fp]], [[project_matches_player_sync]].
