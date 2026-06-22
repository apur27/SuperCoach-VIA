---
name: matches-player-sync
description: matches_*.csv vs player_data sync status — R10 2026 truncation fixed + guard live; residual quarter-score and 2025 R1/R2 caveats
metadata:
  type: project
---

State of the matches-file vs player-data sync (as of 2026-06-09 check).

**RESOLVED — R10 2026 truncation.** matches_2026.csv had lost all 8 Round-10
(2026-05-03) matches; only the lone 2026-05-10 Richmond v Adelaide survived. The
8 rows were reconstructed from player CSVs (commit de9fe39ec) and now reconcile
exactly: R10 holds 9 rows == 9 distinct R10 matchups in data/player_data/.
matches_2026.csv carries all rounds R1–R14, no other count anomalies.

**LIVE GUARD.** `audit_match_rounds` + `fetch_round_fixture` in
scrapers/game_scraper.py compare each round against the afltables season fixture
by team-pair and WARN on any scheduled-but-unscraped matchup by name. Wired into
refresh_data.py (post-write self-check, ~L178–194), which refresh_and_rank.sh L12
runs every cycle. The silent-truncation class is now caught.

**Why:** A scraper gap silently dropped a whole round and went unnoticed because
nothing cross-checked match files against the known fixture or against player CSVs.

**How to apply:**
- The 8 reconstructed R10 rows are LOWER-FIDELITY: quarter scores all 0 (unknown),
  behinds are player-sum lower bounds, venue/time blank. Final goals are sound.
  Any consumer reading quarter splits or exact behinds for R10 2026 will be wrong;
  margin- and aggregate-based consumers (backtest, briefs) are safe. Backfill when
  afltables publishes full R10 box scores.
- Residual: 2025 R1/R2 shows a ±1 row delta vs player-data matchup counts —
  consistent with the AFL Opening Round labeling boundary, NOT a truncation. Low
  severity, not fully reconciled; confirm before treating as a real gap.

**Finals-date fix (b47b06088) — VERIFIED working forward, stale residue remains
(checked 2026-06-22).** The `_FINALS_WEEK` map correctly stamps Aug-Sep dates for
NEWLY-scraped finals rows (wilmot/amon/gunston/laird now show 2024-08-09 EF etc.).
BUT 122 player files carry 243 stale 2024-finals rows still dated `2024-03-01`
(the old `datetime(year,3,1)+weeks` approximation) — they haven't been re-scraped
since the fix landed 2026-06-20, and the weekly DELTA won't touch them (their
finals rows already exist, just mis-dated). Cosmetic date-only: round labels and
years are correct, no games dropped. Pre-2024 finals (27,580 rows) were ALWAYS
March-1 approximations and are NOT the regression — the `date` col in player CSVs
is a known-unreliable synthetic field; rounds/years are the real keys.
- **How to apply:** Do NOT treat the 4,537-file raw March-1 scan count as a bug —
  filter to year>=2024 to isolate the actionable 122. A full re-scrape of just
  those 122 corrects the dates; only worth it if a downstream consumer reads the
  finals `date` field for time-ordering. Backtest/briefs key on round+year, so
  it's currently harmless. keane_mark_17032000 was a GENUINE gap (2025 QF+SF rows
  lost to the pre-fix bug) and was re-scraped + committed (dfbe8448e);
  sidebottom_steele game-35 "gap" is the 2010 drawn-GF+replay collapse and is
  CORRECT — leave it. Rioli Jr/Sr is a name-collision false positive (see below).

**Maurice Rioli name collision (audit false positive, 2026-06-22).**
`rioli_maurice_01092002` = Jr (57 games, Richmond 2021-26, jersey 49→17) is a
SEPARATE player from `rioli_maurice_01091957` = Sr (118 games, 1980s Richmond).
The player-audit compared Jr's CSV to Sr's afltables page (csv=57 vs afltables=118).
Tell: tackles csv=161 (Jr, modern) vs afltables=55 (Sr — tackles weren't recorded
pre-late-1980s). Both CSVs correct; the audit keyed the wrong afltables URL.
Don't "fix" either file. See [[project_player_data_quirks]] (Ablett snr/jnr
surname collision is the same class).
- Related: [[project_finals_doc_stale_2026]], [[project_player_finals_data_lag]].
