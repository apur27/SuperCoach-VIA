# Survey — 2026-07-14 — scope: STANDARD (weekly-refresh flow, R19/R20 cycle)

## Executive read
The pipeline's compute core is sound — both R20 prediction runs produced identical
outputs (413 rows, maxdiff 0) and every gate that fired, fired correctly. What broke
the straight-through run is the *shipping layer*: an ungated-but-staged HOF hub that
kills Phase 4 every week its numbers change (it killed the 07-14 run and forced the
07-13 hold-back), a single-shot DataSentinel gate whose authoring requirements exist
only in the verifier's prompt (guaranteeing a first-pass FAIL), and a synthetic-date
mint in the scraper that is still running with no harness gate behind it. The single
highest-leverage move is the DataSentinel retry contract (F2) + hub gating (F1):
those two prevented four of the six manual interventions this cycle. Two sets of
wrong numbers are live on main today (hub rank-1 figures; ladder claims computed on
a matches CSV missing 9 games) — both escalated.

---

## Ranked findings

### F1 — HOF hub is staged every cycle but can never earn a verdict; killed Phase 4 on 07-14 and publishes stale numbers now  [severity: CRITICAL] [class: 1 + 6]
- Evidence (all measured 2026-07-14):
  - `weekly_refresh.sh:269` stages `docs/hall-of-fame-stat-leaders.md` in Phase 4; the verdict loop (`:167-177`) correctly omits it (S11-F3 fix), and `check_hof_numbers.py:19-30` `_SUBPAGES` never reads it — so when its content changes, no content-hash PASS record can exist and the pre-commit stamp gate fail-closes.
  - 07-14 log line 42258: `updated: docs/hall-of-fame-stat-leaders.md` (update_hof_pages rewrote hub rank-1 cells); log ends at line 42364 `[4/5] Staging and committing...` — the commit died, output untee'd (see F3). Sentinel `last_refresh_complete.json` never rewritten (still the hand-written 07-13 23:33 recovery record).
  - Gaffer retro (`agent-memory/Gaffer/project_weekly_r20_retro.md:19`): same block on 07-13; workaround was `git restore --staged`, "recurs every such week".
  - The reverted hub is now stale ON MAIN: measured against fresh `_stat_leaders.json`, 8 of 13 rank-1 `[data]` figures are wrong — hub says Pendlebury 436 games / 11,069 disposals / 2,012 tackles / 330 goal assists / 5,543 handballs, Dangerfield 4,712, Neale 1,992, Goldstein 10,597; JSON ground truth: 437 / 11,088 / 2,013 / 332 / 5,570 / 4,716 / 1,997 / 10,608. Hub prose also claims "13,343 player career files"; corpus is 13,353 (`ls | wc -l`).
  - The plumbing is half-built: the hub already carries machine sentinels (`<!-- HOF-HUB:career_games -->` etc., lines 31-40) that `check_hof_numbers.py` ignores.
- Impact: kills Phase 4 deterministically every week a rank-1 counter ticks (an active leader playing = most weeks); wrong `[data]` numbers currently published.
- Owner: **Scientist** (extend `check_hof_numbers.py` to verify HOF-HUB sentinel rows; ensure update_hof_pages regenerates hub cells + the "Last refreshed / N files" line) with the harness verdict-loop line routed to **Gaffer**.
- Recommended outcome: hub rank-1 cells are deterministically verified against `_stat_leaders.json` via its existing HOF-HUB sentinels, a PASS verdict is recorded for it in Phase 2b like the 10 sub-pages, and the current stale hub content on main is corrected. Published-wrong disposition → human.
- Effort: S–M · Impact-per-day rank: 2

### F2 — DataSentinel gate is single-shot and its authoring requirements live only in the verifier's prompt  [severity: HIGH] [class: 3 + 10]
- Evidence:
  - `weekly_refresh.sh:248-256`: one DataSentinel invocation; any non-PASS → `exit 1`. No route-back, no retry, no resume point.
  - The methodology-paragraph requirement is codified in `DataSentinel.md:35,45,146` ("Methodology paragraph missing → fail every tag") but appears NOWHERE in `FootyStrategy.md` (grep: zero hits for methodology-as-authoring-requirement) and not in the harness Phase 3 `-p` prompt (`weekly_refresh.sh:232`, which lists sources to read but no gate requirements). First-pass FAIL was structurally guaranteed.
  - Second FAIL (untagged restatements inside the fix's footnote) is documented only in uncommitted memory (`agent-memory/FootyStrategy/datasentinel_gate_traps.md`, indexed in its MEMORY.md line 8) — memory, not definition.
  - 07-13 log :61251 FAIL → :61285 FATAL abort; Gaffer retro :16 confirms the two-FAIL sequence and manual recovery.
- Impact: caused interventions 2 AND 3 this cycle (both DS FAILs and the resulting Phase-3b abort + manual Phase-4 completion). Any recap FAIL = dead harness.
- Owner: **Gaffer**.
- Recommended outcome: (a) FootyStrategy's definition gains a gated-doc authoring checklist (methodology paragraph naming source files; every specific number tagged at every occurrence; footnotes name files, never restate figures); (b) the harness Phase 3b gains ONE bounded retry: on FAIL, re-invoke FootyStrategy with the `failed_tags`/`untagged_numbers` JSON, re-gate, and only then abort. The gate's authority is untouched — this is routing, not relaxation.
- Effort: S (prompt) + M (retry) · Impact-per-day rank: 1

### F3 — Phase 4 failures are invisible and unresumable  [severity: HIGH] [class: 2]
- Evidence: `weekly_refresh.sh:289` — `git_commit_safe.sh` output is NOT tee'd to the log (every other phase is). The 07-14 log ends mid-phase with zero error text; the hook's rejection message went to a console nobody kept. Gaffer retro :29: "the harness has no resume-from-phase entry point"; re-running from scratch costs the full ~35-45 min Phase 1.
- Impact: every Phase-4 death requires forensic reconstruction (this survey needed the reflog + commit diffs to establish the cause); recovery invites allowlist-replication errors — the 07-14 recovery swept 12 HOF docs + cheat sheets into a commit titled "Fix 176 synthetic dates" (`4c798e665`), degrading history legibility.
- Owner: **Gaffer**.
- Recommended outcome: Phase 4 commit/push output tee'd to the audit log; on any phase abort the harness prints which phase died and what a sanctioned resume looks like; ideally a `--resume-from=<phase>` flag so recovery re-enters the harness instead of hand-replicating its allowlist.
- Effort: S (tee) + M (resume) · Impact-per-day rank: 4

### F4 — The synthetic-date mint is still running; no gate stands behind it  [severity: HIGH] [class: 5 + 4-adjacent]
- Evidence:
  - `player_scraper.py:262` still assigns `datetime(year,3,1)+weeks(round-1)` to numeric-round rows; fixture-date resolution exists ONLY for finals (`:339 _resolve_finals_date`, applied `:432-444`). The counter-aware rescue (`a4c1fdc20`) keeps mislabelled rows — with the synthetic date. The exact trigger of this cycle (afltables low-round label on a late-season game) will mint corrupt `days_since_last_game` inputs again.
  - `fix_synthetic_dates.py` (231 lines, `4c798e665`) repaired the corpus once; it is not invoked anywhere in `weekly_refresh.sh` or `refresh_and_rank.sh` (grep: zero hits).
  - `phantom_row_validator.py` validates counter contiguity + drawn finals only — no date-sanity layer (read in full; grep "date" hits are finals-replay logic).
  - The guard tests (`tests/unit/test_player_date_fix.py`, `test_match_dedup.py`) never run in the harness: `grep -c pytest weekly_refresh.sh` = 0. `refresh_and_rank.sh:12` claims the parent adds "QA" — no QA phase exists in the parent (false comment, class 7).
- Impact: intervention 4 recurs on the next mislabelled round; the model's temporal features silently poisoned between manual sweeps.
- Owner: **Scientist**.
- Recommended outcome: fixture-date resolution extended to numeric-round rows (matches the open C1 disposition); a deterministic date-sanity check (e.g. `fix_synthetic_dates.py --check` exit-nonzero mode, or a date layer in the phantom validator) wired into the harness gate stack; the unit suite run as a real harness QA phase so guard tests actually guard.
- Effort: M · Impact-per-day rank: 3

### F5 — Timing guard is an LLM weekday heuristic with three conflicting written cadences  [severity: MEDIUM] [class: 4 + 10]
- Evidence: `weekly_refresh.sh` has NO in-script timing check (full read); the guard is `skills/weekly-cycle.md:22-27` — LLM judgement over "Tuesday or later". Monday-with-settled-data is undefined by the rule text, so the agent blocked and asked (intervention 6). Three surfaces disagree: harness header comment `:6` says "run Wednesday morning"; the skill says Tuesday; the user's recorded feedback (`feedback_weekly_refresh_timing.md`) says Monday settlement is valid.
- Impact: one manual confirmation per early-settled week; a weekday rule proxies for a data condition the repo can measure directly (`audit_match_rounds`/`fetch_round_fixture` already compare scraped vs scheduled).
- Owner: **Gaffer**.
- Recommended outcome: a deterministic Phase-0 settlement probe in the harness — "all fixture games for the just-completed round have final scores on afltables" → proceed; else abort with a clear message. Weekday language removed from the skill, the header comment, and the standing anti-pattern list (amended below), all pointing at the one probe.
- Effort: S · Impact-per-day rank: 5

### F6 — Phantom-row gate placement: fires after ~2h of compute it invalidates  [severity: MEDIUM] [class: 2]
- Evidence: gate runs at `weekly_refresh.sh:66` — after `refresh_and_rank.sh` has completed scrape AND prediction AND backtest AND the Phase-1 commit. 07-13 run 1: scrape corruption existed from the start; gate fired 18:41 (log :18761) only after the full Phase 1; the re-run then repeated everything (Phase 1 re-completed 20:56). Push-safety is correct (S11-F7 retired — nothing reached origin); the cost is placement, not semantics.
- Impact: every scraper-integrity failure costs a full Phase-1 recompute (~2h wall-clock this cycle: 18:41 abort → 20:56 clean).
- Owner: **Gaffer** (harness sequencing), validator itself unchanged.
- Recommended outcome: the validator runs immediately after the scrape step (between refresh_and_rank [1/6] and [2/6]), before ranking/prediction/backtest spend their time; the pre-push placement can stay as a second cheap check.
- Effort: S–M · Impact-per-day rank: 6

### F7 — Prediction/backtest ground truth cited by a gated doc is untracked  [severity: MEDIUM] [class: 6-adjacent, provenance]
- Evidence: `git status` — `data/prediction/next_round_20_prediction_20260713_2050.csv`, `..._20260714_0730.csv`, and 8 backtest CSVs (20260710/20260713 timestamps) untracked; NOT gitignored (grep .gitignore: no match); 150 sibling files ARE tracked (historical convention). The shipped `afl-insights.md` methodology cites the 2050 CSV as a source; DataSentinel's PASS record cites it too. A fresh clone cannot re-verify the gated doc. `refresh_and_rank.sh:122` deliberately avoids `data/prediction/` as "scratch"; neither harness stages the canonical outputs. Gaffer already flagged this (`project_r20_cycle_selfcompleted.md:17`) — confirmed still open.
- Impact: provenance chain for a gated, published doc dangles; the backtest-by-archive path (refresh_and_rank :88-102) also DEPENDS on these archived CSVs surviving — an untracked archive is one `git clean`/re-clone away from forcing the 24-min/round retrain fallback for every historical round.
- Owner: **Gaffer** (harness staging), with Scientist confirming the filename patterns that are canonical vs scratch.
- Recommended outcome: Phase 1's commit stages this cycle's `next_round_<N>_prediction_<ts>.csv` and the four `backtest/*_<ts>.csv` outputs by explicit pattern (never `git add data/prediction/`), and the 10 currently-untracked files are committed.
- Effort: S · Impact-per-day rank: 7

### F8 — matches_2026.csv missing 9 games; the audit that detects them is warnings-only; published ladder claims computed on the gap  [severity: CRITICAL] [class: 1 + recurrence of S11-F1]
- Evidence (measured 07-14): round game counts — R10: 3 (audit WARNING log :177 "3/9 scraped, MISSING 6: Brisbane v Carlton; Essendon v GWS; Fremantle v Hawthorn; Gold Coast v St Kilda; Melbourne v West Coast; North Melbourne v Sydney"), R17: 4 (log :184 "4/7, MISSING 3"). `audit_match_rounds` fires every run and nothing consumes the warnings; the delta scraper is forward-only and never backfills. The published finals-doc ladder (Fremantle 14-2/56pts, Sydney 12-3/48pts) and the gated insights recap were computed from — and DataSentinel-verified against — this same incomplete CSV: the gate cannot see garbage-in consistency. Carried amber from the 07-13 PULSE; unfixed through two cycles.
- Impact: published ladder records for at least the 12 teams in the 9 missing games are potentially wrong on main; every downstream doc inherits it invisibly.
- Owner: **Scientist** (backfill the 9 rows + a backfill path for fixture-vs-scraped gaps); escalated to human as published-numbers-affected.
- Recommended outcome: the 9 games restored and re-derived ladder claims re-verified; the match audit promoted from warnings-only to a gate (or at minimum a backfill trigger) so a fixture-vs-scraped gap cannot persist across cycles silently.
- Effort: M · Impact-per-day rank: 3 (tied with F4; ordered below it only because it needs no design decision — the audit already names the exact missing rows)

### F9 — Cadence/QA claims drift across the meta-surfaces  [severity: LOW] [class: 7 + 9]
- Evidence: `refresh_and_rank.sh:12` says the parent adds "QA" (no QA phase exists); `weekly-cycle.md` description advertises a "QA gate" the harness doesn't run; header comment vs skill vs user feedback disagree on cadence (see F5); the R20 sentinel's `"qa": "PASS_WITH_WARNINGS"` came from a manual council chain, not the harness.
- Impact: the next operator (or agent) trusts a gate that isn't there.
- Owner: **Gaffer**.
- Recommended outcome: comments/skill text match the actual phase graph; if a QA phase is added under F4, the claims become true instead of deleted.
- Effort: S · Impact-per-day rank: 9

---

## Ranked fix list — "would have prevented manual intervention in this cycle"
1. **F2** — DataSentinel authoring checklist in FootyStrategy.md + one bounded harness retry (prevents interventions 2 and 3 — half of this cycle's manual work).
2. **F1** — Gate the HOF hub via its existing HOF-HUB sentinels + verdict record (prevents intervention 5 and the 07-14 Phase-4 death; fixes live wrong numbers).
3. **F4** — Fixture-date resolution for numeric-round rescued rows + date gate + harness QA phase (prevents intervention 4 recurring).
4. **F8** — Backfill the 9 missing matches + promote the match audit from warnings to action (published ladder correctness; would not have blocked this run but is shipping wrong numbers through it).
5. **F5** — Deterministic settlement probe replaces the weekday heuristic (prevents intervention 6).
6. **F6** — Move phantom gate to post-scrape (turns intervention-1-class events from 2h losses into minutes).
7. **F7** — Stage prediction/backtest CSVs by explicit pattern (closes the provenance dangle; protects the by-archive backtest path).
8. **F3** — Tee Phase-4 output + resume-from-phase (makes whatever still breaks diagnosable in one read).
9. **F9** — True up the meta-surface claims.

Intervention 1 (phantom gate + scraper fix) needed no new finding: the gate worked as designed and the fix (`a4c1fdc20`) is committed with tests — except its date residue, which is F4.

---

## Anti-pattern list (standing)
- Never trust an LLM sum; re-measure disputed numbers in pandas before acting.
- Never verify by exit code; re-read file content after any write.
- Never `git add .`; stage by explicit allowlist.
- Never hand-edit anything under `data/` or a generated table body.
- Never let a Pass-1 PASS stand in for Pass-2 clearance.
- Never soften an upstream caveat when translating numbers into prose.
- **AMENDED:** Never run the refresh before round settlement — settlement is a data condition (every fixture game for the round has final scores), not a weekday. The old "Tuesday 8 PM UTC" phrasing is retired: afltables settled Monday afternoon on 2026-07-13 (user-confirmed valid).
- Never push to `main` from parallel agents; serialize through one committer.
- Never define an agent's role, model, or tool scope in more than one place.
- Never leave a gate-enforced convention unwritten.
- **NEW:** Never stage a stamped doc in a harness allowlist without a same-cycle verdict producer for it — a staged-but-unverdictable doc is a scheduled Phase-4 failure (evidence: HOF hub, 07-13 and 07-14).
- **NEW:** Methodology/sources footnotes name files, never restate figures — DataSentinel's untagged-number scan is occurrence-based, and a restated number is a fresh occurrence (evidence: afl-insights FAIL 2, 07-13).

## Watch list (speculation, unranked)
- The 07-14 recovery commit `4c798e665` mixed remediation code, corpus fixes, and the dead Phase 4's staged docs under one message — if this becomes the recovery habit, history-based debugging degrades; watch the next recovery's commit shape.
- Two R20 forward CSVs now coexist (2050/0730); they are identical today (measured maxdiff 0), but the by-archive backtest picks `ls -t` newest — a future re-run that ISN'T identical would silently change which prediction gets scored.
- Nine new agent-memory files from this cycle are untracked; if the repo convention is VCS-shared memory, an unclean shutdown loses the R20 lessons.
- Hub prose beyond the sentinel cells (the Pendlebury five-category paragraph, kick-ratio arithmetic at :66) is hand-written `[data]` prose with no regeneration path — gating the cells (F1) still leaves the prose to drift (F14/F15 territory).

## What was checked and found clean
- Both R20 prediction CSVs: 413 rows each, predicted_disposals identical across all merged rows (maxdiff 0) — the Clarke fix did not perturb the forward prediction.
- The 07-14 DataSentinel verdict is genuine and fresh (checked_at 2026-07-13T21:44Z = 07:44 AEST 07-14, matches file mtime; 12/12 tags verified with pandas reproductions quoted in the record) — not a stale copy despite matching the 07-13 content.
- kicks-handballs / single-season commits on 07-13/07-14 changed ONLY the badge line, which `council-content-hash.sh` strips — their 07-07 PASS records legitimately still match (hashes 2e11ce26 / d2a760b2 re-computed and matched). No gate bypass occurred there.
- `git_commit_safe.sh` + pre-commit hook chain: fail-closed semantics verified by content (COUNCIL_COMMIT_AUTHORIZED guard, missing-check-script blocks, staged-blob hashing).
- Counter-aware delta fix `a4c1fdc20`: read in full; `_get_max_counter` + rescue logic sound, 3 TDD tests present; the phantom gate re-passed on real data 07-13 20:56.
- Phase-1 push deferral (S11-F7 fix) live-verified again: "Push deferred to parent harness" at log :61096, push only after gate at :61138.
- Freshness guard (`weekly_refresh.sh:96-100`) present and correct; round detection returned 20 on both runs.
- main == origin/main (0 ahead / 0 behind); no index.lock residue; no parallel-writer evidence this cycle.
- `enforce_news_limit` present (:195-228); news block untouched by the cycle, as designed.
- Clarke row: corpus date corrected by `a742b37bf` (row now 2025-08-27 per commit diff) — the C1 *on-disk* half is retired; the C1 *scraper* half remains (F4).
