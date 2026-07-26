# Run Report — Gate Integrity Hardening

- **Date:** 2026-07-27
- **Cycle type:** gate-integrity hardening (not a weekly refresh — no scrape, no new predictions, no new round scored)
- **Commits:** `a46ca60e8` → `4cf8a7da2` on `origin/main`, six commits
- **Scope:** 118 files, +5801 / -665 (`git diff --stat a46ca60e8^..4cf8a7da2`)
- **Preceding cycle:** R21 weekly refresh shipped 2026-07-20 (`64b366702`) and has its own record; this session began as a Gaffer idle-state status check on the repo afterwards.

## 1. Cycle Summary

This was an integrity cycle, not a data cycle. No round was scraped and no prediction was produced. What shipped instead is the machinery that was supposed to be checking the last several cycles and silently was not. A Gaffer status check found the published R20 backtest row carried tainted provenance — computed from a corpus that later failed the phantom-row gate, before the dtype-blind dedup fix (`8badf8dc5`) landed, then re-shipped by a retry run that treated the tainted artifact as "already complete." Two independent Surveyor sweeps then found the Hall of Fame profile regeneration and its `check_top100_consistency()` gate had never been invoked by either harness script, and that the root `README.md` stat block had no regenerator at all. Six commits closed those out. The suite went from 384 to **511 passing tests in 15.13s** (498 fast tier + 13 new integration tier), and two new test tiers exist that did not before.

## 2. What Shipped

| File / area | Change | Key delta |
|---|---|---|
| `scripts/backtest_completeness.py` | added, then fixed | 227 lines: completion manifest so a FATALed run cannot be read as complete. `71fa430b3` then fixed its own bootstrap — it adopted every timestamp on disk, grandfathering the two orphan round-1 vintages it exists to catch |
| `data/prediction/backtest/completed_runs.json` | regenerated | 14 summary-backed timestamps; −7 orphan entries removed |
| `data/prediction/backtest/prediction_vs_actual_round_1_2026_{20260518_131738,20260525_182141}.csv` | deleted | 2 orphan vintages, 231 rows each |
| `docs/hall-of-fame-top100.md` | regenerated (gate made reachable) | **7 profile stat-lines rewritten.** Pendlebury 436→438 games, 208→209 goals, 11,069→11,108 disposals; Neale 310→312 games, 8,522→8,594 disposals; Bontempelli 274→276; Kennedy 331→333; Cripps 246→248. Rank swap at #39/#40 (Neale over Vallence) |
| `update_team_analysis.py` | updated | Top-30 deviation vintage selector was `rsplit("_", 1)` — discarding the **date** and sorting on time-of-day only. Now regex-matches full `YYYYMMDD_HHMMSS`; unparseable filenames skipped, not sorted as empty string. Also `1360`→`1,360` thousands separator |
| `.githooks/pre-commit` | updated (+97) | Python gate: previously returned 0 whenever no Markdown was staged, so a pure-Python commit bypassed it entirely. Now fail-closed, bounded by `COUNCIL_PYTEST_TIMEOUT` (300s default) |
| `tests/integration/` | added | New tier, 13 tests, 0.66s — asserts published artifacts against the **real** `data/` tree. Catches "the generator is correct but was never run" |
| `tests/fixtures/golden/` + 2 golden test modules | added | Regression tier for player dedup / ranking / feature engineering |
| `scripts/weekly_refresh.sh` | updated (+57) | Phase 3c: Skeptic adversarial pass on `docs/afl-insights.md` (previously DataSentinel-only, exempt); verdict read deterministically via new `scripts/skeptic_verdict.py` |
| 18× `data/lineups/team_lineups_*.csv` | allowlisted | Previously drifting uncommitted every cycle |
| `README.md` | updated (+31) | Two contradictory bias values reconciled (−0.093 → −0.110); team bias span −0.73 → −0.58 (St Kilda); "all 13 rounds" → "all 20 rounds"; council registry / quick-start / brief-index fixes |
| `docs/afl-backtest-2026.md` | updated (+50) | R18 coverage gap documented as accepted limitation |
| `docs/afl-insights.md` | updated | 4th Skeptic pass cleared the last unsupported causal claim (see §3) |
| `CLAUDE.md` | updated | Test budget 10s → ~20s, two-tier split documented |

## 3. Data Story

- **The HOF gate was the one CLAUDE.md cites to justify an exemption.** `docs/hall-of-fame-top100.md` is explicitly exempt from inline `[data]` tags *because* `check_top100_consistency()` verifies it deterministically. That gate was never invoked by either harness. Seven profiles were live on main with stale numbers — Pendlebury's prose read 436 games / 208 goals / 11,069 disposals against 438 / 209 / 11,108 in the table.
- **Profile prose was drifting further than the tables it sits under.** Neale's stat-line said 8,522 disposals while his prose said 8,237 and "209 from 300 games" — the prose was stale relative even to the stale table. Two independent decay rates in one document.
- **The R18 gap is real and correctly priced.** R18-2026 scores **284** player-rounds across 14 clubs where 9 matches / 18 clubs were played — short **128** against the fuller vintage's 412. Geelong, Melbourne, St Kilda and Western Bulldogs are absent entirely. Cause: a `--from-csv` re-score of an archived forward CSV written before the full fixture existed. Accepted rather than fixed because substituting the fuller vintage moves nothing that the page claims: MAE 3.958→3.960, bias −0.110→−0.105, within-10 unchanged at 95.78%. The **team** table does move — St Kilda's season bias −0.583→−0.733 — which is why the limitation is documented on the page rather than left implicit.
- **One mid-session alarm was itself wrong, and the check caught it.** A sub-agent claimed live contamination in a published table. Surveyor's re-measurement found the selector had been resolving correctly — by luck, because the authoritative runs also happened to hold the later wall-clock time. The fix was still correct to make; the fire drill was not warranted. Worth recording that the verification layer corrected an over-claim in the direction of alarm, not just the direction of complacency.
- **The recap's last causal overreach came out on the 4th pass.** `afl-insights.md` had claimed contested possessions and clearances were "the same skill expressed twice" off r = **+0.75**; it now reads as association explicitly, not shared identity.

## 4. Pipeline Health

| Gate | Status | Notes |
|---|---|---|
| Test suite | **PASS (511)** | 511 passed in 15.13s. 498 fast tier (14.98s) + 13 integration (0.66s). Prior recorded baseline: 352 at r20 |
| Integration tier | **PASS (13)** | New this cycle — asserts shipped docs against real `data/` |
| Golden-file tier | **PASS** | New this cycle — dedup / ranking / feature-engineering regression |
| Pre-commit Python gate | **ENFORCING** | Fail-closed; was exit-0-if-no-markdown. Bounded at 300s |
| DataSentinel | **PASS** | `afl-insights.md` cleared |
| Skeptic | **PASS** | On the 4th pass (`4cf8a7da2`). Now wired into `weekly_refresh.sh` Phase 3c, fail-closed |
| QA | **PASS WITH WARNINGS** | Zero FAILs |
| `check_top100_consistency()` | **REACHABLE** (was unreachable) | Now invoked by the harness; 7 profiles corrected on first genuine run |
| Backtest completion manifest | **CLEAN** | 14 summary-backed timestamps; 2 orphans quarantined and deleted |
| Surveyor independent review | **VERIFIED ×3** | Initial pending-check, post-4-commit audit, final close-out. Every material claim held under direct re-measurement |

**Health read: green, and green for the first time honestly.** No gate regressed this cycle. The corrections shipped are all pre-existing defects that the new machinery made visible — the same shape as Sprint 1's 6/8 legacy-doc finding, one layer further in.

**The through-line, worth preserving as a class:** *a gate that cannot run is indistinguishable from a gate that passed.* Every headline defect this session was that same failure — an unreachable HOF gate, a pre-commit hook that returned 0 when it had nothing to check, `os.path.exists` guards that spelled "skip" as "pass", regexes that matched nothing and reported success, and a manifest bootstrap that endorsed precisely the orphans it existed to catch. The comment now in `.githooks/pre-commit` states it directly: "A gate that cannot run must not pass."

## 5. Backlog Delta

**Closed this run:**
- Tainted R20 backtest provenance → completion manifest (`a46ca60e8`), orphans quarantined and deleted (`71fa430b3`)
- HOF regeneration + `check_top100_consistency()` unreachable → wired into harness (`a46ca60e8`); 7 profiles corrected
- `afl-insights.md` exempt from Skeptic → Phase 3c gate added, fail-closed (`a46ca60e8`)
- Pre-commit hook blind to pure-Python commits → fail-closed Python gate (`fa76f13a3`)
- Two latent vintage-selector bugs → manifest bootstrap + top-30 date-discarding sort (`71fa430b3`)
- README stat block with no regenerator, self-contradictory → drift-gate + reconciliation (`24c047286`, `ce6ff6163`)
- 18 lineup CSVs drifting uncommitted → harness allowlist (`a46ca60e8`)
- Banner `aria-label` stale; test budget mismatch (10s stated vs reality at 511 tests) → both corrected

**Newly surfaced this run:**
- **`docs/afl-backtest-2026.md` now carries three mutually inconsistent headline figure sets.** The auto-generated table reports MAE 3.96 across R1–R20; a hand-written "Cumulative summary" block still says "Rounds backtested 13 (R1–R13)", 4,806 predictions, MAE 4.020, bias −0.097; the new R18 note reports 7,153 predictions, MAE 3.958, bias −0.110. The "Reproducibility" note is also frozen at R1–R13. This is exactly the README defect that `ce6ff6163` just fixed, in the doc the README links to — and it was not in scope.
- **Backlog IDs live in agent memory, not the architecture doc.** `docs/architecture.md` §4 (lines 224–297) contains no F13 and no SV26-N3; both live only in `Gaffer/project_open_backlog.md:19` and `Surveyor/survey_open_findings.md:75-78,87`. Separately, "F13" is reused in `docs/pending-tasks.md:85` for an unrelated Sprint-2 item — an ID collision.

**Still open (top 3 by impact):**
1. **F13 / SV26-N3** — the backtest still writes into the shared `next_round_*` namespace, and `_load_top30_player_deviation` still globs the backtest dir without consulting `completed_runs.json`. The manifest hides orphans from mtime consumers; it does not prevent the collision. Same latent class as this session's headline bug.
2. **~700 legacy garbage rows in committed lineup CSVs** — re-measured this run at exactly **700** across 18 of 24 files (6 defunct-club files are clean). Recipe already written at `docs/experiment-log.md:322–357`, predicate at 337–343. Never executed.
3. **Chart PNGs no longer reproduce byte-identically** (HOF, backtest accuracy) in the current environment. Uninvestigated — and reproducibility is the property the whole gate chain rests on.

**Deferred by explicit decision (not gaps):** `data/top100/yearly/year_2026.csv` regeneration moved to end-of-season cadence. R18 coverage gap accepted and documented. Skeptic SK-R21-04/05 logged non-blocking (stylistic).

**Also open:** gate verdict JSON does not persist full reasoning — only the commit message does.

## 6. Expansion Recommendations

**1. Point the README drift-gate at `docs/afl-backtest-2026.md` and delete the hand-written cumulative block.** (`ce6ff6163` already built the gate for `README.md`; extend `scripts/update_eval_surface.sh` to regenerate or assert the "Cumulative summary" and "Reproducibility" sections, and fail if any headline figure in the doc disagrees with the auto-generated R1–R20 table.) The doc currently publishes 4.020/−0.097/n=4,806 alongside 3.958/−0.110/n=7,153 and 3.96 — a reader has no way to know which is current. Impact: closes the last unregenerated stat surface in the repo, and it is the surface the README links to for detail. Effort: 3–4 hours. **Why now:** the gate, the tests, and the reconciliation pattern were all built this cycle for README; this is the same fix one file over, and the sweep just proved the drift is live.

**2. Make the top-30 selector consult `completed_runs.json` (SV26-N3), then isolate the backtest write namespace (F13).** (`update_team_analysis.py::_load_top30_player_deviation` — add a manifest membership check before the vintage sort; then give each backtest run its own output directory instead of writing into `next_round_*`.) This session fixed *how* the selector sorts but not *what it is allowed to select from*: an orphan landing between quarantine sweeps with a newer timestamp still wins. Impact: retires the entire vintage-collision class rather than filtering it downstream one consumer at a time. Effort: 4 hours for the manifest check, 1–2 days for namespace isolation — ship the check first. **Why now:** the manifest exists and is populated with 14 summary-backed timestamps; the consumer-side check is a small addition to infrastructure already paid for.

**3. Execute the lineup cleanup and add a garbage-row assertion to the integration tier.** (Run the recipe at `docs/experiment-log.md:337–343`; then add a test to `tests/integration/test_pipeline_artifacts.py` asserting zero `is_garbage` rows across `data/lineups/team_lineups_*.csv`.) The 18 files are now in the harness allowlist, so from this cycle onward they are committed every week — which means 700 known-bad rows are now actively maintained rather than merely present. Impact: removes the rows and makes their return a test failure instead of a survey finding. Effort: 1–2 hours (recipe is written; needs network for the backfill leg). **Why now:** allowlisting them this cycle changed this from dormant debt to recurring debt, and the integration tier that can hold the invariant did not exist a week ago.

**4. Persist full gate reasoning to the verdict JSON, not just the commit message.** (`scripts/record-sentinel-verdict.sh` / `scripts/skeptic_verdict.py` — store the `reason` field and the sampled spans for every verdict, PASS included.) Right now a PASS records that it passed, not what it checked. Impact: the audit trail becomes re-auditable — which is the precondition for ever trusting a gate you did not watch run. Effort: 2–3 hours. **Why now:** Skeptic took four passes on `afl-insights.md` this cycle and only the commit messages record why the first three did not clear.

## 7. Forward Metrics

- **Test count: 511 at this commit** (498 fast / 13 integration). Next weekly refresh should hold or grow. A drop, or a fast-tier runtime above ~20s, means tests were skipped or the budget needs another deliberate raise — not a trim.
- **`completed_runs.json` length: 14.** It should grow by exactly one per scored round. Growth by more than one, or any entry without a matching `backtest_summary_*.csv`, means a FATALed run is being adopted again — the bug `71fa430b3` just fixed at the bootstrap.
- **HOF profile diff on the next refresh should be small.** This cycle rewrote 7 stat-lines because the gate had never run. Now that it runs weekly, expect 0–2 changed lines per cycle (active players only). **A large HOF diff next run is the leading indicator** — it means the gate silently stopped running again, and it will show up here before it shows up on a reader's page.
