# Pending decisions

Decisions required before blocked work can proceed. Pick up this file to resume.

---

## 1. `5yr-grand-final-strategy` — data basis

> **RESOLVED 2026-07-07 → Option A.** Re-derive all 18 clubs to end of R17; cap DataSentinel verify at round 17.

**Blocked work:** Scientist has the doc ready to re-derive; DataSentinel cannot re-verify until the round basis is set.

**Context:** The R15 fix targets are already stale (live data is R17). No single round reconciles the current figures cleanly.

**Options:**
- **A** — Re-derive all 18 clubs to end-of-R17; cap DataSentinel verify at round 17. Doc is live-correct. *(Gaffer recommends)*
- **B** — Freeze at R15; declare it a snapshot with explicit as-of date.

---

## 2. `list-quality-draft-pipeline` — frozen vs live

> **RESOLVED 2026-07-07 → Option B.** Re-derive per-player figures to live (R18): Pendlebury, Neale, Butters, etc. Team figures may remain at R1–9 if the article framing requires it, but per-player stats re-derive to R18.

**Blocked work:** Scientist froze the 24 team figures to R1–9. Per-player squad figures (Pendlebury 435, Neale 308, Butters 152) are also frozen and will false-FAIL a current-data verify.

**Options:**
- **A** — Freeze everything at the article's stated R1–9 basis; declare as-of round explicitly; cap DataSentinel at round 9. Consistent with the article's framing. *(Gaffer recommends)*
- **B** — Re-derive per-player figures to live (R17).

---

## 3. Era-boundary player inclusion in threshold counts

> **RESOLVED 2026-07-07 → Include.** Use dropna over recorded games only. Scientist must build the deterministic helper that encodes this rule and emits `N of M` alongside averages — a script requirement, not a prompt rule. Do NOT flip DataSentinel to fill-zero.
>
> **⚠ RE-OPENED for a publish call — helper built, live count is 17, not 16** (`scripts/era_boundary_threshold.py`, verified 2026-07-07). Two stacked movements: (a) the dropna-vs-fill-zero rule adds 4 era-boundary players, 3 of them **partial coverage** where dropna materially overstates the true career rate — Barassi (50 of 254 recorded → 22.6 dropna vs **4.45** fill-zero), Skilton (98/237), Bisset (179/207); (b) **Toby Greene** crossed the threshold live during 2026 (full coverage, qualifies under both conventions) and will keep moving each round. **Human call needed:** publish live **17** with the partial-coverage table, or freeze the rarity claim at a round where it is 16 via F02a. Also: the dustin-martin "genuinely combined disposal volume with goal-scoring" prose is contradicted by the coverage note for the partial-coverage players — needs FootyStrategy rework before ship. No doc corrected pending this call.
>
> **Decision 4 — RESOLVED 2026-07-07 (human) → Option A.** Publish the **live figure of 17** with a coverage table showing each qualifying player's recorded-game fraction (`N of M` per player, partial-coverage flagged). The deterministic era-boundary helper (`scripts/era_boundary_threshold.py`) is built and tested; the figure updates live each week. The dustin-martin interpretation prose must be reworked (FootyStrategy) so it does not claim full-career volume+scoring for the partial-coverage players — the coverage table stands alongside the claim. **F02 unblocked for dustin-martin at live basis 17.**

**Blocked work:** dustin-martin re-verification will keep FAILing under the current DataSentinel rule until this convention is settled. Affects any published threshold-count claim over era-boundary players.

**Context:** For players whose career spans a stat-recording gap (e.g. Barassi — most games predate disposal recording), a career per-game disposal average is literally uncomputable. Whether such players qualify for threshold queries like "players with 200+ games AND 20+ disp/g AND 300+ goals" is an editorial call:
- DataSentinel's current rule (dropna / skipna=True) computed **16** — includes era-boundary players, averaging only over their recorded games
- The doc's figure is **12** — implicitly excludes players without a full career's worth of recorded data
- **Do NOT flip DataSentinel to fill-zero** — that contradicts DataSentinel.md line 82 and the coverage-era memo (fill-zero deflates means and creates fake era differences)

**Options:**
- **Include** — dropna over recorded games only (gate's current behaviour → 16). Scientist builds a deterministic helper that encodes this and emits `N of M` alongside averages.
- **Exclude** — must have sufficient recorded data to qualify (doc's intent → 12). Scientist defines the coverage threshold (e.g. ≥80% of career games in the stat-recording era) and the helper enforces it.

**Once decided:** Scientist builds the deterministic helper (prompt text alone is fragile against pandas `skipna=True` default). Human editorial call determines the inclusion rule; Scientist implements it. Do not let this be resolved by a unilateral prompt edit.

---

## 4. As-of-date verification mode for DataSentinel (Sprint 2 design)

**Not blocked on a decision — needs a design pass before build.**

Surveyor recommended a machine-readable `<!-- verify-asof: round=9 -->` directive in the methodology paragraph. Guards to design in:
1. Directive is part of the content hash (changing as-of changes the record — not stripped)
2. Badge renders the as-of visibly: "✓ All N stats verified as of R9" — snapshot cannot masquerade as current
3. Cap applies to ALL source tables the doc's tags touch (matches + player_data)
4. Recurring live-re-verify gate must skip as-of docs (or it false-FAILs them against current data)
5. Start doc-level cap; per-tag cap deferred

Owner: Gaffer (directive parse + prompt rule) + Scientist (round-cap compute). Depends on decisions 1–3.

---

*Last updated: 2026-07-07 (Decisions 1–4 RESOLVED by human; F02 fully unblocked — era-boundary count = live 17, Option A. Route questions to Gaffer.)*
