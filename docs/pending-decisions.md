# Pending decisions

Decisions required before blocked work can proceed. Pick up this file to resume.

---

## 1. `5yr-grand-final-strategy` — data basis

**Blocked work:** Scientist has the doc ready to re-derive; DataSentinel cannot re-verify until the round basis is set.

**Context:** The R15 fix targets are already stale (live data is R17). No single round reconciles the current figures cleanly.

**Options:**
- **A** — Re-derive all 18 clubs to end-of-R17; cap DataSentinel verify at round 17. Doc is live-correct. *(Gaffer recommends)*
- **B** — Freeze at R15; declare it a snapshot with explicit as-of date.

---

## 2. `list-quality-draft-pipeline` — frozen vs live

**Blocked work:** Scientist froze the 24 team figures to R1–9. Per-player squad figures (Pendlebury 435, Neale 308, Butters 152) are also frozen and will false-FAIL a current-data verify.

**Options:**
- **A** — Freeze everything at the article's stated R1–9 basis; declare as-of round explicitly; cap DataSentinel at round 9. Consistent with the article's framing. *(Gaffer recommends)*
- **B** — Re-derive per-player figures to live (R17).

---

## 3. Era-boundary player inclusion in threshold counts

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

*Last updated: 2026-07-03. Route questions to Gaffer.*
