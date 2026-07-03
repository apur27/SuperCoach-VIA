# Run Report — Harness Hardening Sprint 1

- **Date:** 2026-07-03
- **Cycle type:** harness-hardening sprint (not a weekly refresh — no scrape, no new predictions, no HOF recompute)
- **Commit:** `e84cd41f9` on `origin/main` — "Sprint 1: harden council pipeline (enforce=1, shared tag vocab, staged-blob gate, trust badge)"
- **Scope:** 33 files, +1062 / -26

## 1. Cycle Summary

This was an infrastructure sprint, not a data cycle: no round was scraped and no predictions or backtests were produced. Six council-pipeline hardening items (Q1, F1, F2, F4, F6, plus commit-authorization and trust-badge work) shipped under full TDD, taking the suite from its prior baseline to **239 passing tests in 1.15s**. The provenance stamp is now non-forgeable by default (`AUDIT_ENFORCE=1`), the `[data]` tag vocabulary has a single source of truth, and the stamp gate now verifies the staged git blob rather than the working tree. The headline outcome is not a code change but a data-integrity finding it exposed: **running genuine DataSentinel on 8 stamped-but-recordless legacy docs returned 6 FAILs** — six published "verified" documents were carrying stale or wrong numbers behind a text-only PASS stamp.

## 2. What Shipped

| File / area | Change | Key delta |
|---|---|---|
| `scripts/check-council-stamp.sh` | updated | `AUDIT_ENFORCE` default flipped `0→1` (line 35); a PASS stamp with no content-hash-keyed audit record now fails closed |
| `scripts/tag_vocabulary.py` | added | single matcher for bare `[data]` **and** `[data: spec]` tags; 151 lines, 181 lines of tests |
| `scripts/skeptic_sample_tags.py` | added | Skeptic sampler now uses shared vocab (was a bare-only regex that missed `[data: spec]` and went vacuous) |
| `scripts/inject_trust_badge.py` | added | counts genuine `[data]` tags via shared vocab; hash-neutral; zero-tag docs get no badge |
| `scripts/git_commit_safe.sh` | updated | only-Gaffer-commits enforced structurally (auth marker + `.githooks/pre-commit` block; human escape is `--no-verify`) |
| `.githooks/pre-commit` | updated | commit-authorization block (+13 lines) |
| `DataSentinel.md` / `Skeptic.md` / `Gaffer.md` | updated | agents enumerate tags via the shared CLI; DataSentinel masks tagged spans so its untagged-number backstop stays independent |
| 13× `docs/hall-of-fame-stat-*.md` | updated | trust badge injected (+1 line each) — e.g. games page: "✓ All 29 stats verified against source data · council-pipeline-gated · 2026-07-03" |
| `docs/news/2026-05-25-neale-daniher-tribute.md` | updated | trust badge injected (+1; "All 3 stats verified") — earned a genuine DataSentinel PASS |
| `docs/news/2026-05-29-greg-williams-possession-engine.md` | updated | trust badge injected (+1) — genuine DataSentinel PASS |
| `tests/unit/` (7 files) | added/updated | `test_tag_vocabulary`, `test_inject_trust_badge`, `test_staged_blob_check`, `test_commit_authorization`, `test_skeptic_sample_tags`, `test_requires_stamp_routing`, `test_council_stamp_audit` |

Total badged this run: **15 docs** (13 deterministic HOF stat pages + 2 news docs that earned a genuine PASS). Badges were **withheld** from the 6 legacy docs that FAILed (see §3).

## 3. Data Story — the 6/8 legacy-doc failure

The sprint's most important output is not in the diff; it is what the diff made observable. F1 ran genuine DataSentinel (Option A) against 8 legacy docs that carried a "DataSentinel: PASS" stamp but no backing audit record.

- **Result: 2 PASS, 6 FAIL.** A 75% failure rate on documents that were all publicly marked "verified." The badge backfill plus `AUDIT_ENFORCE=1` is what surfaced this — before this sprint the stamps were self-asserted text and nothing re-checked them.
- **The 6 FAILs split two ways.** Bucket 1 is **staleness drift** — current-season / active-player counts advanced since the doc was authored (`list-quality-draft-pipeline`, `free-agency-trade-window`, `5yr-grand-final-strategy`). Bucket 2 is **genuine authoring/data errors** frozen into the text (`jonathan-brown`, `dustin-martin`, `forgotten-heroes`).
- **Canonical case — `jonathan-brown-fist-of-god.md`.** Its "DataSentinel: PASS" was text-only, never backed by a record, and the doc's peak-triple line reads "232 goals, **46 Brownlow votes**" (confirmed present at line 161 and the cross-check footer line 241) where the correct figure is 49 — a 46≠49 arithmetic error published under a PASS stamp. This is Exhibit A for why `enforce=1` is not optional: a stamp you cannot trace to a content-hash record is a stamp you cannot trust.
- **`dustin-martin`** carried a "12 players" claim that should read 16; **`forgotten-heroes`** mis-counted retired players (Sewell / Chad Cornes / Hewett) with a genuine Hopper staleness drift on top.
- **The structural lesson:** staleness, not fraud, is the dominant failure mode — 3 of 6 FAILs are numbers that were correct when written and rotted afterward. A verification that only runs once (at authoring) is guaranteed to decay for any doc that references active-player or current-season counts.

## 4. Pipeline Health

| Gate | Status | Notes |
|---|---|---|
| Test suite | **PASS (239)** | 239 passed in 1.15s (`pytest tests/ -q`) |
| DataSentinel | **PASS** (this commit's docs) | 15 badged docs backed by genuine PASS; badges withheld from 6 legacy FAILs |
| Skeptic | N/A | no FootyStrategy draft in this sprint; sampler bug (missed `[data: spec]`) fixed under F2 |
| QA | N/A | infra sprint — no pipeline output artifacts to schema-check this cycle |
| `check_hof_numbers.py` | N/A this run | no HOF recompute; the 13 HOF pages are deterministic and badged on that basis |
| Provenance stamp (`check-council-stamp.sh`) | **ENFORCING** | `AUDIT_ENFORCE=1` is now default; staged-blob verification (F4), symlink-in-index fails closed |
| Commit authorization | **ENFORCED** | only-Gaffer-commits structural via auth marker + hook block |
| Legacy-doc audit (F1) | **6 WARNINGS** | 6 stamped-but-recordless docs FAILed genuine re-check; routed to Scientist (§5) |

Health read: **green on tests and gates, but the sprint deliberately opened a data-quality wound.** The 6 FAILs are not regressions introduced this run — they are pre-existing defects the new machinery finally made visible. They are correctly quarantined (no badge) and routed; none shipped a badge this cycle.

## 5. Backlog Delta

**Closed this run (commit `e84cd41f9`):**
- **Q1** — provenance stamp forgeable → stamp now trusted only if a content-hash-keyed audit record backs the staged content; DataSentinel records its verdict, deterministic HOF lane records its `check_hof_numbers` verdict so `enforce=1` doesn't brick the weekly refresh.
- **F2** — `[data]` tag vocabulary fragmentation → single source of truth (`scripts/tag_vocabulary.py`) shared by sampler, badge, and DataSentinel.
- **F4** — stamp gate verified working tree, not commit → now verifies the staged blob via `git cat-file`; symlink-in-index fails closed.
- **F6** — legacy pre-gate news exempt by exact filename; coaches-strategy-corner opt-in-sticky; FootyStrategy recap mandates bold `**[data]**`.
- Only-Gaffer-commits enforced structurally; trust-badge injector live.

**Newly surfaced this run (routed to Scientist):**
- **Bucket 1 staleness:** `list-quality-draft-pipeline`, `free-agency-trade-window`, `5yr-grand-final-strategy`.
- **Bucket 2 authoring/data errors:** `jonathan-brown` (46≠49), `dustin-martin` ("12 players"→16), `forgotten-heroes` (Sewell / Chad Cornes / Hewett retired-player counts wrong; Hopper genuine drift). **Priority: jonathan-brown + dustin-martin first.**

**Still open (top 3 by impact), queued for Sprint 2:**
1. **Q2a** — hard-abort audits + DOB-collision guard (highest: turns advisory audits into blocking gates).
2. **afl-insights.md deterministic recap gate + Phase-3 record wiring** — closes the biggest un-gated auto-generated surface.
3. **F9** — `weekly_refresh.sh` wiring for the new gates, so `enforce=1` and the badge injector run in the automated cadence, not just by hand.

Also queued: Q2b pandera-on-CSV-write, I1 executable-data-tags spike, renamed-doc `--diff-filter` gap (excludes `R`), F5 Skeptic-record, F8 architecture.md §5 drift.

## 6. Expansion Recommendations

**1. Build a scheduled staleness re-verification gate — `scripts/reverify_stale_docs.py` — driven by the F1 finding.** The 6/8 failure rate proves that one-time authoring verification decays; 3 of the 6 FAILs are pure staleness on active-player / current-season counts. Add a script that re-runs DataSentinel against every badged doc whose numbers reference active players or the current season, on a cadence (e.g. monthly, or after each weekly refresh), and **strips the trust badge + files a WARNING the moment a re-check FAILs**. Threshold: any badged doc with no PASS record newer than N weeks is auto-flagged. Impact: converts the badge from a one-time claim into a maintained guarantee — the single highest-leverage thing this run made buildable. Effort: 0.5–1 day (reuses `inject_trust_badge.py` + the DataSentinel CLI). **Why now:** enforce=1 + the shared vocab + the badge injector are all live, so the read/verify/strip loop has every component it needs; and F1 just proved the decay is real, not theoretical.

**2. Split the "verified" claim into two badge classes: `deterministic` vs `LLM-verified`, in `inject_trust_badge.py`.** The 13 HOF pages are regenerated deterministically from `_stat_leaders.json` and cannot drift silently; the 2 news docs passed via LLM DataSentinel and *can* rot (that is exactly what bit the 6 legacy docs). Right now both wear the identical "✓ All N stats verified" badge, which flattens a real trust difference. Give deterministic pages a badge that says so ("regenerated from source each refresh") and reserve the dated "verified on YYYY-MM-DD" phrasing for LLM-verified prose — so the badge's own text tells a reader which docs are self-healing and which need re-verification. Impact: makes the staleness risk legible on the page itself and scopes recommendation #1 (only the LLM-verified class needs re-checking). Effort: 2–4 hours. **Why now:** this sprint is the first time both classes coexist under one injector, so the distinction is newly meaningful.

**3. Add a `--diff-filter` renamed-doc gate to the pre-commit hook (Sprint-2 backlog item, promote its priority).** F4 hardened the stamp gate to check the staged blob, but the renamed-doc gap means a `git mv` of a council doc (diff-filter excludes `R`) can slip a stamped doc past the check under a new name without re-verification — the same "text-only stamp" hole F1 just found in legacy docs, but reachable going forward. Wire `--diff-filter=ACMR` (add `R`) into the hook and re-verify any renamed council doc. Impact: closes the forward-looking version of the exact defect class this sprint exposed retroactively. Effort: 2–3 hours. **Why now:** the staged-blob machinery (F4) is in place, so extending it to renames is incremental rather than net-new.

## 7. Forward Metrics

- **Legacy FAIL burn-down:** 6 open at this commit. Expect **2 closed** next cycle (jonathan-brown + dustin-martin are prioritized to Scientist); if that count does not drop, the routing-to-Scientist handoff is stalling.
- **Badged-doc count:** 15 this run. On the next weekly refresh with recommendation #1 live, expect the deterministic HOF pages (13) to stay badged automatically and the news-doc count to only rise via genuine PASS — a *drop* in badge count would signal a doc silently lost its record.
- **Leading indicator of a data problem:** the ratio of PASS records to badged docs. It should be 1:1. If a scheduled re-verify finds a badged doc with **no PASS record newer than its badge date**, that is staleness drift surfacing *before* it reaches a reader — the earliest possible signal, and exactly the failure mode F1 proved is dominant.
