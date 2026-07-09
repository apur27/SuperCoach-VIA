---
name: legacy-doc-staleness
description: Sprint-1 finding — 6 of 8 legacy "verified" council docs FAILed genuine DataSentinel; published stamps were text-only, numbers drift as the season advances
metadata:
  type: project
---

When `AUDIT_ENFORCE=1` + genuine DataSentinel ran on 8 stamped-but-recordless legacy docs (2026-07-03), only 2 PASSed. The other 6 carried stale or wrong numbers under a `DataSentinel: PASS` stamp that was **text-only, never backed by a record**.

**Why it matters:** the old stamp was forgeable/aspirational; nobody re-verified after the season advanced. Any doc citing current-season or active-player counts drifts within ~1-2 rounds; all-era rankings shift when a new season's data lands.

**Canonical case — "enforce=1 is not optional":** `docs/news/2026-05-29-jonathan-brown-fist-of-god.md` carried a PASS stamp yet contained a plain **arithmetic error** (2007-09 Brownlow votes stated 49, actual 46). A data refresh would NOT catch it — only a real verification gate does. This is Exhibit A for why the stamp must be record-backed.

**How to apply:**
- Treat every pre-2026-07-03 `DataSentinel: PASS` stamp as UNVERIFIED until it earns a fresh record.
- Two failure classes, don't conflate: pure staleness (data drift, class 1 — refresh fixes it) vs authoring/data errors (arithmetic, wrong scans, retired-player miscounts — a refresh will NOT fix; needs a real correction).
- Retired players' career counts should NOT drift — a mismatch there (Sewell/Chad Cornes/Hewett in forgotten-heroes) signals an original error or a data-source defect (appended/placeholder rows), worth investigation not a blind bump.
- Routed to Scientist in two buckets (authoring-errors first). See [[sprint1-and-backlog]].

## Remediation outcome (2026-07-03, Scientist) — NOT yet re-verified/re-committed
- **Fixed, pending DataSentinel re-verify → re-badge:** jonathan-brown (46; Cameron 88 at #9, Brown #10), dustin (peer counts 371/248/436/309 + Fyfe disp/g 23.4), forgotten-heroes (Sewell 199→200 & Chad Cornes 254→255 = **backfilled-finals-row data corrections, not drift**, orig were undercounts; Hewett is ACTIVE not retired = genuine drift; Hopper drift), free-agency (37 figures).
- **doc#4 list-quality: PINNED to a frozen R1-9 snapshot** (numbers correct at round_num≤9; methodology tightened). OPEN: its per-player squad figures (Pendlebury 435, Neale 308, Butters 152) are frozen too but will FAIL a current-data verify — need a frozen-vs-live ruling + DataSentinel round-cap.
- **doc#6 5yr-grand-final: ESCALATED, not edited** — the R15 fix-targets are themselves stale (live data now R17); no single round reconciles the doc (Collingwood +3.8 matches no round, likely a copy of the Q4-net). Needs Option A (re-derive all 18 clubs to R17 + round-cap) or B (freeze at R15).

## TWO gate findings this surfaced (backlog → Sprint 2)
1. **DataSentinel dropna-denominator false-FAIL:** it FAILed dustin's "12 players (200+g, 20+disp/g, 300+goals)" claiming 16, but 16 is a coverage-bias artifact (disp ÷ only recorded-games inflates era-limited players — Barassi 4.45 real → 22.6 partial). The doc's **12 is correct** under the fill-zero convention DataSentinel's own prompt mandates. A re-verify of dustin will false-FAIL again unless DataSentinel honours fill-zero denominators. (See Scientist memory `dropna_denominator_coverage_bias.md`.) This is a DEFECTIVE-GATE class per [[llm-datasentinel-arithmetic]].
2. **No frozen-snapshot verification:** DataSentinel verifies against CURRENT CSVs, so any intentional historical-snapshot doc (doc#4 R1-9, doc#6 R15) false-FAILs. Need a round-cap / as-of-date verification mode.
