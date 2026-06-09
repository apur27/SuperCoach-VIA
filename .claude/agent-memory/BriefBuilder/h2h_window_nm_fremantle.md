---
name: h2h-window-nm-fremantle
description: H2H window selection rationale for North Melbourne vs Fremantle matchups
metadata:
  type: feedback
---

## North Melbourne vs Fremantle (R14 2026)

Window: last 7 meetings (2019–2026).

**Why:** Both clubs are in rebuild eras that started from approximately 2019 (NM) and 2019–2021 (Fremantle). Pre-2019 meetings pre-date the current roster cores. 7 meetings gives enough context to show the full dominance picture (Frem 6-1) while staying within the modern player/coaching era. The 2020 Carrara COVID-bubble fixture is retained in the table but flagged as not suitable for venue analysis.

**How to apply:** For NM vs Fremantle matchups, use 2019 as the floor year. Flag the 2020 Carrara game as a COVID-bubble neutral venue. Non-bubble venue meetings: Perth Stadium (4) and Docklands (1), all Fremantle wins. The 2026 Hands Oval fixture is the first Tasmanian neutral venue meeting — no historical venue precedent available for this ground for this matchup.

## North Melbourne model coverage gap

NM are absent from R14 prediction CSVs (confirmed per [[model-coverage-gap-portadelaide]] memory). For any NM brief:
- Label all NM predicted-disposal cells `[unavailable — North Melbourne had R13 bye; no R14 model output]` (or adapt label to actual reason)
- Use last-5 form averages from performance_details.csv for NM player tracking
- NM is also absent from backtest_by_team files — no bias figure available
- Escalate to Scientist to confirm whether a separate model run covers NM

Related: [[model-coverage-gap-portadelaide]]
