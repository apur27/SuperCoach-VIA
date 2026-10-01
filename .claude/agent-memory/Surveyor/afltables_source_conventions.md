---
name: afltables-source-conventions
description: Measured AFL Tables page semantics (2026-10-01 live samples) — averages HALF_UP + denominators, summary-only pre-1984 BR, credited unused subs, replay not a token, homonym suffixes, robots 404
metadata:
  type: reference
---

Measured on 27 live GETs (2026-10-01; bodies hashed in `.claude/surveys/2026-10-01-afltables-reconciliation-design-survey.md` §8). Use as leads for the reconciliation acceptance review; re-verify against the frozen corpus, not memory.

- Printed averages: ROUND_HALF_UP (ties 28.125→28.13, 0.625→0.63). Season non-BR = total/GM, including credited all-blank rows. BR = total / home-and-away games. Career average excludes era-unrecorded games (McMullin TK 27/29). GM average = GM / distinct seasons. W-D-L average = win %. Denominators are never printed.
- Pre-1984 BR exists only as season-row totals (per-game cells blank 1935–83; 1933–34 per-game present). CORRECTED 2026-10-01 run 2: career Totals BR IS printed (Reynolds 154 = 31 per-game + 123 summary-only); my run-1 "blank" was a colspan misread.
- Career BR average denominator = all H&A games EXCEPT no-award seasons (Reynolds 154/234 excludes 1942–45); pre-1984 and zero-vote seasons ARE included (Sidebottom 16/56, Ablett0 100/232). Notes "BR from 1984" is NOT the BR denominator rule.
- Footer rows (Totals/Averages) start with a `colspan=3` label cell: always expand colspan before aligning to header labels.
- Printed season totals = game-cell sums 1,015/1,015 in samples; 0 counter gaps in 1,370 rows. Match page "games-to-date" equals profile Gm counter (Sexton 139).
- 2021–22 unused medical subs: listed on the match page with all 23 cells blank (no arrow, no %P). The profile credits the game and it counts in averages. Local rows: 202 in 2021, 183 in 2022.
- Drawn GF and replay: same "GF"/"Grand Final" token, distinct match URLs and dates; no "Replay" text anywhere.
- Profiles: every game row's `Rd` cell links the match page (1,370/1,370 rows, 1897–2026). Header colspan 28 in all eras. No rowspans. Some profiles lack DOB (Kelly Robinson). Rows carry no dates.
- Directory: no DOB. Homonyms are always fully suffixed (X0, X1 …); the base URL 404s. Each letter regenerates separately. Multiword surnames diverge locally (Ah Chee, De Abel, El Achkar).
- Notes: availability table 1965–2010 only. Labels I5/OP. Club names are anachronistic ("1975 Sydney"). "In most cases" hedge on averages.
- robots.txt is an HTTP 404 (custom HTML "Broked!" page, the same body as a missing profile).

**How to apply:** At final acceptance, check that the implementation models these as explicit rules with fixtures (findings S-01…S-05 of that survey), not as tolerances or convention-shopping. Related: [[survey-open-findings]].
