# Astra final design review

3 October 2026. Reviewed the real candidate site served locally under `/SuperCoach-VIA/`, release `20261003T113058Z-7662fe99ba96`, against [the ordered plan](ASTRA_SOL_USABILITY_PLAN.md). Read-only review; no source/data edits.

## Verdict

**PASS — no unresolved presentation defects requiring changes before publication.** The final rebuilt site satisfies the corrected design plan. This verdict covers presentation; the parent retains responsibility for the complete regression, accessibility, preservation, integrity and publication checks.

## Corrections verified in the final build

1. **Mobile statistic headings:** the first review found that pinned identity cells obscured long rightmost headings at 320px. The final build wraps these headings and uses narrower Season/Date cells. Rechecked history, career, seasons and game-log tables at 320/390px in both themes. “Uncontested possessions per game” and “Time on ground pct” are readable at the far-right limit while row identity and values remain visible. Corrected evidence uses `after-final-*`; earlier `after-*` captures preserve the defect for comparison.
2. **Build metadata semantics:** parent code review established that `freshness.validation_state` records whether the build used the locally selected snapshot. My initial plan incorrectly interpreted it as release consistency. The corrected plan and final rendered page now use “Snapshot selection” with an accurate explanation, and “Freshness at build.” The independent release-consistency audit is a separate result. Confirmed the exact copy in the final disclosure, with the provisional audit-failure notice still visible and `noindex` present.

Minor existing behaviour: captions scroll out of view when a wide table is scrolled horizontally, leaving the caption strip blank. Adjacent section headings and labelled scroll regions retain context. This does not prevent reading or navigating the data and is not a blocker for this bounded change.

## Passed design and interaction checks

- Inspected actual homepage screenshots at 320/390/768/1440 in light and dark, plus the open mobile menu and table screenshots. Clear purpose, search, direct destinations, grouped sections and balanced leader previews replace the earlier long report-like layout.
- Header now measures 104px at 320/390 (previously 169/137px). Brand and Menu share one row; 768px uses deliberate disclosure navigation. Desktop navigation aligns with body content.
- At 390px, the homepage search is visible in the first 900px viewport. All sampled home widths had zero document horizontal overflow and no browser page errors.
- Provisional audit-failure warning remains visible, including snapshot/build dates and Audit details. Freshness is labelled separately. The real candidate still states there is no current forecast because no valid future fixture is available.
- Five season-leader rows appear in the preview. Keyboard-opening the native disclosure exposes all 40 supplied rows. No invented player values are introduced by this presentation change.
- Real homepage search for Pendlebury opens Scott Pendlebury. Keyboard activation of the Game log local link reaches `#log-h`. Menu Escape returns focus to Menu.
- History/player tables keep identity cells visible in both themes at 320/390. Rightmost values remain reachable. Keyboard ArrowLeft scrolls each tested region back from its right edge. History supporting columns remain present on mobile.
- Homepage article titles/dates remain accessible without exposing internal publishing-chain excerpts. All articles link is present.

## Evidence

Local evidence is under `var/ui-review/20261003/` (outside Git).

- Initial interaction checks: `astra_final.mjs`, `astra-final-checks.json`.
- Definitive final recheck: `astra_final_recheck.mjs`, `astra-final-table-detail.json`.
- Final user-facing homepage screenshots: `screens/after-final-home-390-light.png`, `screens/after-final-home-390-dark.png`, `screens/after-final-home-1440-light.png`, `screens/after-final-home-1440-dark.png`.
- Corrected table evidence: `screens/after-final-{history,career,seasons,gamelog}-heading-right-{320,390}-{light,dark}.png`.
- Corrected metadata: `screens/after-final-metadata-390-light.png`.
- Homepage: `screens/after-home-{320,390,768,1440}-{light,dark}.png`; viewport crops use `after-home-top-*`.
- Menu/disclosure: `screens/after-menu-390-light.png`, `screens/after-leaders-expanded-390-light.png`.
- Tables: `screens/after-{history,player}-right-{320,390}-{light,dark}.png`; focused header captures use `after-{history,career,seasons,gamelog}-heading-right-*`.

Broader full-site accessibility, both-base regression suites, release integrity, source preservation and publication checks are owned by the parent/Sol. This is a presentation verdict and does not change the known source-audit FAIL or the meaning of release-consistency PASS.

## Operator validation

The final release is `20261003T113058Z-7662fe99ba96`, built under
`/SuperCoach-VIA/`. The repository remains `apur27/SuperCoach-VIA`.

- Web type checks and lint passed; 181 unit tests passed, with two existing skips.
- The complete two-base browser suite passed 354 tests, with four existing skips.
- The focused table and keyboard checks passed 40 repeated cases.
- The final real-site browser sweep passed 184 page/viewport/theme states, with
  no unexpected request errors or document overflow. Accessibility scans across
  six routes in both themes found zero violations. Search, watchlist persistence,
  Grand Final/player navigation, all 14 downloads and four no-JavaScript pages passed.
- The sealed site is 264,070,775 bytes, below the unchanged 300 MiB budget.
- Seal and release validation passed. Seal:
  `65bbef200a2326c697240f0b0813f06d3c47c28ec700a049d918348ba24d9201`.
- The full input-consistency audit passed all 26 checks, with
  `semantic_complete=true`, five warnings and one informational finding. Report:
  `d1ae0b0de3aac1f25952156820cfc45c4687d4b62e06821383c62a58943a51ad`.
- Hash checks found no changes in 697 protected files: 90 files in Claude's
  unfinished worktree, 604 retained candidate files and three existing memory files.
  Claude's worktree status was also unchanged.

The independent AFL Tables source audit still FAILs. This presentation work does
not correct that dataset or close acceptance of Claude's unfinished reconciler.
No numeric pipeline, harness, hook, schedule or source dataset was changed.

## URL case handling

The repository name and project base remain `SuperCoach-VIA` and
`/SuperCoach-VIA/`. A separate owner Pages site serves a lowercase entry and a
case-aware fallback. Its source and publish allowlist are in
[ops/pages-alias](../../ops/pages-alias/README.md).

The redirect preserves the remaining path, query and fragment. Canonical and
unrelated paths do not redirect. A plain project link is available without
JavaScript. Deep/mixed-case aliases can initially return HTTP 404 before the
browser redirects; this does not change the host's case-sensitive routing.

The alias passed 23 deterministic path checks and nine browser checks. Two
additional browser regressions verified exact-case assets and player/match
links under the unchanged root and `/SuperCoach-VIA/` fixture bases.
