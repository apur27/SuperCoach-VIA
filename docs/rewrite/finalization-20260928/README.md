# Final local verification — 28 September 2026

> This is the record of the September 28 verification. Later source reconciliation found
> corpus discrepancies; see [current data status](../../../README.md#data-status).
> The results and artifact identities below remain specific to this earlier run.

This directory records checks on the recovered Claude/Grok implementation. The tested source inventory is `318c37e8b2a80d6f518f89cde1427923dc4d7a1a389224d58f81400442c8ea5a` (854 files). The scratch copy matched every selected source file. The independent Grok 4.7 review returned PASS for local merge; see `independent-review.txt` and `review-metadata.json`. The parent finalization document records the subsequent merge.

## Updated grand final

The fetched grand final has 46 player rows **[historical record]**. Every source statistic cell, including blanks, matches the accepted snapshot. Public match detail, player logs, dates, links and the sealed site's embedded resources agree. See `grand-final-verification.json`, `grand-final-source-statistics.json`, and `retained-grand-final-verification.json`.

Source: [AFL Tables match page](https://afltables.com/afl/stats/games/2026/081920260926.html). `grand-final-source-capture.json` records the capture. The legacy CSVs remain migration inputs. The season's `schedule_complete` and `source_status` remain unknown; the presence of the final does not change that policy. Forecasts are unavailable because the snapshot has no future fixture.

## Final artifact

- Release: `20260928T111014Z-ca96603163ad`.
- Snapshot: `sha256:aa836549e10e96a9039cc4642bbe563b94247e0bfaf95e745a0971cc0bb2b98f`.
- Seal: `bd9a655b06795c2d83ca77a48adb9df85fb2774463fa390f814dac2ff72474fa`.
- Archive: `59496b26d421998b874ea879277b63365bc49fa1427c95902565bc4abc9599f7`.
- Site: 276,821,039 bytes. Three additional seasons at the 2021–2025 mean, plus 5 MiB, project to 280.46 MiB; the limit stays 300 MiB and the headroom target stays 282 MiB.
- Retained outside the worktree at `var/finalized/` in the main checkout. The retained metadata here was captured before merge and deliberately has no merged commit yet. The live local metadata is updated after merge.

## Checks and scope

| Check | Result | Evidence |
|---|---|---|
| Full hermetic scvia tier | 695 passed, 29.33s on four distinct physical cores, four workers and native thread caps of one | `hermetic-four-cores.txt`, `cpu-topology.json` |
| Same tier, previous placement | 695 passed, 34.15s on CPU IDs 0–3, which are two physical cores with SMT | `hermetic-two-cores.txt` |
| Full integration before final freshness/provenance changes | 25 passed | `integration-before-last-fixes.txt` |
| Integration affected by those final changes | 9 passed, 49.92s | `affected-integration.txt` |
| Playwright production builds | 320 passed, 4 skipped | `browser-tests.txt`; web code unchanged by final fixes |
| Browser units | 163 passed, 2 skipped | `web-unit-tests.txt`; Python-produced release bridge also covered by pytest |
| Final-source Ruff and mypy | PASS | `ruff.txt`, `mypy.txt` |
| Real grand-final browser | Desktop and mobile, source caption, match/player navigation, no page errors or mobile overflow | `grand-final-browser.json`, screenshots |
| Final scratch weekly rehearsal | PASS, 620s, same source inventory, legacy parity, failed upload leaves live bytes intact, publish and rollback | `scratch-smoke-result.json`, `scratch-source-inventory.json`, `scratch-smoke.log` |
| Final site validation, sealing and archive round trip | PASS | `seal-result.json`, `validation-result.json`, `pack-result.json` |

The timing result is specific to the recorded machine and CPU placement. It does not guarantee the same duration on CI. No tests were removed or skipped to meet the budget. Existing test skips are visible in their reports.

The scratch rehearsal uses the captured pre-final corpus. It is not either of the two genuine shadow cycles needed for production activation. No host deployment, schedule change or default harness switch occurred. Existing published holdout metrics were not recomputed.
