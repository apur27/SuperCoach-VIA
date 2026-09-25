# B1 bounded repair fetch (2026-09-25)

Owner decision on blocker B1: at most 10 AFLTables requests through the rate-limited
`HttpClient`, to fill the missing 2026 rows of Flynn Perez, Will Brodie and Jack Dalton
(the 2026 player, not the 1876 namesake). No number was typed by hand; every row below was
parsed from an archived payload and is re-verified on every re-import.

## Requests

6 network requests in total, one attempt per URL, 1 request/second.

| Run | URL | HTTP | sha256 (prefix) | bytes |
|---|---|---|---|---|
| 1 | https://afltables.com/afl/seas/2026.html | 200 | `6c0d5a3b` | 239,669 |
| 1 | https://afltables.com/afl/stats/games/2026/131820260329.html (R4 Port Adelaide v West Coast) | 200 | `08a2bf59` | 57,675 |
| 1 | https://afltables.com/afl/stats/games/2026/091020260406.html (R5 Geelong v Hawthorn) | 200 | `e1675c9d` | 56,990 |
| 2 | https://afltables.com/afl/stats/players/F/Flynn_Perez.html | 200 | `e8bf6034` | see manifest |
| 2 | https://afltables.com/afl/stats/players/J/Jack_Dalton1.html | 200 | `147ff4fa` | see manifest |
| 2 | https://afltables.com/afl/stats/players/W/Will_Brodie.html | 200 | `c02ddb7f` | see manifest |

Run 1 failed closed: the new match-page parser (until then verified only on synthetic
markup) rejected the real final-score cell `13.12.<b>90</b>`. Nothing was written. The
parser was fixed against the two archived pages (now unit-test fixtures), and run 2
re-served the three run-1 payloads from their archived bytes (no network) and fetched
only the three player pages. Player-page URLs were read from the match pages' hrefs, not
guessed (the 2026 player is `Jack_Dalton1`).

## Outcome

| Player | Identity | Proof | 2026 rows added |
|---|---|---|---|
| Flynn Perez | `legacy:perez_flynn_25082001` (existing) | page DOB 25-Aug-2001 = legacy file DOB | 7 |
| Jack Dalton | `src:afltables:J.Jack_Dalton1` (new) | page DOB 05-Apr-2007, not the 1876 namesake's | 5 |
| Will Brodie | `legacy:brodie_will_23081998` (existing) | page DOB 23-Aug-1998 = legacy file DOB | 2 |

Cross-checks: each existing player's page shows the same pre-2026 game count as the legacy
file (Perez 24, Brodie 54) and the career counters continue from it (Perez 25-31, Brodie
55-56). After the rows were merged and the 14 quarantined 2026 lineup tokens were re-linked,
Hawthorn's R17 and R18 goal totals reconcile and `validate_dataset` passes, with 0 blocking
and 0 error issues and 18 historical warnings.

## Files

- `fetch-manifest-run1.json`: run 1, the failed parse, with its request accounting.
- `fetch-manifest.json`: run 2, with targets, per-player evidence, source observations
  (fetched-at, sha256, HTTP metadata), the work log and the request totals.
- `raw/<sha256>.html.gz`: every payload, byte-exact.
- `rows.jsonl`: the 14 `player_games` rows and 3 `players` rows, each with source URL and sha256.
- `b1_repair_fetch.py`: the driver that ran. Re-running it performs network requests. Do not
  run it again without a new authorization.

Offline re-application is `ingest.refresh.replay_repair_evidence`, which the pipeline's
`apply-repair` step calls. It re-hashes every payload, re-parses the pages with no network,
and requires the result to reproduce `rows.jsonl` exactly, or it raises
`RepairEvidenceError`.
