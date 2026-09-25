# Data catch-up on 23–24 September 2026

This is the operational record for the owner's request to update the existing data immediately, alongside preparing the rewrite plan. It is separate from implementation of the replacement application.

## Scope

Refresh match results, team lineups and player source rows locally. Preserve existing local edits and forecast archives. Do not invoke the weekly publishing harness, LLM council, model training, Git commit/push or release deployment. Generated rankings, predictions, evaluation reports, charts and editorial documents keep their existing vintage until explicitly regenerated and verified.

Before refresh, the latest match date in `data/matches/matches_2026.csv` was **[data]** 23 August 2026. The previous completion marker was dated 29 August. The catch-up ran in an isolated copy, and validated raw CSV changes returned to the checkout on 24 September at 19:31 Australia/Melbourne. The old weekly completion marker remains unchanged because this was not a complete publication cycle.

## Procedure and evidence

1. Copied tracked working-tree files into an isolated `/tmp` checkout, preserving the original checkout's dirty state.
2. Created a temporary Python environment. A fresh unpinned install failed on a pandas 3 incompatibility; used pandas 2.3.3 to execute the legacy application.
3. Ran `refresh_data.py --allow-direct` in the isolated copy. The restricted-network attempt exposed a fail-open bug; reran with working source access and did not accept the failed attempt as a successful refresh.
4. Compared file hashes, schemas and row counts independently of the scraper's log.
5. Detected synthetic wildcard-final dates and source corrections missed by delta-only scraping. Reconciled recent-season player source pages using their actual discovered URLs and verified identities, with a global request limiter and retained raw-page hashes.
6. Validated new match/player/lineup relationships, dates, counting arithmetic and unchanged historical identities; preserved unchanged cell representations to avoid spurious CSV formatting changes.
7. Copied only the validated raw files back after checking every original file hash against concurrent modification and backing up the originals. Verified all resulting hashes and preservation of existing ranking edits, prediction archives and refresh markers.
8. Rechecked the [2026 AFLTables season page](https://afltables.com/afl/seas/2026.html) on 24 September: its completed-match links agreed with the refreshed season file's **[data]** 217 matches, latest **[data]** 19 September.

## Findings the rewrite must preserve as regression cases

- A failed source fetch can be reported as a clean audit and a zero-exit refresh.
- The existing player parser does not map `WF` to `Wildcard Final`; an otherwise successful run fabricates a date for those rows. Resolve each from the season match table, never from a round-to-weeks approximation.
- The career audit reconstructs URLs from names and can compare same-name players incorrectly or miss apostrophes/capitalization. Use the actual source URL and DOB/identity metadata.
- Date/counter delta selection misses corrections to already stored game statistics and later-populated award votes. A bounded recent-season overlap is required.
- Name-token matching misses multi-part surnames. Seven additional source aliases were verified explicitly. Two pre-existing duplicate player files were also identified; their handling is recorded below.

## Completion record

**Completed locally.** The accepted update changes 756 raw CSV files: one match-season file, ten lineup files and 745 player-performance files. No raw file was deleted. All existing rows were retained.

| Accepted change | Count |
|---|---:|
| Added completed matches | **[data]** 10 |
| Added player-game rows | **[data]** 460 |
| Added team-lineup rows | **[data]** 20 |
| Latest completed match | **[data]** 19 September 2026, 17:15 in the legacy source timestamp format |
| Corrected cells in previously stored rows | **[data]** 16,563 |
| Date cells corrected against uniquely resolved fixtures | **[data]** 15,596 |
| Brownlow-vote cells reconciled with the source | **[data]** 867 |
| Other corrected cells, including round labels and match statistics | **[data]** 100 |

Corrections were limited to the current and prior seasons; older existing rows were preserved. Source reconciliation covered 900 player pages: 779 matched recent player files, and 121 had no recent games. All 781 files containing recent-season rows are accounted for by those 779 verified files and the two preserved legacy duplicates below. No source request/parse failures or unresolved fixture dates remained in the accepted reconciliation. This coverage statement does not certify every historical source cell.

Validation passed for unchanged schemas, no deleted rows, unique row identities within changed files, no new career-counter gaps, new player dates matching fixtures, disposal arithmetic, exact agreement between new lineups and imported player identities, and player-goal totals reconciling to new match scores. The temporary reconciliation/validation helper tests passed **10/10**, and the existing lineup/match integration module passed **3/3** against the staged refreshed corpus. The wider pre-refresh application test results and their limitations remain in [AUDIT.md](AUDIT.md).

Machine-readable evidence:

- [Refresh manifest](evidence/refresh-manifest.json): every changed path, before/after SHA-256, row counts, correction totals, validation results, source-season recheck and copy timestamp.
- [Source reconciliation inventory](evidence/refresh-sources.json): source URLs, page hashes, outcome, alias evidence and preserved duplicates.
- [Cell correction ledger](evidence/refresh-corrections.csv): original/replacement values for every corrected existing cell, with source URL/hash. Newly appended rows are accounted for separately in the manifest.

Original changed files are backed up locally at `/tmp/supercoach-refresh-backup`; temporary paths are operational scratch space, not durable release storage. Preserve the manifest and accepted checkout bytes before implementing the rewrite.

## Remaining limitations and implementation handoff

Two pre-existing duplicate identities remain untouched because deleting or merging legacy records requires the rewrite's explicit identity migration and reconciliation:

| Preserved legacy file | Verified canonical file | Evidence |
|---|---|---|
| `green_william_08092005_performance_details.csv` | `green_will_08092005_performance_details.csv` | William/Will spelling variant, matching source DOB and overlapping game identity |
| `steele_roan_19092002_performance_details.csv` | `steele_roan_22102001_performance_details.csv` | Incorrect legacy DOB; source confirms the canonical identity and overlapping game identities |

Both live under `data/player_data/`; the source URLs and canonical mappings are retained in the evidence files. Legacy consumers that scan every CSV can still double-count these duplicates. The new importer must quarantine the duplicate inputs, reconcile overlapping game cells, and retain their provenance before producing canonical aggregates. It must not silently discard or merge rows by name.

Rankings, trained models, predictions, evaluation reports, charts and editorial publications **were not regenerated**. They retain their earlier vintage, even though the raw data has advanced. Application code and the weekly harness remain unchanged. No weekly publishing cycle, Git commit/push, public release or deployment ran. A complete future publication must retrain/rebuild as appropriate and pass its gates against an explicitly accepted snapshot.
