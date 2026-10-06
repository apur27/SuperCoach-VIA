# AFL Tables reconciliation (`scvia reconcile-afltables`)

An operator audit, a correction tool, and a weekly gate. It compares the repository's player
statistics with a complete, captured copy of AFL Tables, per player, per game and per statistic,
then checks season, stint and career totals and printed averages. `compare` only reports;
`propose-corrections` and `apply-corrections` turn source-backed findings into edits (never a guess,
never a similarity score); `scripts/reconciliation_gate.py` re-audits the seasons a weekly scrape
changed before the scrape is pushed (see "Weekly gate" below). Nothing here publishes.

Design and review record: `docs/rewrite/afltables-reconciliation/DESIGN.md` and
`docs/reviews/AFLTABLES_RECONCILIATION_DESIGN_REVIEW.md`. Run evidence:
`docs/reviews/AFLTABLES_RECONCILIATION_RUN.md`.

**What a result means.** PASS means agreement with *captured AFL Tables evidence* inside the
declared availability scope. AFL Tables says its figures are unofficial and may contain errors,
so this is agreement with a reference, not proof of historical truth. The capture is
`reference_mode=observed_current`: pages are retrieved over a window (hours), not as a snapshot of
the `--through-date`.

## Commands

All paths absolute. `--run-dir` owns only new audit artifacts; inputs are read-only.

```bash
scvia reconcile-afltables plan    --data-root DATA --snapshot sha256:<id> --legacy-root REPO \
                                  --through-date 2026-09-30 --scope all --run-dir RUN        # offline
scvia reconcile-afltables capture --plan RUN/plan.json --allow-network --resume              # network
scvia reconcile-afltables compare --plan RUN/plan.json --capture-manifest RUN/capture/manifest.json \
                                  --out RUN/reports/cold-1 --workers 4 --cache RUN/cache-4   # offline
```

| Exit | Meaning |
|---|---|
| 0 | PASS (compare) or capture complete |
| 2 | invalid invocation, unsafe path, incompatible resume, tampered plan |
| 4 | FAIL: at least one confirmed discrepancy in a requested layer |
| 5 | capture directory locked by another process |
| 8 | UNKNOWN (compare) or capture incomplete/resumable (capture) |
| 9 | software failure; no completed report |

* `plan` pins the snapshot (every fragment hash), the legacy CSV membership and content, the
  policies/rules and the code that decides which pages are fetched. `plan_id` covers everything;
  `capture_identity` covers only what determines *which pages are fetched*, so comparison rules can
  be re-issued (a second `plan` in the same run directory) without invalidating a capture.
* `capture` makes one request at a time, at least two seconds apart, persists spacing and any server
  `Retry-After` in `capture/checkpoint.sqlite`, retries at most three times, never sends a conditional
  request, stops on a 403/challenge page, and pauses (exit 8) on rate limiting or repeated failures.
  Raw bodies are content-addressed under `capture/objects/`; every attempt is a line of
  `capture/observations.jsonl`. `--seed-from OLD/capture` reuses re-hashed objects from an earlier
  capture. Resume refuses if the capture-relevant files changed since the plan (`Capture._check_code`).
* `compare` never touches the network. Same frozen inputs give byte-identical canonical outputs for
  any `--workers`, cache state or `--previous`.

### Corrections

```bash
scvia reconcile-afltables propose-corrections --plan RUN/plan.json --capture-manifest RUN/capture/manifest.json \
      --report RUN/reports/cold-4 --out RUN/corrections/roundN --workers 4 --cache RUN/cache-4   # offline
scvia reconcile-afltables apply-corrections --changes RUN/corrections/roundN/changes.jsonl --layer legacy_csv \
      --legacy-root REPO
scvia reconcile-afltables apply-corrections --changes RUN/corrections/roundN/changes.jsonl --layer snapshot \
      --data-root DATA --expected-snapshot sha256:<audited id>
```

* Every change line names the finding, rule id, source URL and body SHA-256 it rests on, and the
  audited old value. `apply` checks each old value first; one conflict aborts before anything is written.
* Snapshot layer: upserts, then the full dataset validator, then a promoted child snapshot (the old one
  is kept). Legacy layer: byte-preserving CSV rewrite; only the named cells change.
* Only findings a rule can settle from the page are corrected (a cell, a proven zero, a missing
  appearance or match, a row date, a jersey, a stint total). Identity proposals bind a verified profile
  URL, repair a name/birth date when every local game sits in exactly one unclaimed profile, or retire a
  duplicate/stub file. Anything else stays a finding (`summary.json` `unsupported_fail_findings`).
* Corrections can uncover the next layer of findings, so audit again after applying and repeat until a
  round proposes nothing. The 2026-10 run took four rounds (`docs/reviews/AFLTABLES_RECONCILIATION_RUN.md`).

### Season scope and re-using a capture

`plan --scope seasons --season 2025 --season 2026` audits only those seasons: the capture fetches the
season pages, their match pages and the profiles of players in them (no letter census), and `compare`
judges cells, matches and season/stint aggregates but not careers. A seasons audit can PASS.

`plan --capture-plan OLD/plan.json` issues a new plan (new rules, corrected inputs) bound to an existing
capture, so a re-audit needs no new network pass; the plan records the current capture-code hashes
alongside the bound ones.

### Weekly gate

`scripts/weekly_refresh.sh` runs `scripts/reconciliation_gate.py --fix` after the match-completeness gate
and before the Phase 1 push. It audits the seasons whose rows in `data/player_data` or `data/matches`
differ from `origin/main` (override with `RECON_GATE_BASE`) against the accepted snapshot in
`RECON_DATA_ROOT` (default `var/finalized/data`), applies source-backed legacy corrections, re-audits, and:

| Decision | Exit | When |
|---|---|---|
| pass | 0 | the changed seasons agree with AFL Tables (corrections, if any, are committed) |
| warn | 0 | capture outage, an AFL Tables record that is itself incomplete, or no data root: fails open, loudly |
| block | 1 | a confirmed discrepancy, or a duplicate/unresolved player file, remains after corrections |

A block stops the cycle with the Phase 1 commit unpushed. Read `gate.json` in the run directory under
`var/reconciliations/afltables/gate/` (the log line names it); `reports/after-fix/findings.jsonl` lists
what remains. A scraper defect: fix it and re-run the cycle. A change on AFL Tables itself: route to
Scientist. A season's capture costs one polite request every two seconds (a full season is about
1,000 pages, roughly 35 minutes).

## Scope

Men's senior VFL/AFL premiership matches from 1897 to the inclusive `--through-date`, including
finals, drawn finals and replays. Excluded (and stated in every report): preseason, reserves,
representative football, AFLW, coaches, umpires, and mutable height/weight, fantasy scores,
predictions and locally invented proxy measures. Population = source census ∪ local players ∪
legacy CSV players; nothing is discovered from local files alone.

## Layers and verdicts

| Layer | What is compared |
|---|---|
| `snapshot` | the pinned immutable canonical snapshot (primary app data) |
| `legacy_csv` | the raw `data/player_data` and `data/matches` CSVs, as strings (blank stays blank) |
| release | not compared here: run `scvia check-integrity --scope full` separately and bind both input IDs |

Each layer has its own verdict; overall FAIL if any requested layer FAILs. A passing layer never
hides a failing one, and a confirmed FAIL stands even when other evidence is incomplete.

## Source cell semantics

Per appearance each of the 23 statistics resolves to exactly one state: `RECORDED_VALUE`,
`RECORDED_ZERO` (blank cell, team total non-blank), `NOT_RECORDED` (notes/era/column structure),
`NOT_APPLICABLE` (finals Brownlow votes), `NOT_APPLICABLE_DNTF` (credited game, player did not take
the field), `UNRESOLVED_BLANK`, `MALFORMED`, `SOURCE_CONFLICT`. Rules are executable Python
(`cells.py`); each carries a rule id into findings. The versioned data rules are
`config/reconciliation_rules.toml`; identity overrides `config/reconciliation_identity_overrides.csv`
(overrides resolve identity only and need a captured locator).

Season Brownlow votes before 1984 appear on profiles while per-game cells are blank: such a
statistic is `SOURCE_SUMMARY_ONLY`; the season value is the reference, and a local layer that
cannot reproduce it gets `LOCAL_MISSING_SUMMARY_VALUE` (FAIL). Both layers now hold those season
totals: the snapshot table `player_season_awards` and the legacy file
`data/awards/brownlow_season_votes.csv` (key player, season, club, award). Career Brownlow totals
in `analytics/players.py` add them, so a pre-1984 career is complete.

## Outputs (`--out`)

`report.json` (canonical, schema `schemas/reconciliation/report.schema.json`), `findings.jsonl`
(every finding, stable order and ids), `players.csv`, `coverage.csv` (formula-neutralised),
`summary.md`, `execution.json` (timings, cache, memory; not canonical) and `output-manifest.json`
(written last: the completion marker with every file's hash). An existing completed destination is
refused. Findings carry the exact source URL, body hash, table/row/column and the local fragment or
file/row, so a reviewer can open the evidence.

Accounting identities (appearances, cells, source states, aggregates, matches) must hold; a violation
exits 9 rather than publishing a verdict.

## Known limits

* AFL Tables may itself be wrong; derived printed figures that disagree with its own game cells are
  reported as `SOURCE_DERIVED_INCONSISTENCY` (info) and never change a local verdict.
* Each match page's "Player Details" table prints career games-to-date; it must equal the profile's
  row counter for that game (`R-SOURCE-COUNTER`), and a profile's counters must not skip or repeat
  (`R-SOURCE-COUNTER-SEQUENCE`). Either disagreement is a `SOURCE_CONFLICT` (UNKNOWN) and blocks PASS.
* A capture interrupted during the end-of-acquisition revalidation is not complete: the manifest counts
  unfinished revalidations as pending, and `compare` also requires `capture/receipt.json` to say `complete`.
