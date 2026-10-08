# Data integrity checker (`scvia check-integrity`)

> **Current data status (7 October 2026):** the corrected candidate and current legacy
> CSVs have an **UNKNOWN** source verdict: zero confirmed discrepancies and 63 unresolved
> source cells per layer. Coverage combines the full parent audit with the corrected
> candidate's 2026 audit; the [coverage receipt](reviews/evidence/afltables-candidate-source-coverage-20261007.json)
> preserves both original report identities. The older hosted preview retains its
> snapshot's FAIL notice. A release consistency PASS does not resolve source uncertainty.
> See the [dated status](../README.md#data-status) and
> [provisional Hall of Fame tables](hall-of-fame/provisional/README.md).

*Operator and engineering document for the rewrite package. Numbers about the corpus carry
**[data]** tags; timings and byte counts are engineering measurements, not player statistics.*

`scvia check-integrity` audits one pinned snapshot and, in full scope, the release built
from it. It is deterministic, offline and read-only. It makes no model calls and no network
requests, it never loads a model, and it never writes into its inputs. Identical input
bytes and options produce byte-identical canonical reports. That holds across repeated
runs, relocated directories, directory-iteration order, host time zone, worker count and
a cold or warm cache (tests in `tests/scvia/unit/test_integrity_determinism.py` and
`test_integrity_cache.py`).

The checker is **opt-in**. No harness, hook, schedule or gate runs it. It does not change
what `validate-release`, `publish` or the weekly harness accept.

## Command

```bash
scvia check-integrity \
  --data-root /abs/var/finalized/data \
  --snapshot current \
  --release-dir /abs/var/finalized/releases/20260928T111014Z-ca96603163ad \
  --scope full \
  --as-of 2026-09-28T12:00:00Z \
  --evidence /abs/checkout/docs/rewrite/evidence/b1/raw \
  --content-root /abs/checkout --content-manifest /abs/checkout/config/public_content.toml \
  --report /abs/var/reviews/opus55/<run>/integrity-full.json \
  --findings-stream /abs/var/reviews/opus55/<run>/integrity-findings.jsonl \
  --cache /abs/var/reviews/opus55/cache \
  --workers 4 \
  --json
```

| Option | Meaning |
|---|---|
| `--data-root` | Data root holding `current.json`, `snapshots/`, `fragments/`, `raw/`; also the default `models/` and `predictions/`. |
| `--snapshot` | `current` (the pointer is read **once**) or an explicit `sha256:<id>`. |
| `--release-dir` | The release directory (`public/`, optional `site/`, `checksums.json`, `seal.json`, `validation.json`). Required for `--scope full`; without it every release check is UNKNOWN and the audit is incomplete. |
| `--scope` | `full` (snapshot + release + models) or `data` (no release checks are requested). |
| `--as-of` | An explicit UTC instant for time-dependent rules. Without it those rules are NOT_APPLICABLE; the current clock is never used in a verdict. |
| `--evidence` | Extra content-addressed source payload directories (repeatable). Files are found by SHA-256 as `<sha>`, `<sha>.html`, `<sha>.html.gz` or `objects/<sha[:2]>/<sha>`, and count only when their (decompressed) bytes hash to that digest. |
| `--models-root`, `--predictions-root` | Override `<data-root>/models`, `<data-root>/predictions`. The prediction artifacts are also the input a published prediction set is compared with. |
| `--evaluation` | An evaluation directory behind a published accuracy report (repeatable). Without it such a report cannot be compared and the audit is UNKNOWN. |
| `--live-root` | The live monitor's state root behind published live snapshots. Required only when the release publishes a live snapshot. |
| `--content-root`, `--content-manifest` | The curated article sources and `public_content.toml` the release's articles and assets were built from. Required when the release publishes curated articles or assets. |
| `--report` | Canonical report path. Refused (exit 2) if it resolves inside any input. Execution metadata goes to `<report>.execution.json` unless `--execution` is given. |
| `--findings-stream` | Every finding as canonical JSONL (the report's examples are capped; totals are always exact). |
| `--cache` | Semantic-result cache directory (outside every input). |
| `--changed-since` | A prior report. Needs `--cache`. See [changed-data mode](#changed-data-mode). |
| `--workers` | Processes for the release comparison. Does not change the report. |
| `--sample-limit` | Examples per rule in the report (policy default 20). |

`--json` prints one summary object on stdout: outcome, completeness, report digest,
snapshot and release identities, open findings by severity, check statuses and written paths.

## Outcomes and exit codes

Each check reports `PASS`, `FAIL`, `UNKNOWN` (could not run, or required evidence is
missing) or `NOT_APPLICABLE` (the input it needs is absent by design, such as no
`--as-of` or no prediction artifacts). A check `FAIL`s when it has an open **blocking**
finding. The overall outcome is:

| Outcome | Rule | Exit |
|---|---|---|
| `FAIL` | any check FAILs (required or not) | **4** |
| `UNKNOWN` | no FAIL, but a required check is UNKNOWN (incomplete verification) | **8** |
| `PASS` | every required check PASSed or is NOT_APPLICABLE | **0** |
| (usage) | invalid options, a policy file that does not parse, or an output inside an input | **2** |
| (checker failure) | the checker itself raised; no report is written and nothing is reported as a verdict | **9** |

Severity follows `domain.schemas.Severity` and the `validate_dataset` convention. A rule
marked "blocking in current season" is escalated when the finding's season is the current
season (the latest season with a completed match). `error`, `warning` and `info` findings
are reported with exact counts and **permit exit 0**, exactly as they permit promotion
today. Historical defects that the era policy tolerates therefore do not fail an audit;
current-season defects do.

`scope.complete` is true only when no check was restricted and no required check is
UNKNOWN. A `data`-scope report never claims release integrity, because release checks are
not requested at all.

Checks that executed are not the same as semantic coverage achieved.
`scope.semantic_complete` is true only when `release.coverage` PASSes: every public file
was examined by an executed semantic comparison against its own authoritative input, or
belongs to a type that is provenance-only by design (article prose, rendered chart images
and Markdown reports). A comparison that did not run, or whose input was not supplied, leaves
its resources uncompared, which makes `release.coverage` and therefore the audit UNKNOWN.
Byte, checksum and seal verification alone never count as a semantic comparison.

## What it checks

The checker pins the files selected by its requested checks. JSON, configuration,
article and table inputs are hashed and parsed from retained bytes. Model payloads are
hashed without loading them; source captures are checked against their pinned physical
and content hashes before parsing. Selection includes auxiliary content, predictions,
evaluations, live captures, policies and ranking configuration, as well as the snapshot
and release.

After the checks, selected inputs and directory membership are checked again. A changed,
deleted or newly selected input makes `inputs.stable` UNKNOWN and semantic coverage
incomplete; existing blocking findings remain FAIL. Paths and drift details belong in
execution metadata; canonical identities use logical names and content hashes so that
relocating identical inputs preserves the report. A change restored byte for byte before
the final check cannot be observed; semantic checks still use the captured content.
The existing gates are reused rather than reimplemented:

- `validate_dataset`, the promotion gate, is re-run on the pinned snapshot. Its issues
  are mapped as `dataset.*` with their own severity and acceptance. Rows it checked but
  did not list (it caps listings at 200 per historical season) are counted exactly,
  reported as `unlisted_by_producer`, and make the findings stream `complete: false`.
- `validate_release(write=False)` provides the allowlist, private-content, JSON, schema,
  manifest-reference and reference rules. Closure, hashes, the seal and the embedded
  site data are checked file by file from the captured inventory, so every differing
  file is its own finding.

Everything else recomputes an invariant from the facts, independently of the producer.

- **Stored bytes and contracts:** pointer and manifest identity (the semantic id formula
  is shared with the store: `storage.snapshots.semantic_snapshot_id`); fragment
  existence, containment, size, SHA-256, Parquet row count, schema and partition
  declarations; global key uniqueness; required non-null columns; enumerations;
  non-finite floats; JSON columns (duplicate keys and NaN refused).
- **Relationships and aggregates:** one player per match side, season/stage/date
  agreement with the match (`stage_label` vocabularies differ by design; `stage_id` is
  the identity), result versus score, lineup clubs, alias targets, cycles and ambiguity,
  and season rows recomputed from accepted match rows.
- **Football rules and era coverage:** player behinds never exceed the team's behinds
  (rushed behinds only add); Brownlow votes are 0–3; values above `plausible_max` raise
  a warning, never an error; zeros stored before a stat's `recorded_from` are reported
  (null converted to zero). Blank semantics are read independently of
  `domain/blanks.py`: AFL Tables prints 0 as a blank, so a null inside a match that
  reports the statistic, for a row that took the field, inside the recorded era, is
  reported as `football.blank_as_null`; a zero with no such evidence is
  `football.unevidenced_zero`; a finals Brownlow value is `football.brownlow_not_applicable`.
  Time on ground is never zero-filled.
- **Freshness:** `fixture_checked_at` must be backed by a PASS season-fixture observation
  and must not trail a later one. A partial fetch must not be marked fresh. The pinned
  season revision must be the content of the latest PASS observation. `schedule_complete`
  needs a source-declared status, `verified` status may not cover legacy rows, and no
  timestamp may be later than `--as-of`. Staleness is a warning. The pinned season page
  is read independently and compared with the accepted matches. A missing completed
  result, a scheduled fixture the snapshot lacks, an unexpected match, and each differing
  score, date, venue, attendance or stage are distinct findings.
- **Source → snapshot:** every captured match page (pinned, or behind a `source_fetch`
  row) is read by `integrity/sourcepages.py`. That reader shares no code with
  `ingest/afltables.py`: it uses the standard-library tokenizer and its own label table,
  and tests pit the two against real pages and a hand-annotated page with permuted
  columns. The checker compares match facts, player membership and identity (by source
  URL, else by unique name within the club), jumper numbers, and every statistic cell
  (blank means null). The page's own Totals/Rushed rows are checked against its player
  rows. Captures are counted under `coverage.source_capture`.
- **Captured player pages (B1 evidence):** `source.player_pages` reads each captured AFL
  Tables player page (adapter `afltables.player_page`) with the same independent reader and
  compares it with **every** canonical row of that player's career. A page lists a whole
  career, so it establishes row membership both ways; per row it establishes the link
  (season, club, opponent, round), the career game counter, result, jumper number and every
  non-blank statistic cell exactly. A blank cell establishes only that the count was not
  positive; whether it is a recorded zero is the match-level rule's decision. A page whose
  payload is missing is UNKNOWN when source-fetched rows depend on it. This covers the three
  B1 players only; it does not give other historical rows independent source evidence.
- **Release → canonical:** artifact binding checks that `release.json`, `checksums.json`,
  `seal.json` and `validation.json` name this release, this snapshot and these exact
  bytes. The public comparison then recomputes, from canonical rows, every season's
  match index, every match detail, every player season log and the player index, plus
  every player detail's career and season aggregates (observed denominators,
  `recorded_from` eligibility). Everything is compared cell by cell and checked for
  membership, order, compact-array width, canonical stat-column order, omitted all-null
  columns, shared match facts and unexpected or missing resources. JSON `true` is never
  equal to `1`.
- **Every other published resource (`release.derived`):** recomputed from the verified
  snapshot without calling the release builder or the analytics that produce it. Team
  pages: the regular-season ladder (4 points a win, 2 a draw, percentage, ordering), the
  club's position, form (last five completed matches), fixtures, per-team-game means with
  observed denominators, stat leaders, the five-year ladder view, and the numeric facts
  inside the finals-race heuristic (position and points; the clinched/eliminated status
  text is a labelled heuristic and is not audited). Team index membership and seasons.
  History: career, games and single-season leader tables for every era (membership, rank,
  value, denominators, coverage, warnings), the `legacy_v1` all-time and yearly rankings
  re-derived from the pinned `config/ranking_legacy_v1.toml` with the checker's own
  implementation of the documented formula, and the era summary. List seasons against the
  snapshot's draft, contract and school rows. `overview.json` (freshness, upcoming and recent
  matches, leaders, form highlights, forecast status, warnings) and `quality.json` (table
  counts, quarantine, issues, limitations). `downloads.json` against the files (bytes,
  SHA-256, CSV row count, kind, `as_of`, membership). Numeric downloads: `players.csv`,
  the ranking CSVs, `era-summary.csv`, the Brownlow proxy CSV (the published index formula,
  labelled as a proxy, never as votes), forecast CSVs against the published current set,
  `charts.json` plotted values, and every fan-pack member byte for byte.
- **Forecast, accuracy and live resources (`release.forecast`):** each published prediction
  set against its own prediction artifact (every row, target match, status and omission)
  and the index's status and reason against the fixture; each accuracy report's headline
  recomputed from the evaluation artifact's scored rows (populations and cohorts compared
  with the artifact); the legacy archive report re-joined from the snapshot's
  `legacy_predictions` by its documented rules; live snapshots against the monitor's last
  accepted capture. A resource whose input is not supplied is UNKNOWN.
- **Articles and assets (`release.content`), provenance only:** the index against the
  article files; a curated article's stated source path and SHA-256 against
  `public_content.toml` and the source file; a generated note's checked snapshot against the
  audited snapshot; asset bytes against their sources. Prose is not audited.
- **Coverage (`release.coverage`):** maps every public file to its resource type and
  authoritative input (`coverage.semantic.authority`), counts compared and provenance-only
  files per type, and reports a file of unknown type as `release.unclassified_resource`.
- **Models and predictions:** bundle manifest self-hash, streamed payload SHA-256 (the
  payload is never deserialized), the persisted feature spec rebuilt and fingerprinted by
  `ml.features`, feature order, the recorded code fingerprint, and a knowledge cutoff no
  later than `--as-of`. Predictions are checked for file hashes, labels, row identity,
  finite non-negative values, `0 <= low <= prediction <= high`, bundle eligibility at
  the forecast cutoff, and target matches present in the snapshot. An unavailable
  forecast is valid only if no scheduled fixture exists on or after its cutoff. The
  release's forecast status must match the artifacts.

### Numeric comparisons

Integer statistics compare exactly. In the match and player comparison the only tolerance
is on `time_on_ground_pct` **totals** (float64 sums), whose last bits depend on summation
order: `|a - b| <= 1e-9 * max(1, |a|)`. In the derived comparison a value that is a float
on either side (means, percentages, ranking scores, proxy index values) compares with
`|a - b| <= 1e-9 * max(1, |a|, |b|)`; integers, strings and booleans compare exactly, and a
timestamp compares as a UTC instant (the same instant written in another offset is equal).

### Semantic coverage and what is not audited

`coverage.semantic` lists, per resource type, the files, how many an executed comparison
examined and how many are provenance-only, and the authoritative input of each type:

| Resource type | Authoritative input | Compared by |
|---|---|---|
| match index, match detail, player season log, player index, player detail | snapshot matches, player games, players | `release.public` |
| team index, team season | snapshot clubs, matches, player games | `release.derived` |
| history index and tables | snapshot player games; `ranking_legacy_v1.toml` for rankings | `release.derived` |
| list index and seasons | snapshot draft, contract and school rows (pinned list imports) | `release.derived` |
| overview, quality | snapshot facts and manifest; the release's own verified forecast and article index | `release.derived` |
| `downloads.json`, numeric downloads, `charts.json`, fan pack | the files themselves; snapshot facts; the ranking method file | `release.derived` |
| prediction index and sets | prediction artifacts (evaluated model output) and the fixture | `release.forecast` |
| accuracy index and reports, `accuracy-rows.csv` | evaluation artifacts; snapshot `legacy_predictions` for the archive | `release.forecast` |
| live index and snapshots | live monitor accepted captures (not the snapshot) | `release.forecast` |
| article index, assets | article files; curated sources | `release.content` |
| articles | curated sources, provenance only | provenance |
| chart PNGs, Markdown reports, download README | rendered from compared values | provenance |
| `release.json` | the build record | `release.artifact` |

Deliberately not audited, and said so in `coverage.semantic.unaudited`: article prose
(curated articles are frozen at their own as-of scope; generated notes carry their own
claim check), rendered chart images and report prose (their numbers are compared in
`charts.json` and the source resources), and presentation labels such as table titles,
method descriptions and model-card prose (counted as `fields_unaudited`). Comparisons never
mix vintages: a live snapshot is compared with the capture it was published from, not with
the snapshot, and a proxy index is never read as an observed statistic.

Source coverage is separate from semantic coverage. Only rows on a captured match or player
page have independent source evidence (`coverage.source_capture`, `coverage.player_pages`);
everything else is `legacy_unverified`, as labelled. An absent optional historical capture is
a stated limitation, not a failure.

## Result contract

The report is `scvia.integrity-report/1`, validated against
`integrity/report_schema.py` before it is written. The generated JSON Schema is
`schemas/integrity/integrity-report.schema.json`; it lives outside the browser type
generator's directory scan. Top-level fields:

| Field | Contents |
|---|---|
| `checker` | version, SHA-256 over the code that decides verdicts, rules digest, rule count |
| `policy` | version and SHA-256 over `integrity_policy.yaml`, `coverage.yaml`, `stat_coverage_eras.yaml` |
| `inputs` | snapshot identity (selector, id, manifest and pointer digests, per-partition fragment hashes), release identity (checksums, seal, validation and full-inventory digests), evidence found/missing digest, selected model/prediction file digests, comparator inputs (evaluation files, live state and capture, content manifest and source bytes), and one `digest` over all of them |
| `scope` | name, `as_of`, families, `restricted`, `complete`, `semantic_complete`, current season |
| `outcome`, `counts` | overall outcome; checks requested/performed/by status; open and total findings by severity; rows, resources and cells examined |
| `checks` | per check: family, `required`, status, reason, examined counts, findings by severity |
| `rules` | per rule: kind, severity, exact `total`/`open`/`accepted`, `sampled`, `truncated`, `unlisted_by_producer` |
| `findings` | deterministic samples: the smallest keys per rule, with stable `issue_id` (`ic:` + hash of rule, entity, table, field), kind (`contradiction`, `anomaly`, `missing_evidence`, `policy`), severity, status, entity, table, field, season, expected, actual, evidence, message and operator action |
| `exceptions` | accepted, rejected (current season) and stale policy exceptions |
| `coverage` | source-capture and player-page accounting, public and derived comparison accounting, the typed `semantic` coverage summary, the reused `validate_dataset` verdict |
| `findings_stream` | when requested: count, SHA-256 and completeness of the JSONL stream |
| `report_sha256` | SHA-256 of the canonical report with this field empty |

A report digest is an evidence fingerprint, not a trust signature. A self-consistent
report proves only that nobody edited it after sealing, not that a statistic is correct.

Execution metadata (`<report>.execution.json`) is kept apart so that it cannot change the
canonical bytes. It holds start time, elapsed and per-check seconds, host, platform,
Python, CPU count, absolute paths, workers, peak RSS, input drift, cache statistics and
changed-data details.

## Policy and exceptions

`config/integrity_policy.yaml` (packaged copy identical) sets `sample_limit`,
`freshness.max_fixture_age_hours` (192), `plausible_max` per stat (set above the largest
value in the 2026-09-28 snapshot) and `known_exceptions`. An exception names an exact
`rule_id`, `entity` and `reason`. It is accepted only for a finding outside the current
season; one that names a current-season finding is rejected and raises a blocking
`policy.exception_rejected_current_season`; one that matches nothing is reported as
`policy.exception_stale`. `validate_dataset`'s own exceptions stay in `coverage.yaml`. Any
change to these files changes the report's policy digest and invalidates the cache.

## Cache and changed-data mode

`--cache DIR` stores the result of each comparison unit and of the `validate_release`
run. The comparison units are one per season (match index, details, logs), groups of up
to 600 players keyed by last season (detail pages), the player index, one per season of
team pages, the team index, the history tables, the list seasons, the summaries and the
downloads. A unit's key is
SHA-256 over all of the following:

- a salt: checker code identity, rules digest, policy digest, scope and `--as-of`;
- the unit's canonical dependencies: fragment hashes of every partition it reads, and
  the player and club rows it uses;
- the SHA-256 of every public file it reads, as measured **in this run**. An absent file
  is part of the key.

File names, sizes and modification times are never part of a key. Byte verification
always runs in full. Each entry carries a digest of its own body; a corrupt, truncated,
edited or misplaced entry is discarded and recomputed, and counted as `invalidated`. A
cache entry is trusted like local code, so keep the cache directory owner-writable only:
the digest detects accidents, not a forger with write access.

`--changed-since PRIOR --cache DIR` verifies the prior report's digest, checker code and
scope. It records the changed partitions (`player_games/2026`, for example) and the
unchanged ones, and whether the release inventory changed. The verdict is **always the
full audit**. The baseline only permits reuse of results whose content keys still match,
so every changed partition, every unit that reads it (for example the career page of a
player who played in that season) and every changed file is rechecked, and every global
check reruns. If the baseline is missing, corrupt or incompatible, the audit runs with the
cache disabled. The tests assert that changed-mode reports equal a fresh full audit of
the same final inputs.

## Determinism

- Inputs are pinned by content. `current` is resolved once, and the report is bound to
  the snapshot id, manifest digest and release inventory digest.
- There is no clock in a verdict: `--as-of` is explicit, and DuckDB runs in UTC with one
  thread, so float aggregation order is fixed.
- Findings are keyed and sorted by `(rule_id, entity, table, field)`. Samples are the
  smallest keys, so they do not depend on insertion order, worker count or directory
  order. Directory walks sort their entries.
- The canonical report holds no absolute path, host, time or cache statistic. JSON is
  sorted, compact and UTF-8, and NaN and Infinity are refused. Integral floats are written
  as integers only inside findings; the public-data comparison compares numerically.

## Measured on the follow-up candidate (30 September 2026)

Candidate release `20260929T213900Z-c8938f4ddb83` and snapshot
`sha256:3de6597513b5c6eaca1b102d304a2cf128d8c08f229ee1de384c1d40fe746db0`, same machine,
otherwise idle, `--as-of 2026-09-30T00:00:00Z`, B1 evidence and this checkout's curated
content. All checks, including the new derived, forecast, content, coverage and player-page
checks. Logs: `var/reviews/opus55/20260929T203757Z-followup/audit-final/`.

| Run | Wall | Peak process-tree RSS | Outcome |
|---|---|---|---|
| full, cold, 1 worker, filling a cache | 96.7 s | 1.61 GiB | PASS, complete, `semantic_complete` |
| full, cold, 4 workers, filling another cache | 88.5 s | 1.86 GiB | PASS |
| full, warm, 4 workers | 18.7 s | 1.25 GiB | PASS |
| full, `--changed-since` the cold report, 2 workers | 18.9 s | 1.25 GiB | PASS |
| full, cold, 4 workers, with findings stream | 86.9 s | 1.85 GiB | PASS |
| reference: the first retained inputs, 4 workers | 85.0 s | 1.84 GiB | FAIL (kept as reference) |

The four canonical candidate reports without a findings stream are byte-identical
(`sha256 5d038738…`). A copy of the candidate release with one team total edited and every
hash recalculated, audited `--changed-since` against the warm cache, recomputed exactly 2
cached results, reused 396, and FAILed `release.derived` with `release.team_value` on
`teams/collingwood/2026.json` (49 s).

## Measured on the retained artifacts (first review)

Release `20260928T111014Z-ca96603163ad` and snapshot
`sha256:aa836549e10e96a9039cc4642bbe563b94247e0bfaf95e745a0971cc0bb2b98f`, measured on an
i5-12450H (12 logical CPUs, 62 GiB) that was otherwise idle. Inputs: 408 fragments,
38 MB of Parquet and 1,506,178 rows **[data]**, plus 182,926 release files (581 MB). The
review report has the full run records.

| Run | Wall | Peak process-tree RSS | Notes |
|---|---|---|---|
| full, cold, 1 worker, with findings stream | 65.1 s | 1.51 GiB | `validate_release` 28.3 s, public comparison 23.0 s, capture 4.6 s, re-read 3.3 s |
| full, cold, 4 workers, filling the cache | 57.1 s | 1.86 GiB | public comparison 15.2 s; 263 units stored (6.8 MB cache) |
| full, warm, 4 workers | 15.4 s | 1.20 GiB | 263 units reused; `validate_release` result reused |
| full, `--changed-since` the warm report | 15.4 s | 1.19 GiB | no partition changed; every unit reused |
| data scope, 1 worker | 13.5 s | 1.20 GiB | no release checks requested |

All four full-scope canonical reports are byte-identical, except that the report written
with `--findings-stream` also carries the stream's block. Removing that block and
re-sealing gives the same bytes as the other three. The first implementation peaked at
2.58 GiB; dictionary-encoded strings and Arrow unit payloads reduced that, and
content-identity cache keys reduced the warm run from 26 s to 15 s. `counts.rows_examined`
sums each check's own count, so a row read by several checks is counted once per check.

Worthwhile optimisations not yet made:

- let `validate_release` reuse the checker's captured file hashes, since it re-walks the release twice (about 28 s of each cold run);
- cache the data-family checks (about 5 s) under the snapshot identity;
- skip the post-audit re-read when the caller accepts a copy-on-read guarantee (3.3 s).

## Example failure and operator response

On the retained snapshot, `freshness.fixture_inventory` FAILs with one blocking finding:

```json
{"rule_id": "freshness.fixture_value_mismatch", "entity": "match:m:2026:r17:collingwood:richmond:0",
 "field": "attendance", "expected": 62117, "actual": 0, "severity": "blocking",
 "evidence": {"sha256": "87c99c1a2eb84c4957759e5ec82e02aec97d7daf5264e1ba6db800cdd59bc83a"}}
```

The pinned 2026 season page (`87c99c1a…`) states 62,117 **[historical record]**; the
snapshot holds 0 **[data]**, from the legacy CSV, and the site shows "attendance 0". The
operator response:

1. Confirm that the evidence payload is the pinned revision; the digest is in the finding.
2. Repair the value from the pinned capture, offline: `scvia apply-corrections --season 2026 --data-root <root>`
   patches a fixture field only when the capture's match identity, scores, quarters and start
   time agree with the snapshot, records a resolved `fixture_field_corrected` quality issue
   naming the page URL and SHA-256, validates, and promotes a child snapshot. Rebuild the
   release from it. (A bounded network refresh also works, but a normal refresh does not
   re-read an unchanged match; that is why the 0 survived.)
3. Re-run the audit.

Do not add an exception: current-season defects cannot be suppressed.

Other findings on the same snapshot are reported but do not block. The 25 `relations.result_mismatch` errors **[data]** are replay player rows linked to drawn finals (1928–1990). The `football.player_behinds_exceed_team` error is 1990 QF Collingwood **[data]**, the same cause. The 22 `football.blank_as_null` warnings **[data]** exist because AFL Tables prints zero as a blank; see the review.

## Tests

- Hermetic unit tier: `tests/scvia/unit/test_integrity_*.py`. Every negative case has a clean control. `test_integrity_derived.py` changes team totals, history ranks and values, download row membership and stale summaries with every structural hash recalculated, in cold, warm-cache and changed-since modes, and requires the responsible semantic rule; it also checks that missing comparator inputs, or a comparison that did not run, leave the audit UNKNOWN. `test_integrity_source.py` covers the B1 player-page reader on the real archived page and the comparator's membership, value and blank rules.
- Real-data tier: `tests/scvia/integration/test_integrity_real.py`. It needs `SCVIA_INTEGRITY_DATA_ROOT` and `SCVIA_INTEGRITY_RELEASE_DIR` (and `SCVIA_INTEGRITY_AS_OF` for artifacts newer than the default), uses this checkout's curated content, and is skipped (never passed) without them. Point it at a correct artifact: the first retained artifacts fail `source.match_pages`.
- `tests/scvia/unit/test_integrity_catalog.py` keeps this rule catalog and the code in step.

## Rule catalog

### `storage`

| Check | Rule | Severity | Kind | Meaning | Operator action |
|---|---|---|---|---|---|
| `storage.identity` | `storage.manifest_identity` | blocking | contradiction | the manifest's content does not hash to its snapshot id | treat the snapshot as tampered; restore it from backup or re-import |
| `storage.identity` | `storage.manifest_invalid` | blocking | contradiction | the snapshot manifest does not parse under the contract | restore the manifest from backup; never edit manifests by hand |
| `storage.identity` | `storage.manifest_missing` | blocking | contradiction | the selected snapshot manifest is missing | restore snapshots/<id>.json from backup |
| `storage.identity` | `storage.pointer` | blocking | contradiction | current.json is missing, invalid or names another manifest | restore current.json from backup or re-promote the intended snapshot |
| `storage.fragments` | `storage.fragment_escape` | blocking | contradiction | a fragment path escapes the fragment store | treat the manifest as hostile; restore it from backup |
| `storage.fragments` | `storage.fragment_hash` | blocking | contradiction | a fragment's bytes do not hash to the manifest sha256 | restore the fragment from backup (bytes changed after sealing) |
| `storage.fragments` | `storage.fragment_missing` | blocking | contradiction | a referenced fragment is missing or not a regular file | restore fragments/ from backup; do not promote this snapshot |
| `storage.fragments` | `storage.fragment_rows` | blocking | contradiction | a fragment's Parquet row count differs from the manifest | restore the manifest or fragment; the declared row count is wrong |
| `storage.fragments` | `storage.fragment_schema` | blocking | contradiction | a fragment's column names or types differ from the table contract | re-import with the current contract; do not cast by hand |
| `storage.fragments` | `storage.fragment_size` | blocking | contradiction | a fragment's byte size differs from the manifest | restore the fragment from backup (truncated or replaced bytes) |
| `storage.fragments` | `storage.fragment_unreadable` | blocking | contradiction | a fragment is not readable Parquet | restore the fragment from backup |
| `storage.fragments` | `storage.partition_declaration` | blocking | contradiction | partition labels are missing, duplicated or present on an unpartitioned table | rebuild the snapshot so each partition is declared once |
| `storage.fragments` | `storage.partition_mismatch` | blocking | contradiction | a fragment holds rows of a different partition | rebuild the snapshot; a misplaced row is invisible to partition-scoped readers |
| `storage.fragments` | `storage.table_missing` | blocking | contradiction | a core table is absent from the manifest | re-import; a snapshot without core tables cannot be promoted |
| `storage.fragments` | `storage.table_row_count` | blocking | contradiction | a table's row_count differs from its fragments' rows | restore the manifest; the declared size is not the stored size |
| `storage.fragments` | `storage.table_unknown` | blocking | contradiction | the manifest lists a table outside the contract | re-import with the current code; unknown tables are not readable by consumers |

### `contract`

| Check | Rule | Severity | Kind | Meaning | Operator action |
|---|---|---|---|---|---|
| `contract.keys` | `contract.key_duplicate` | blocking | contradiction | two rows share a table key | re-import; resolve the duplicate identity at the source |
| `contract.keys` | `contract.null_required` | blocking | contradiction | a non-nullable column holds nulls | re-import; the contract forbids nulls here |
| `contract.values` | `contract.enum` | blocking | contradiction | a column holds a value outside its enumeration | re-import with a mapped value; never widen the enumeration to hide it |
| `contract.values` | `contract.json_column` | blocking | contradiction | a JSON column holds invalid JSON, duplicate keys or non-finite numbers | re-import the row from its source |
| `contract.values` | `contract.non_finite` | blocking | contradiction | a float column holds NaN or infinity | re-import; a missing value must be null, not NaN |

### `dataset`

| Check | Rule | Severity | Kind | Meaning | Operator action |
|---|---|---|---|---|---|
| `dataset.validate` | `dataset.career_counter_gap` | warning | anomaly | validate_dataset rule career_counter_gap (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.career_counter_vs_rows` | info | anomaly | validate_dataset rule career_counter_vs_rows (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.complete_without_scores` | blocking | anomaly | validate_dataset rule complete_without_scores (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.current_season_row_quarantined` | error | anomaly | validate_dataset rule current_season_row_quarantined (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.disposals_arithmetic` | warning | anomaly | validate_dataset rule disposals_arithmetic (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.enum_violation` | blocking | anomaly | validate_dataset rule enum_violation (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.exception_rejected_current_season` | blocking | anomaly | validate_dataset rule exception_rejected_current_season (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.fact_on_noncanonical_identity` | blocking | anomaly | validate_dataset rule fact_on_noncanonical_identity (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.fk_orphan` | blocking | anomaly | validate_dataset rule fk_orphan (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.fk_parent_missing` | blocking | anomaly | validate_dataset rule fk_parent_missing (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.import_issue` | error | anomaly | an import-time quality issue escalated by validate_dataset | repair the current-season input row named in the finding |
| `dataset.validate` | `dataset.key_duplicate` | blocking | anomaly | validate_dataset rule key_duplicate (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.match_player_goals_mismatch` | warning | anomaly | validate_dataset rule match_player_goals_mismatch (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.match_players_missing` | warning | anomaly | validate_dataset rule match_players_missing (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.non_null_violation` | blocking | anomaly | validate_dataset rule non_null_violation (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.player_game_club_not_in_match` | error | anomaly | validate_dataset rule player_game_club_not_in_match (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.quarantine_resolution_unverified` | blocking | anomaly | validate_dataset rule quarantine_resolution_unverified (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.quarter_scores_decrease` | warning | anomaly | validate_dataset rule quarter_scores_decrease (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.schema_mismatch` | blocking | anomaly | validate_dataset rule schema_mismatch (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.score_arithmetic` | blocking | anomaly | validate_dataset rule score_arithmetic (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.score_negative` | error | anomaly | validate_dataset rule score_negative (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.stage_unrecognized` | error | anomaly | validate_dataset rule stage_unrecognized (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.stat_before_recorded_from` | info | anomaly | validate_dataset rule stat_before_recorded_from (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.stat_negative` | error | anomaly | validate_dataset rule stat_negative (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.table_missing` | blocking | anomaly | validate_dataset rule table_missing (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.table_unknown` | blocking | anomaly | validate_dataset rule table_unknown (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |
| `dataset.validate` | `dataset.time_on_ground_over_100` | warning | anomaly | validate_dataset rule time_on_ground_over_100 (the promotion gate) | read the finding; repair the input or add a documented historical exception to coverage.yaml |

### `relations`

| Check | Rule | Severity | Kind | Meaning | Operator action |
|---|---|---|---|---|---|
| `relations.membership` | `relations.lineup_not_participant` | error (blocking in current season) | anomaly | a lineup row names a club that did not play the match | re-link the lineup row |
| `relations.membership` | `relations.player_in_both_sides` | blocking | contradiction | one player has fact rows for both clubs of a match | fix the identity link at import |
| `relations.membership` | `relations.result_mismatch` | error (blocking in current season) | anomaly | a player-game's W/L/D disagrees with the match score | re-link the row or correct the source result |
| `relations.membership` | `relations.season_mismatch` | blocking | contradiction | a player-game's season differs from its match's season | re-import; the fact is in the wrong partition |
| `relations.membership` | `relations.stage_mismatch` | blocking | contradiction | a player-game's stage differs from its match's stage | re-link the player row to the right match |
| `relations.membership` | `relations.verified_date_mismatch` | blocking | contradiction | a fixture_verified player-game date differs from its match date | re-link the row or downgrade its date quality; a verified date cannot disagree |
| `relations.identity` | `relations.alias_ambiguous` | warning | anomaly | one alias string names more than one player | disambiguate the alias with evidence |
| `relations.identity` | `relations.alias_target` | blocking | contradiction | an alias or duplicate identity does not resolve to one canonical identity (missing, chained or cyclic) | repair the identity registry so every alias points at a canonical player |
| `relations.identity` | `relations.canonical_self_link` | blocking | contradiction | a canonical identity points at a different canonical_player_id | clear canonical_player_id or mark it as an alias |

### `aggregates`

| Check | Rule | Severity | Kind | Meaning | Operator action |
|---|---|---|---|---|---|
| `aggregates.seasons` | `aggregates.season_row` | blocking | contradiction | a season with matches has no seasons row, or a seasons row has no matches | rebuild season aggregates from the match rows |
| `aggregates.seasons` | `aggregates.season_value` | blocking | contradiction | a seasons row disagrees with values recomputed from its accepted match rows | rebuild season aggregates; a stale aggregate misstates the season |

### `football`

| Check | Rule | Severity | Kind | Meaning | Operator action |
|---|---|---|---|---|---|
| `football.arithmetic` | `football.brownlow_range` | error (blocking in current season) | anomaly | Brownlow votes outside 0-3 in one game | re-import the row |
| `football.arithmetic` | `football.player_behinds_exceed_team` | error (blocking in current season) | anomaly | players' behinds sum to more than the team's behinds (rushed behinds only add to the team) | re-check the match's player rows against the source |
| `football.values` | `football.unusual_value` | warning | anomaly | a value above the policy's plausible maximum (valid but unusual) | confirm against the source; raise plausible_max only with evidence |
| `football.coverage` | `football.blank_as_null` | warning (blocking in current season) | anomaly | a statistic is null for a player who took the field although the match reports that statistic (AFL Tables prints 0 as a blank) | import with domain.blanks or run scvia apply-corrections; observed-denominator means are inflated |
| `football.coverage` | `football.brownlow_not_applicable` | error (blocking in current season) | anomaly | a finals row has a Brownlow value; votes are awarded only in home-and-away matches | re-import the row with Brownlow votes null for finals |
| `football.coverage` | `football.unevidenced_zero` | warning | anomaly | a statistic is 0 for every player with a value in a match, so nothing shows the match reported it | confirm against the source; an unreported column must stay null, not become 0 |
| `football.coverage` | `football.zero_before_recorded` | error | anomaly | zeros stored before the stat's recorded_from season: a missing value was probably zero-filled | re-import with null for unrecorded eras, or document the isolated fragment |

### `freshness`

| Check | Rule | Severity | Kind | Meaning | Operator action |
|---|---|---|---|---|---|
| `freshness.observations` | `freshness.checked_at_behind_observation` | blocking | contradiction | a later PASS observation exists but fixture_checked_at did not advance (stale metadata) | rebuild season aggregates for the checked season |
| `freshness.observations` | `freshness.checked_from_failed_observation` | blocking | contradiction | the latest season-fixture observation at the check time did not PASS (partial fetch marked fresh) | revert fixture_checked_at to the last successful check and re-run the refresh |
| `freshness.observations` | `freshness.checked_without_observation` | blocking | contradiction | fixture_checked_at is set but no PASS season-fixture observation exists at or before it | re-run the refresh; a check time needs a successful observation behind it |
| `freshness.observations` | `freshness.revision_mismatch` | blocking | contradiction | the pinned season revision is not the content of the latest PASS observation | re-pin source_revisions from the accepted observation |
| `freshness.observations` | `freshness.schedule_complete_unsupported` | blocking | contradiction | schedule_complete is true without a source-declared season status | leave schedule_complete unknown until the source declares the season complete |
| `freshness.observations` | `freshness.verified_claim` | blocking | contradiction | the dataset is labelled verified but contains legacy-import rows | label the snapshot legacy_unverified or partial; only source verification upgrades it |
| `freshness.as_of` | `freshness.after_as_of` | blocking | contradiction | the inputs contain timestamps after the audit's --as-of (knowledge from the future) | audit with an --as-of on or after the snapshot, or audit an earlier snapshot |
| `freshness.as_of` | `freshness.fixture_stale` | warning | missing_evidence | the current season's fixture was last checked longer ago than the policy allows at --as-of | run a bounded refresh; stale is not wrong, but it is not current |
| `freshness.fixture_inventory` | `freshness.evidence_missing` | error | missing_evidence | the pinned season page is not in the evidence store | copy raw/objects from the refresh host or pass --evidence DIR |
| `freshness.fixture_inventory` | `freshness.fixture_value_mismatch` | blocking | contradiction | a match value differs from the pinned season page | repair the match row from the pinned source |
| `freshness.fixture_inventory` | `freshness.missing_result` | error (blocking in current season) | contradiction | the pinned source lists a completed match the snapshot does not hold | run a bounded repair for the season |
| `freshness.fixture_inventory` | `freshness.page_unreadable` | error | missing_evidence | the pinned season page could not be read by the independent reader | inspect the payload; parser drift must be fixed before the comparison can run |
| `freshness.fixture_inventory` | `freshness.scheduled_fixture_absent` | warning (blocking in current season) | contradiction | the pinned source lists a scheduled fixture the snapshot does not hold | refresh the fixture; forecasts cannot target a fixture the snapshot lacks |
| `freshness.fixture_inventory` | `freshness.team_unresolved` | error | missing_evidence | a source team name does not resolve through the snapshot's clubs/club_aliases | add the alias to the club registry with evidence |
| `freshness.fixture_inventory` | `freshness.unexpected_match` | error (blocking in current season) | contradiction | the snapshot holds a match the pinned source does not list | check the match identity against the source |

### `source`

| Check | Rule | Severity | Kind | Meaning | Operator action |
|---|---|---|---|---|---|
| `source.match_pages` | `source.evidence_missing` | error | missing_evidence | a required source payload (pinned or behind a source_fetch row) is not in the evidence store | copy raw/objects from the refresh host or pass --evidence DIR |
| `source.match_pages` | `source.fetch_unrecorded` | blocking | contradiction | a source_fetch row names a payload no observation recorded | re-run the refresh; provenance must be recorded |
| `source.match_pages` | `source.jersey_mismatch` | error (blocking in current season) | contradiction | a player's jumper number differs from the page | re-check the identity link |
| `source.match_pages` | `source.match_unlinked` | error (blocking in current season) | missing_evidence | a captured match page matches no canonical match | check match identity and team aliases |
| `source.match_pages` | `source.match_value_mismatch` | blocking | contradiction | a match value (date, venue, attendance, stage, score) differs from the captured page | repair the match row from the captured source |
| `source.match_pages` | `source.page_totals_inconsistent` | warning | anomaly | the page's own Totals/Rushed rows disagree with its player rows | note the source inconsistency |
| `source.match_pages` | `source.page_unreadable` | error | missing_evidence | a match page could not be read by the independent reader | inspect the payload for parser drift |
| `source.match_pages` | `source.player_ambiguous` | error (blocking in current season) | missing_evidence | a captured player matches several canonical rows | disambiguate the identity with the player's source URL |
| `source.match_pages` | `source.player_extra` | blocking | contradiction | a canonical player row is not on the captured page | remove or re-link the row |
| `source.match_pages` | `source.player_missing` | blocking | contradiction | a player row on the captured page has no canonical row | re-import the match's player rows |
| `source.match_pages` | `source.revision_unobserved` | blocking | contradiction | a pinned source revision has no observation with that content | re-pin from a recorded observation |
| `source.match_pages` | `source.stat_cell_mismatch` | blocking | contradiction | a statistic cell differs from the captured page (blank on the page = null) | repair the row from the captured source; a swapped or edited value surfaces here |
| `source.match_pages` | `source.value_without_source_column` | warning | anomaly | a canonical statistic has a value although the page has no such column | confirm where the value came from |
| `source.player_pages` | `source.player_page_membership` | blocking | contradiction | a game on the captured player page has no canonical row, or the player has a row the page does not list | repair the player's rows from the captured page |
| `source.player_pages` | `source.player_page_missing` | error | missing_evidence | a captured player page that source-fetched rows depend on is not in the evidence store | restore the payload from the archive, or pass --evidence DIR |
| `source.player_pages` | `source.player_page_unlinked` | error (blocking in current season) | missing_evidence | a captured player page names no canonical player, or one of its games matches several rows | record the player's source URL or resolve the duplicate rows |
| `source.player_pages` | `source.player_page_unreadable` | error | missing_evidence | a captured player page cannot be read by the checker's own reader | inspect the archived page; the markup may have changed |
| `source.player_pages` | `source.player_page_value` | blocking | contradiction | a canonical row differs from the captured player page (counter, result, jumper, non-blank cells; a blank cell only establishes that the count is not positive) | repair the row from the captured page |

### `release`

| Check | Rule | Severity | Kind | Meaning | Operator action |
|---|---|---|---|---|---|
| `release.artifact` | `release.embedded_mismatch` | blocking | contradiction | site/data/<release>/ is not the validated public tree | rebuild the site against this release's public tree and re-seal |
| `release.artifact` | `release.manifest_ref` | blocking | contradiction | a release.json resource reference disagrees with checksums.json | rebuild the release |
| `release.artifact` | `release.metadata_invalid` | blocking | contradiction | checksums.json, seal.json or validation.json is missing or invalid | rebuild or restore the release; never edit its metadata |
| `release.artifact` | `release.public_extra` | blocking | contradiction | a public file is not listed in checksums.json | rebuild the release; unlisted files are never published |
| `release.artifact` | `release.public_hash` | blocking | contradiction | a public file's bytes differ from checksums.json (truncated or substituted) | restore the release from its archive |
| `release.artifact` | `release.public_missing` | blocking | contradiction | a file listed in checksums.json is missing | restore the release from its archive |
| `release.artifact` | `release.release_id` | blocking | contradiction | release.json, checksums.json or seal.json names another release | restore the release directory from its archive |
| `release.artifact` | `release.seal_mismatch` | blocking | contradiction | site/ differs from seal.json, or the seal's self-hash is wrong | rebuild and re-seal the site; never copy over a sealed site |
| `release.artifact` | `release.snapshot_mismatch` | blocking | contradiction | the release was built from a different snapshot than the one audited | audit the release against its own snapshot, or rebuild the release from the audited snapshot |
| `release.artifact` | `release.tree_entry_refused` | blocking | contradiction | a symlink or non-regular file is inside public/ or site/ | remove it and rebuild; releases hold regular files only |
| `release.artifact` | `release.validation_binding` | blocking | contradiction | validation.json names other checksums, another seal or another release than these bytes | re-validate this exact release; a validation record for other bytes authorizes nothing |
| `release.artifact` | `release.validation_outcome` | blocking | contradiction | validation.json does not record PASS | run scvia validate-release and fix what it reports |
| `release.validate` | `release.json_invalid` | blocking | contradiction | a public JSON file does not parse | rebuild the release |
| `release.validate` | `release.private_content` | blocking | contradiction | a published file contains a local path or secret marker | remove the content at its source and rebuild |
| `release.validate` | `release.reference_missing` | blocking | contradiction | an index references a resource the release lacks | rebuild the release |
| `release.validate` | `release.schema_invalid` | blocking | contradiction | a public JSON file fails its view-model schema | rebuild the release |
| `release.validate` | `release.unsafe_path` | blocking | contradiction | a public path is outside the allowlist | rename or drop the file and rebuild |
| `release.validate` | `release.validate_other` | blocking | contradiction | validate_release reported another failure | run scvia validate-release |
| `release.public` | `release.aggregate_mismatch` | blocking | contradiction | a player detail aggregate differs from totals recomputed from the facts (observed denominators) | rebuild the release; a stale or mis-joined aggregate surfaces here |
| `release.public` | `release.box_membership` | blocking | contradiction | a match box score lists the wrong players or names | rebuild the release |
| `release.public` | `release.box_order` | error | contradiction | a box score is not in contract order | rebuild the release |
| `release.public` | `release.cell_mismatch` | blocking | contradiction | a published statistic differs from the canonical cell (null stays null) | rebuild the release; a value on the wrong player or stat surfaces here |
| `release.public` | `release.changed_during_audit` | blocking | contradiction | a public file changed between inventory and comparison | re-run the audit on a stable copy |
| `release.public` | `release.compact_row_width` | blocking | contradiction | a compact stats row is not as wide as its stat_columns | rebuild the release; a short row shifts values onto the wrong statistic |
| `release.public` | `release.index_membership` | blocking | contradiction | an index lists a match/player the facts lack, or omits one they hold | rebuild the release from the audited snapshot |
| `release.public` | `release.index_order` | error | contradiction | an index is not in its contract order | rebuild the release |
| `release.public` | `release.json_duplicate_key` | blocking | contradiction | a public JSON object repeats a key (consumers may read either value) | rebuild the release; the serializer must never emit duplicate keys |
| `release.public` | `release.log_membership` | blocking | contradiction | a player season log lists the wrong games | rebuild the release |
| `release.public` | `release.log_order` | error | contradiction | a player season log is not in chronological order | rebuild the release |
| `release.public` | `release.log_value_mismatch` | blocking | contradiction | a player log field differs from the canonical row | rebuild the release |
| `release.public` | `release.match_summary_mismatch` | blocking | contradiction | a published match fact differs from the canonical match row | rebuild the release; shared match facts must equal the match row |
| `release.public` | `release.player_index_value` | blocking | contradiction | a player index entry (seasons played, games, clubs, search) differs from the facts | rebuild the release; season membership must be exact, not a first-last span |
| `release.public` | `release.resource_missing` | blocking | contradiction | a resource the canonical rows require is absent from the release | rebuild the release from the audited snapshot |
| `release.public` | `release.shared_facts` | blocking | contradiction | a player log's shared match facts point at the wrong index, duplicate it, or name a match it lacks | rebuild the release |
| `release.public` | `release.source_label` | blocking | contradiction | a match's source label disagrees with its recorded provenance | rebuild; provenance labels are claims to readers |
| `release.public` | `release.stat_columns` | blocking | contradiction | stat_columns are not the canonical stats observed in the file, in canonical order (an all-null column must be omitted) | rebuild the release |
| `release.public` | `release.unexpected_resource` | blocking | contradiction | a match/player resource has no canonical row behind it | rebuild the release; stale or foreign resources must not ship |
| `release.derived` | `release.derived_resource` | blocking | contradiction | a team/history/list/summary/download resource the facts require is missing, unreadable, or not backed by any fact | rebuild the release from the audited snapshot |
| `release.derived` | `release.download_metadata` | blocking | contradiction | downloads.json disagrees with the download files (bytes, sha256, rows, kind, membership) | rebuild the release |
| `release.derived` | `release.download_value` | blocking | contradiction | a download (CSV rows, chart values, fan-pack member) differs from values recomputed from the facts | rebuild the release |
| `release.derived` | `release.history_value` | blocking | contradiction | a history table differs from leaders or legacy_v1 rankings recomputed from the facts (membership, rank, value, denominators) | rebuild the release; never hand-edit a ranking |
| `release.derived` | `release.list_value` | blocking | contradiction | a list season differs from the snapshot's draft, contract and school rows | rebuild the release |
| `release.derived` | `release.summary_value` | blocking | contradiction | overview.json or quality.json states a value the facts or the release's own resources contradict | rebuild the release; a stale summary is shown on the home page |
| `release.derived` | `release.team_value` | blocking | contradiction | a team page or the team index differs from values recomputed from the matches and player games (ladder, position, form, fixtures, team-game means, leaders, five-year view, heuristic facts) | rebuild the release; a stale or mis-joined team aggregate surfaces here |
| `release.forecast` | `release.forecast_resource` | blocking | contradiction | a prediction, accuracy or live resource is missing, unreadable, or has no pinned input behind it | rebuild the release from the pinned artifacts |
| `release.forecast` | `release.forecast_value` | blocking | contradiction | a published prediction, accuracy or live value differs from its pinned artifact or capture | rebuild the release; a forecast must be the artifact's own numbers |
| `release.content` | `release.content_provenance` | blocking | contradiction | an article's provenance (source path and hash, or checked snapshot) does not match its input | rebuild the release from the current curated sources |
| `release.content` | `release.content_resource` | blocking | contradiction | an article or asset is missing from its index, listed without a resource, or an unreferenced asset | rebuild the release |
| `release.coverage` | `release.unclassified_resource` | blocking | contradiction | a public file has a type no comparison knows (it could publish anything unchecked) | remove the file or add its type to the resource inventory with a comparator |

### `models`

| Check | Rule | Severity | Kind | Meaning | Operator action |
|---|---|---|---|---|---|
| `models.bundles` | `models.bundle_hash` | blocking | contradiction | a bundle manifest's self-hash or directory name is wrong | treat the bundle as tampered; retrain |
| `models.bundles` | `models.bundle_invalid` | blocking | contradiction | a bundle manifest does not parse under its contract | retrain; never edit a bundle manifest |
| `models.bundles` | `models.code_fingerprint` | warning | anomaly | the bundle was trained with different feature/model code than this checkout | retrain before the next forecast (forecast refuses a stale bundle) |
| `models.bundles` | `models.feature_order` | blocking | contradiction | the bundle's feature order disagrees with its persisted feature list | retrain; serving must use the training feature order |
| `models.bundles` | `models.feature_spec` | blocking | contradiction | the persisted feature specification cannot be rebuilt or does not match its fingerprint | retrain with the current feature code |
| `models.bundles` | `models.knowledge_after_as_of` | blocking | contradiction | the bundle learned from outcomes after the audit's --as-of | audit later or use an earlier bundle |
| `models.bundles` | `models.knowledge_cutoff_missing` | error | contradiction | the bundle records no knowledge cutoff | retrain so the manifest records one |
| `models.bundles` | `models.payload_hash` | blocking | contradiction | a bundle's payload bytes do not match its manifest | treat the payload as untrusted; never load it; retrain |
| `models.bundles` | `models.payload_missing` | blocking | contradiction | a bundle's predictor payload is missing | restore or retrain the bundle |
| `models.predictions` | `models.bundle_missing` | blocking | contradiction | an available forecast names a bundle that is not present | restore the bundle or rebuild the forecast |
| `models.predictions` | `models.forecast_after_as_of` | blocking | contradiction | a forecast cutoff or generation time is after --as-of | audit later or audit an earlier forecast |
| `models.predictions` | `models.forecast_unavailable_with_fixture` | blocking | contradiction | the forecast says no future fixture, but the snapshot schedules one after the cutoff | rebuild the forecast; unavailability must reflect the fixture |
| `models.predictions` | `models.ineligible_bundle` | blocking | contradiction | a forecast cutoff precedes the bundle's knowledge cutoff (the model saw later outcomes) | forecast with a bundle whose knowledge cutoff is on or before the cutoff |
| `models.predictions` | `models.interval_order` | blocking | contradiction | a prediction interval is not low <= prediction <= high | rebuild the forecast |
| `models.predictions` | `models.prediction_file_hash` | blocking | contradiction | a prediction file's bytes differ from its manifest | rebuild the forecast |
| `models.predictions` | `models.prediction_invalid` | blocking | contradiction | a prediction manifest does not parse | rebuild the forecast |
| `models.predictions` | `models.prediction_label` | blocking | contradiction | a prediction's origin/status/reason is outside the contract or inconsistent | rebuild the forecast |
| `models.predictions` | `models.prediction_snapshot` | warning | anomaly | a prediction was made from another snapshot | rebuild the forecast from the audited snapshot before publishing it |
| `models.predictions` | `models.prediction_value` | blocking | contradiction | a predicted value is missing, negative or not finite | rebuild the forecast |
| `models.predictions` | `models.release_forecast` | blocking | contradiction | the release's forecast status/model disagrees with the prediction artifacts | rebuild the release |
| `models.predictions` | `models.row_count` | blocking | contradiction | the manifest's predicted count differs from its rows | rebuild the forecast |
| `models.predictions` | `models.row_identity` | blocking | contradiction | a prediction row names another run, snapshot, model, origin or cutoff than its manifest | rebuild the forecast |
| `models.predictions` | `models.target_unknown` | error | contradiction | a prediction targets a match the snapshot does not hold | rebuild the forecast against the audited snapshot |

### Policy bookkeeping

| Check | Rule | Severity | Kind | Meaning | Operator action |
|---|---|---|---|---|---|
| `policy.exceptions` | `policy.exception_rejected_current_season` | blocking | policy | accepted exception names a current-season defect | remove the exception and repair the current-season defect |
| `policy.exceptions` | `policy.exception_stale` | warning | policy | accepted exception matched no finding | delete the stale exception from the policy |
| `inputs.stable` | (status only) | — | — | an input's bytes changed between capture and the end of the audit | re-run on a stable copy |
