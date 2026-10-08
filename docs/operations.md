# Operations: running SuperCoach VIA

*Operator runbook for the rewrite package (`scvia`). Spec: [rewrite/PLAN.md](rewrite/PLAN.md) §5, §14 and §16. This document contains no player statistics.*

The legacy weekly harness (`scripts/weekly_refresh.sh`) stays operational until the Phase 9
switch-over, and its rules in `CLAUDE.md` §6 still apply to it. Nothing below invokes
the harness, the council agents, Git or any network source unless the step says so explicitly.

## Install

```bash
uv sync --locked --group dev --group legacy --extra ml   # legacy: parity tests import the old package
(cd web && npm ci)
uv run scvia doctor                        # versions, writable roots, lock, source policy; no network
```

Settings come from `--config FILE` or `SCVIA_CONFIG` (see `config/app.example.toml`), followed by
the narrow overrides `SCVIA_DATA_ROOT`, `SCVIA_OUTPUT_ROOT`, `SCVIA_SOURCE_ROOT`,
`SCVIA_PUBLIC_BASE`, `SCVIA_SITE_URL` and `SCVIA_TIMEZONE`. Unknown keys fail validation.

## Offline demo (no network, GPU or credentials)

```bash
uv run scvia demo --output dist/demo                 # ~10 s; every output is labelled DEMO
# layout: dist/demo/source, dist/demo/var, dist/demo/releases/<id>  (use --output-root dist/demo)
(cd web && npm run dev)                              # uses the web DEMO fixture unless SCVIA_RELEASE_DIR is set
```

## Real data, end to end

```bash
# 1. Import + validate + promote (about 60 s and 2 GiB peak on the reference box).
#    The legacy CSVs contain the B1 repair rows since the AFL Tables reconciliation corrections
#    (commit 67217df40), so the current source needs no --repair. For a capture taken BEFORE those
#    corrections, add `--repair docs/rewrite/evidence/b1:2026` (replayed offline, re-hashed, re-parsed).
uv run scvia import-legacy --source . --json
uv run scvia status --json

# 2. Train (or reuse the cached bundle) and forecast the next REAL fixtures.
#    With no future fixture in the source, forecast_status=unavailable is the correct outcome.
uv run scvia forecast --train-cutoff 2025-01-01 --calibration-end 2026-01-01 \
  --cutoff 2026-09-28T10:00:00+00:00 --json

# 3. Build and validate a release from the accepted snapshot. Forecast inputs are named
#    explicitly (IDs and paths printed by step 2); nothing is selected by mtime.
SCVIA_PUBLIC_BASE=/SuperCoach-VIA/ uv run scvia build-release --snapshot current --editorial off \
  --bundle <bundle_id> --predictions var/predictions/<prediction_run_id> \
  --content-manifest config/public_content.toml --content-root . --json
uv run scvia validate-release --release <release_id> --json

# 4. Browser build against that release, attach the site, then seal and validate it.
#    SCVIA_RELEASE_DIR must be ABSOLUTE and point at the release's public/ directory (the build
#    refuses a relative path). SCVIA_PUBLIC_BASE must equal the base the release was built with.
RELEASE_DIR="$(realpath dist/releases/<release_id>)"
(cd web && SCVIA_RELEASE_DIR="$RELEASE_DIR/public" SCVIA_PUBLIC_BASE=/SuperCoach-VIA/ npm run build)
# Use a fresh release. Do not copy over an existing sealed site.
test ! -e "$RELEASE_DIR/site" && cp -a web/dist "$RELEASE_DIR/site"
uv run scvia seal-site --release <release_id> --json
uv run scvia validate-release --release <release_id> --json
# 5. Preview at http://127.0.0.1:4321/SuperCoach-VIA/ (Ctrl-C to stop).
node web/scripts/serve.mjs --dir "$RELEASE_DIR/site" --base /SuperCoach-VIA/ --port 4321
```

Build the release with the same `SCVIA_PUBLIC_BASE` as the site: article image URLs are resolved against the base at release-build time.

The dates above are examples; use the current UTC time for a new forecast. A replay
also requires every stage cutoff to be on or after the bundle's knowledge cutoff,
including its holdout evaluation period. Do not replay that holdout as if it were an
independent forecast. Attach an eligible evaluation with `--evaluation` only when
you have produced one; the recorded holdout results are described in the model card.

`scvia preview` serves the site at `/`. For a release built with a subpath such as
`/SuperCoach-VIA/`, use the base-aware Node server shown above.

## Integrity audit (optional, read-only)

```bash
uv run scvia check-integrity --data-root "$(realpath var)" --release-dir "$(realpath dist/releases/<release_id>)" \
  --scope full --as-of 2026-09-28T12:00:00Z --report /path/outside/inputs/integrity.json --json
```

Offline and deterministic; exit 0 PASS, 4 blocking violations, 8 incomplete verification, 2 usage, 9 checker
failure. It is not part of any gate. Rules, outputs and operator responses: [data-integrity.md](data-integrity.md).

## Refresh from sources

```bash
uv run scvia refresh --season 2026 --plan --json                    # offline: sources, estimated requests, no writes
uv run scvia refresh --season 2026 --data-only --allow-network      # explicit network opt-in
uv run scvia refresh --season 2026 --repair-season 2026 --allow-network
# owner-bounded: current season page + only new/changed matches, hard request cap, retries off
uv run scvia refresh --season 2026 --data-only --allow-network --new-matches-only --max-requests 2 \
  --proxy "$HTTPS_PROXY" --json
```

`--max-requests N` counts every HTTP request (retries are disabled when it is set). Once the
budget is spent, the remaining items are recorded as failed and the run ends PARTIAL, with
exit 3 and no promotion. It never exceeds N. Each attempt spends budget, so don't poll with it.
`--new-matches-only` skips the prior-season overlap and the recheck of unchanged matches.

A refresh fetches the season fixture first. It then plans only the missing and changed match
details, merges upserts into a new candidate snapshot, validates it, and promotes it only on
PASS. An unreachable or partial source returns exit 3 and never moves `var/current.json`.
The source policy (hosts, path grammars, 2 requests/s, 3 attempts, 10 MiB cap) lives in
`config/source_policies.toml`.

## Exit codes

| Code | Meaning | Typical recovery |
|---|---|---|
| 0 | requested operation succeeded | - |
| 2 | invalid input or config | fix the flag or settings file named in the message |
| 3 | source unavailable / partial refresh | re-run later. The previous accepted snapshot is still current |
| 4 | validation failure | read `var/runs/<run_id>/validation-report.json`, repair, re-run |
| 5 | locked (another writer holds the data-root lock) | wait for the other run to finish. Never delete the lock while it runs |
| 6 | requested model/forecast unavailable | the release still builds with `forecast.status=unavailable` |
| 7 | publication failure | the previous release stays active. Retry `publish` with the same release ID |

Every command writes structured JSONL to `var/runs/<run_id>/`. `--json` prints one final result
object on stdout, which includes the recovery command.

## Publication, rollback and retention

- `scvia publish --release <id> --destination <dir>` is the only public mutation. It refuses
  unvalidated releases and writes a receipt. It is prepared but **not** run by the rewrite.
  The GitHub Pages workflow (`.github/workflows/scvia-pages.yml`) needs an explicit dispatch.
- `scvia rollback --release <previous_id> --destination <dir>` re-activates an earlier
  validated release and writes a new receipt. Snapshots and source evidence are never modified.
- Retention: all accepted dataset manifests and their fragments are kept, along with the active
  release and the two before it (`retain_releases`). No prune command runs automatically.

## Backups

`var/` holds the accepted snapshots, runs, models, predictions and evaluations. Back up
`var/snapshots/`, `var/fragments/`, `var/current.json`, `var/models/`, `var/predictions/`
and `var/evaluations/` together. A restore is the reverse copy followed by
`scvia validate --snapshot current`. The legacy CSV corpus under `data/` is migration input v0.
Its hashes are pinned in `docs/rewrite/evidence/input-v0-manifest.json`.

## Scheduling (example)

Run the refresh as an operator action, not a blind schedule. If you do schedule it, use an IANA zone:

```cron
CRON_TZ=Australia/Melbourne
17 9 * * 1  cd /srv/supercoach-via && uv run scvia refresh --season 2026 --data-only --allow-network --json >> var/cron.log
```

The job does not publish. A person reviews `scvia status` and then runs the build and publish steps.


## Local audit state and owner permissions

The weekly harness writes `.claude/audit/last_refresh_status.json`,
`last_refresh_complete.json` and dated `insights_*.json` verdict output locally.
Some historical instances are tracked, although Phase 4 does not stage them. A fresh
clone therefore contains historical cycle markers; read their timestamp and inspect
actual running harness processes before treating them as evidence of current activity.
Do not commit or discard another checkout's dirty audit files during unrelated work.

A dedicated migration is still required: preserve existing marker/verdict bytes and
hashes outside Git, define how a fresh clone initializes fail-closed cycle state, test
active/complete/absent-marker handling, then untrack only the agreed operational paths
and add matching ignore rules. Do not delete retained evidence or infer that an old
complete marker proves no live cycle exists. This review patch leaves markers intact.

`.claude/settings.local.json` belongs to the operator. Its broad command grants and
retired interpreter rule need an owner review; do not rewrite local/global permissions
as part of code delivery. The durable interpreter contract is `scripts/harness_env.sh`:
`SUPERCOACH_PYTHON` then `<repo>/.venv/bin/python`; the commit hook additionally accepts
`COUNCIL_PYTHON` first. Replace stale grants only in a separate owner-approved settings
change after preserving the current file. No such permission change is made here.

The legacy Pylint, pip/old-Python and Conda workflows have been consolidated into the
locked `scvia-ci` Python and legacy unit jobs. Ruff/mypy are the canonical lint/type
checks; this retirement does not claim rule-for-rule Pylint equivalence. Real-data
integration remains in the weekly harness's Phase 3d. The header's cron example does
not establish an installed schedule; schedule activation remains an owner action.


The changed-season reconciliation gate still reads the configured reference snapshot
(`RECON_DATA_ROOT`, default `var/finalized/data`); its decision is explicitly scoped to
legacy CSVs. `gate.json` and the persisted status record now name that scope, the
snapshot resolved by the captured plan and the audited seasons. Snapshot findings are
reference-only and cannot certify or promote a candidate. Comparing that reference
still has a cost; eliminating stale reference work or following a future promotion
needs a separate scope/binding design. No snapshot selection or decision policy changed.
