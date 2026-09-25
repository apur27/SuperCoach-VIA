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
#    The B1 repair evidence is replayed offline, re-hashed and re-parsed.
uv run scvia import-legacy --source . --repair docs/rewrite/evidence/b1:2026 --json
uv run scvia status --json

# 2. Train (or reuse the cached bundle), forecast the next REAL fixtures, and replay-evaluate a season.
#    With no future fixture in the source, forecast_status=unavailable is the correct outcome.
uv run scvia forecast --train-cutoff 2025-01-01 --calibration-end 2026-01-01 \
  --cutoff 2026-09-25T10:00:00+00:00 --replay-season 2026 --json

# 3. Build and validate a release from the accepted snapshot. Forecast inputs are named
#    explicitly (IDs and paths printed by step 2); nothing is selected by mtime.
uv run scvia build-release --snapshot current --editorial off \
  --bundle <bundle_id> --predictions var/predictions/<prediction_run_id> \
  --evaluation var/evaluations/<evaluation_id> \
  --content-manifest config/public_content.toml --content-root . --json
uv run scvia validate-release --release <release_id> --json

# 4. Browser build against that release, then local read-only preview.
#    SCVIA_RELEASE_DIR must be ABSOLUTE and point at the release's public/ directory (the build
#    refuses a relative path). SCVIA_PUBLIC_BASE must equal the base the release was built with.
(cd web && SCVIA_RELEASE_DIR="$(realpath ../dist/releases/<release_id>/public)" SCVIA_PUBLIC_BASE=/SuperCoach-VIA/ npm run build)
uv run scvia preview --release <release_id>          # binds 127.0.0.1
```

Build the release with the same `SCVIA_PUBLIC_BASE` as the site: article image URLs are resolved against the base at release-build time.

The dates above are examples. Pick a training cutoff before the calibration block, and the
calibration end before the season you replay.

## Refresh from sources

```bash
uv run scvia refresh --season 2026 --plan --json                    # offline: sources, estimated requests, no writes
uv run scvia refresh --season 2026 --data-only --allow-network      # explicit network opt-in
uv run scvia refresh --season 2026 --repair-season 2026 --allow-network
```

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
