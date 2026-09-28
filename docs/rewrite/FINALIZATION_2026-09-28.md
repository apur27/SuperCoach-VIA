# Finalized local app — 28 September 2026

The reviewed rewrite is merged into local `main` at **`2bcdfbeb4142c98bf120e5056c06f3f142515c3f`**. It continues Claude Code's implementation at `2aad17873` with Cursor Grok 4.7's fixes. The independent review returned PASS, and the required commit hook passed its full fast tier and council checks. There was no remote push, deployment, schedule change, or default harness activation.

The prior checkout is preserved in Git stash `484438286b5beaf2ad1a715b44cb1ee38627c539` and in `var/recovery/cursor-grok47-20260928/main-before-merge/`. Do not apply that stash over the merged rewrite: it contains superseded application files as well as the original local changes.

## Open the finished app

From this repository:

```bash
node web/scripts/serve.mjs --dir var/final-site --port 4321
```

Open <http://127.0.0.1:4321/>. The preview uses the retained, sealed real-data site at `var/finalized/releases/20260928T111014Z-ca96603163ad/site/`, built for `/`. It needs Node, without an npm build or Python server. `var/final-site` points to that directory.

The local Python environment has been synced with:

```bash
uv sync --locked --group dev --group legacy --extra ml
.venv/bin/scvia doctor --data-root var/finalized/data --output-root var/finalized --json
```

The retained data, models, release and deployment archive live under `var/finalized/`; they are local artifacts and are excluded from Git. A fresh clone needs those artifacts copied separately or regenerated using [operations](../operations.md). The committed [verification reports](finalization-20260928/README.md) remain available in a fresh clone.

## Grand final: updated in the new app

The accepted snapshot includes the grand final on **26 September 2026** and all **46 player rows** **[historical record]**. The source, snapshot, public match detail, player logs and sealed site agree. Verification compared 1,058 source statistic cells, including blanks, and checked the match-to-player links in a real browser at desktop and mobile sizes.

Source: [AFL Tables match page](https://afltables.com/afl/stats/games/2026/081920260926.html). Evidence: [data verification](finalization-20260928/grand-final-verification.json), [retained-copy verification](finalization-20260928/retained-grand-final-verification.json), and [browser verification](finalization-20260928/grand-final-browser.json).

The new pipeline's season summary and player logs are updated. The legacy CSVs remain pre-final migration inputs. The live website has not been updated by this task. `schedule_complete` and `source_status` remain unknown, and forecasts correctly say `no_valid_future_fixture`.

## Release and checks

| Item | Result |
|---|---|
| Release | `20260928T111014Z-ca96603163ad` |
| Snapshot | `sha256:aa836549e10e96a9039cc4642bbe563b94247e0bfaf95e745a0971cc0bb2b98f` |
| Seal | `bd9a655b06795c2d83ca77a48adb9df85fb2774463fa390f814dac2ff72474fa` |
| Site size | 276,821,039 bytes; three-season projection plus 5 MiB is 280.46 MiB |
| Hermetic suite | 695 passed in 29.33s on four distinct physical cores; the earlier two-core/SMT placement took 34.15s |
| Integration | 25 passed before the final small fixes; 9 affected tests passed afterward |
| Browser suite | 320 passed, 4 skipped; web unit tests 163 passed, 2 skipped |
| Final Ruff / mypy | PASS |
| Final scratch weekly rehearsal | PASS; legacy parity, failed upload, publish and rollback; zero source-inventory differences |
| Independent Grok 4.7 review | [PASS for local merge](finalization-20260928/independent-review.txt) |

The source inventory is `318c37e8b2a80d6f518f89cde1427923dc4d7a1a389224d58f81400442c8ea5a`. The retained release and data were copied outside the agent worktree, every copied file hash matched, and the retained site's seal verified. See `var/finalized/metadata.json` for local paths and archive digest.

The performance result is specific to the measured machine and CPU placement. The 30-second test budget and the 300 MiB site budget were not raised. Tests were not removed to meet either target.

## Remaining production prerequisites

The local app is ready for use and further development. Production activation still requires two genuine shadow cycles and the deployment/retention decisions in [SWITCH_PLAN.md](SWITCH_PLAN.md). The offline rehearsal does not count as either shadow cycle. `SCVIA_NUMERIC_ENTRY=1` remains opt-in; the legacy default harness and schedules remain in place.

The next-agent payload is [OPUS_55_PAYLOAD_2026-09-28.md](OPUS_55_PAYLOAD_2026-09-28.md). It was prepared after the code merge. Opus was not launched.
