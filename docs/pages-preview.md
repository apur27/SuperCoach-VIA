# Provisional GitHub Pages preview

The [provisional site](https://apur27.github.io/SuperCoach-VIA/) was published on
3 October 2026. The address is case-sensitive; `/supercoach-via/` returns 404.
Publication uses the existing, manually dispatched `scvia-pages.yml` workflow.
The Python pipeline runs locally; publishing a site does not activate a refresh schedule.

## Publication record

- Source commit: `a5b2b804acdbf06952851d36c4dc9b2ec6bfaae0`.
- Release: `20261003T105359Z-67fb4988599a`.
- [Successful Pages workflow](https://github.com/apur27/SuperCoach-VIA/actions/runs/37118887973).
- [Public prerelease archive](https://github.com/apur27/SuperCoach-VIA/releases/tag/preview-2026-10-03).
- Site seal: `56fb11cb70e0fa4b30647743467843e964babe53b288cdf970f3410276965de1`.
- Archive SHA-256: `c3bf3235a4fed6060bdb80c8825555cf9cb624634f53b85101a195134d9d6ae0`.
- Site size: 264,045,507 bytes, within the 300 MiB budget.

Local validation passed: 179 web unit tests (2 existing skips), 322 browser tests
(4 existing skips), type checks, lint, the payload budget, seal validation and the full
input-consistency audit (`semantic_complete=true`). The real-site review covered 184
page states with no unexpected request errors, horizontal overflow or accessibility
violations. Search, watchlist persistence, Grand Final/player navigation and all 14
downloads were checked. A live-browser recheck covered 20 page states, accessibility,
downloads and navigation; a separate search check waited for remote results before
opening the player. Six deployed pages, manifests and assets matched the sealed bytes.

Repository-wide CI is **not green**. The [source commit's CI run](https://github.com/apur27/SuperCoach-VIA/actions/runs/37118404379)
fails because a Python browser-contract test assumes `/usr/bin/node` exists on the
runner, and the provenance strip measures 194px at 320px width against its 170px limit
in the runner's browser. The [preceding commit also failed CI](https://github.com/apur27/SuperCoach-VIA/actions/runs/37107281828).
The existing Pages workflow separately verified and published the tested, sealed artifact.
Those repository CI issues remain open; no test or publication gate was weakened.

Local logs, reports and screenshots are in `var/pages-preview/20261003/` (outside Git).
Hash checks confirmed that 90 paused Claude-worktree files, 604 retained candidate files
and the three pre-existing main-checkout memory files were unchanged.

## Data status

This preview uses snapshot
`sha256:3de6597513b5c6eaca1b102d304a2cf128d8c08f229ee1de384c1d40fe746db0`,
created on 29 September 2026. Its captured AFL Tables source audit is **FAIL**.
Missing appearances, zero/null differences, incomplete historical Brownlow totals and
unresolved evidence can affect totals and rankings. The [README](../README.md#data-status)
records the distinction between source agreement and release consistency.

The browser pins that audit notice to this exact snapshot, displays it on every page,
links to `/data-status/`, and requests `noindex` from search engines. DEMO fixtures and
other snapshots do not inherit this snapshot's audit result. A future corrected snapshot
needs its own review before publication.

The recorded source-audit report SHA-256 is
`58eff517a29d30b932087f919a31d36567e8be330301ecff53feb06494d8db28`.
The preview has no current forecast because its inputs contain no valid future fixture.

## Build and publication

1. Start from a committed checkout. Copy the selected data root into a new run directory;
   all build logs and new outputs must go there. Keep retained snapshots and Claude's
   unfinished reconciliation worktree unchanged.
2. Follow [operations](operations.md) to build a fresh release. Set
   `SCVIA_PUBLIC_BASE=/SuperCoach-VIA/` for both Python and Astro. Pass the exact snapshot ID.
   Do not overwrite an earlier sealed release.
3. Build Astro with `SCVIA_RELEASE_DIR` pointing to the release's absolute `public/` path
   and `SCVIA_PUBLIC_SITE=https://apur27.github.io`. Build inside the checkout filesystem,
   then copy `web/dist` to the fresh release's `site/` directory.
4. Run the web checks, payload budget and real-data browser review. Verify the provisional
   notice, mobile navigation, search, match/player links, downloads and accessibility.
5. Run `scvia seal-site` and release validation. Run the existing full integrity checker
   for consistency with the selected inputs; it does not replace AFL Tables reconciliation.
6. Pack the sealed release:

   ```bash
   .venv/bin/python -m supercoach_via.publish.deploy pack /absolute/release /absolute/preview-site.tar
   ```

   Record the printed `seal_sha256` and `archive_sha256`. Upload only this public-site
   archive as a GitHub prerelease asset, tied to the source commit. The archive contains
   `site/`, `seal.json` and `validation.json`; it excludes the local data root and evidence.
7. Set GitHub Pages to **GitHub Actions**, then dispatch `scvia-pages.yml` on main with
   the asset's HTTPS URL and both recorded digests. The workflow verifies the downloaded
   archive and deploys those exact bytes without rebuilding Astro.
8. Check the live page, asset base paths, release ID, warning, downloads and key user flows.
   Preserve the deployment URL, workflow run and artifact digests in the publication record.

## Rollback and scope

Keep each published archive and its two digests. To roll back a later preview, dispatch
the same workflow with the previous archive URL and its original digests. Do not reseal
or rebuild it during rollback. For the first deployment there is no earlier Pages release;
an incident requiring withdrawal can be handled by unpublishing the site in Pages settings.

This work changes browser presentation and publishes a static artifact. It changes no
numeric pipeline, source data, reconciliation implementation, harness, hook or schedule.
It does not count as either of the production shadow cycles in [the switch plan](rewrite/SWITCH_PLAN.md).
