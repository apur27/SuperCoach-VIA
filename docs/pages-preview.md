# Provisional GitHub Pages preview

The static site is prepared for `https://apur27.github.io/SuperCoach-VIA/`.
Publication uses the existing, manually dispatched `scvia-pages.yml` workflow.
The Python pipeline runs locally; publishing a site does not activate a refresh schedule.

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
