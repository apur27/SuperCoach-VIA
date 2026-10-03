# Provisional GitHub Pages preview

The [provisional site](https://apur27.github.io/SuperCoach-VIA/) was published on
3 October 2026. The lowercase `/supercoach-via/` entry redirects to the canonical
`/SuperCoach-VIA/` address; the repository has not been renamed.
Publication uses the existing, manually dispatched `scvia-pages.yml` workflow.
The Python pipeline runs locally; publishing a site does not activate a refresh schedule.

## Publication record

The final homepage update preserves “Why this repo exists” verbatim from the
GitHub README. It was published on 3 October 2026, following the navigation and
mobile-table improvements. Other homepage copy and layout remain improved.

- UI source commit: `852049df1f695115f997e51fc7109c7729904c8d`.
- Deployment checkout: `852049df1f695115f997e51fc7109c7729904c8d`.
- Release: `20261003T124334Z-08e8e0eb65d4`.
- [Successful Pages workflow](https://github.com/apur27/SuperCoach-VIA/actions/runs/37124172045).
- [Public prerelease archive](https://github.com/apur27/SuperCoach-VIA/releases/tag/preview-verbatim-2026-10-03).
- Site seal: `569bb2dc78f92c1f32b9885bf5b38a3bbbcd4edaf68fba74b3dfaf8325efb2bc`.
- Archive SHA-256: `a475bc341e264fcfad52fac3aaadda11b548ae7521f871d017eec9e26fb86f14`.
- Site size: 264,074,405 bytes, within the 300 MiB budget.

[Astra's plan](reviews/ASTRA_SOL_USABILITY_PLAN.md) and
[final review](reviews/ASTRA_SOL_USABILITY_REVIEW.md) record the first changes.
The [purpose plan](reviews/ASTRA_PURPOSE_PLAN.md) and
[follow-up review](reviews/ASTRA_PURPOSE_REVIEW.md) cover the README story.
Local checks passed: 181 unit tests (two existing skips), 354 browser tests
(four existing skips), plus two exact-case regression tests. The alias passed
23 mapping and nine browser checks. The final verbatim update passed 28 focused
homepage/mobile/no-JavaScript checks, including exact text comparison with the
README, and Astra’s eight-state content and visual review.
Type checks, lint, budgets, seal validation and the full input-consistency audit passed (`semantic_complete=true`).
Before the purpose-only addition, the real-site review passed 184 states and all
14 downloads. Final live checks passed 24 states, search, accessibility and no-JavaScript navigation. Six deployed
pages, manifests and assets matched the sealed bytes.

The [preceding purpose source’s web CI job passed](https://github.com/apur27/SuperCoach-VIA/actions/runs/37123467071).
Repository-wide CI remains **not green**. The [recorded Python failure](https://github.com/apur27/SuperCoach-VIA/actions/runs/37122256985)
comes from a browser-contract test assuming `/usr/bin/node` exists on the runner. The
[legacy Unit Tests workflow](https://github.com/apur27/SuperCoach-VIA/actions/runs/37122141315)
also fails collection because its environment lacks dependencies including
`pydantic` and the installed `supercoach_via` package. No CI gate was weakened.
The Pages workflow separately verified and published the sealed artifact.

Evidence is under `var/ui-review/20261003/` (outside Git). Hash checks confirmed
that 90 paused Claude-worktree files, 604 retained candidate files and three
pre-existing memory files were unchanged.

### Lowercase and mixed-case entry points

The project repository remains `apur27/SuperCoach-VIA`; its canonical base is
`/SuperCoach-VIA/`. A separate [owner Pages repository](https://github.com/apur27/apur27.github.io)
publishes the seven files listed in [ops/pages-alias](../ops/pages-alias/README.md).
Its [Pages deployment passed](https://github.com/apur27/apur27.github.io/actions/runs/37122421896)
at commit `4d391ea16e35d390a88d4b1ca3fc54897a57dff9`.

Live browser checks confirmed lowercase entries with and without a trailing
slash return 200 and open the project. Deep and mixed-case entries preserve
paths, queries and fragments, but initially return the owner's custom 404 before
JavaScript redirects. Without JavaScript, a plain project-home link is available.
Unrelated paths and canonical addresses do not redirect. The live redirect modules
have JavaScript MIME types and match their reviewed bytes. This does not make
GitHub Pages routing itself case-insensitive.

### Previous previews retained for rollback

The [earlier purpose archive](https://github.com/apur27/SuperCoach-VIA/releases/tag/preview-purpose-2026-10-03)
contains release `20261003T123306Z-fe8a6ca23166` from source `1510eb967`.
It contains the earlier adapted story, before the owner requested verbatim text.
Seal: `d0bbee6ccc192e373a9a58c6424a0b452243f2cdf359ab51980d689d8a0b7deb`;
archive SHA-256: `f838b5ee63deee4b7bfb5999ce58d06489acc2f67c0e29e9914c254e7f43a7f6`.
Its [deployment](https://github.com/apur27/SuperCoach-VIA/actions/runs/37123557820) is retained.

The [usability archive](https://github.com/apur27/SuperCoach-VIA/releases/tag/preview-ui-2026-10-03)
contains release `20261003T113058Z-7662fe99ba96`, from UI source `7e4098bd2`.
Seal: `65bbef200a2326c697240f0b0813f06d3c47c28ec700a049d918348ba24d9201`;
archive SHA-256: `7e9264de620619ebe7e67fc9e5f40f1a1ebcdab754970f6d47d93a7f72cc1e50`.
Its [deployment](https://github.com/apur27/SuperCoach-VIA/actions/runs/37122424667) is retained.

The [first archive](https://github.com/apur27/SuperCoach-VIA/releases/tag/preview-2026-10-03)
contains release `20261003T105359Z-67fb4988599a`, from UI source `a5b2b804a`.
Its seal is `56fb11cb70e0fa4b30647743467843e964babe53b288cdf970f3410276965de1`;
archive SHA-256 is `c3bf3235a4fed6060bdb80c8825555cf9cb624634f53b85101a195134d9d6ae0`.
Its [deployment](https://github.com/apur27/SuperCoach-VIA/actions/runs/37118887973)
and evidence under `var/pages-preview/20261003/` are retained.

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
