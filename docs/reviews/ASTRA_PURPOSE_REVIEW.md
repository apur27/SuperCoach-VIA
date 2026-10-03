# Astra verbatim section review — PASS

Reviewed release `20261003T124334Z-08e8e0eb65d4` at `/SuperCoach-VIA/` against the authoritative GitHub capture `../why-verbatim.md`.

**PASS. The “Why this repo exists” section is verbatim.**

- Compared the rendered section as eight ordered blocks: exact H2, semantic opening blockquote and six body paragraphs. Every word and punctuation mark matches the captured source; only HTML layout whitespace was normalised.
- Also compared the entire section text to the joined source blocks, detecting any additional attribution or prose. No added or rewritten text remains inside the section.
- Verified exact content at 320/390/768/1440px in both themes and with JavaScript disabled.
- The existing `#why-this-project` anchor works by keyboard and without JavaScript. The section retains its prominent placement before the statistics.
- Actual screenshots show readable prose and blockquote layout. No document horizontal overflow or browser page errors in the eight sampled states.
- The product introduction, improved navigation/layout and search remain outside the copied section. They are not subject to the verbatim comparison.
- Provisional audit-failure notice, snapshot notes, unavailable forecast and noindex remain present.

No required corrections remain for this change. Broader release/publication verification is owned by the parent.

Evidence:

- `review.mjs` and `review.json`: exact comparisons, rendered states and checks.
- `home-{320,390,768,1440}-{light,dark}.png`: complete homepage captures.
- `intro-*`: opening viewport; `story-*`: section reached through its anchor.
- `story-nojs-390.png`: original section and native anchor without JavaScript.

Suggested user-facing evidence: `intro-390-light.png`, `story-1440-dark.png`, `home-390-light.png`.

This review supersedes the adapted-purpose-copy signoff in `../purpose/ASTRA_PURPOSE_REVIEW.md`. No source edits or preview-server termination by the reviewer.

## Implementation checks

All 28 focused browser checks passed at the root and project base paths. Type
checks and lint passed. No CSS, data, harness or schedule changes were made in
this correction. The sealed site is 264,074,405 bytes, within the 300 MiB budget.

## Release checks

Release `20261003T124334Z-08e8e0eb65d4` passed seal validation and all 26 full
input-consistency checks (`semantic_complete=true`, five warnings and one
informational finding). Report SHA-256:
`fc67a23f8388daf8bffb9eb8cc92c30ce87c1dc21805d16da4956e65af985d5c`.

The separate AFL Tables source-audit FAIL remains open. This content correction
changes no source dataset, forecast eligibility, Claude work, harness or schedule.
The [publication record](../pages-preview.md) lists the deploy and rollback links.


## Live verification

The [published section](https://apur27.github.io/supercoach-via/#why-this-project)
was compared directly with the GitHub README capture. The exact heading,
blockquote and six paragraphs passed in both themes at mobile and desktop widths;
only HTML layout whitespace was normalised. The same paragraph check passed
without JavaScript after following the lowercase entry's fallback link.

The final live sweep passed 24 page states, eight redirect/non-redirect cases,
search and accessibility scans. Six live pages, manifests and assets matched the
sealed bytes. Final screenshots and `live-review.json` are under
`var/ui-review/20261003/verbatim/`. All 697 protected files remained unchanged.
