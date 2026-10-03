# Verbatim README purpose correction

The user explicitly requires copying the GitHub README section verbatim, not rewriting it. This replaces the earlier adapted purpose copy and its prior signoff.

Authoritative input: the [README section](../../README.md#why-this-repo-exists), fetched from GitHub main and checked against the local file. The captured source is `var/ui-review/20261003/why-verbatim.md` (outside Git). Astra read it in full.

## Bounded implementation

1. Restore the exact previous product lede: “SuperCoach VIA brings together player and match history, comparisons, rankings and downloads.” Remove the adapted “A noncommercial AFL project…” lede. This avoids retaining a second rewritten version of the motivation.
2. Use the exact text “Why this repo exists” for both the prominent intro link and section H2. Preserve the existing `href="#why-this-project"` and section ID for URL stability.
3. Replace the complete adapted story with the authoritative section: exact H2, original opening blockquote and six paragraphs, in order. Preserve every word, punctuation mark, apostrophe, hyphen and em dash. Translate Markdown structure to semantic HTML only: heading, blockquote and paragraphs. Do not modernise wording, fix style, add summaries or change the first-person voice.
4. Remove the added “From the project creator” attribution. No extra editorial text belongs inside the copied section.
5. Preserve the current visible placement before season results, existing prose measure and body typography, search/actions, warning text and all other homepage content. No new About page, data change or layout redesign is needed. Give the blockquote a modest scoped margin/border only if default styling causes mobile clipping; do not change its words.

## Verification

- Compare the rendered section’s ordered blocks against `why-verbatim.md`: one exact H2, one blockquote with exact opening text, six exact body paragraphs. Normalise HTML layout whitespace only. Preserve punctuation differences and fail on paraphrase.
- Confirm the adapted lede, story and attribution are absent.
- Capture 320/390/768/1440px light/dark; the longer story may occupy more vertical space, as requested. Do not shorten it to meet a height target.
- Confirm keyboard/no-JS anchor behaviour, no document overflow, readable blockquote/prose, and unchanged search/provisional notice/forecast state.
- Astra reviews exact rendered content and actual screenshots after Sol implements. Parent handles publishing and release checks.

Source scope: homepage and focused tests; a scoped blockquote style only if necessary. Do not edit the parent’s pending README/docs publication updates.
