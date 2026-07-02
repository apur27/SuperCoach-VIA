---
name: hof-full-table-regen
description: How HOF stat-page full-table regeneration is wired in update_hof_pages.py, and the two gotchas (inert whitelist, name convention)
metadata:
  type: project
---

# HOF full-table regeneration (scripts/update_hof_pages.py)

Full-table (ranks 1-20) regeneration from `docs/hall-of-fame/_stat_leaders.json` is gated by
`_FULL_TABLE_CATS` (a frozenset of category keys). All tables listed there use the SAME standard
7-column layout rendered by `build_full_table_row`:
`| # | Player | Club(s) | Span | Games | <Stat> | Per game |`
(`games_only` cats drop the Games/Per-game trailing columns.) There is NO multi-column disposals
(kicks+handballs) or goals (seasons/career-totals) table on the live pages — kick/handball
composition and era notes live in PROSE only. `fmt: "thousands"` adds comma grouping to the total.

**Gotcha 1 — whitelisting is INERT without doc markers.** `run_updates` regenerates a table only
if the target doc has BOTH `<!-- HOF-TABLE-START:<key> -->` and `<!-- HOF-TABLE-END:<key> -->`
around the rows. `replace_table_body` returns `changed=False` (graceful no-op) if either marker is
missing. So adding a key to `_FULL_TABLE_CATS` does nothing until the doc page also carries the
markers. As of 2026-07-03, disposals and goals pages were whitelisted in code but still LACKED the
markers (the other 8 whitelisted pages have them) — flagged to Gaffer as a doc edit outside a
code-only scope.

**Gotcha 2 — name convention: bare in table, suffixed in prose.** The JSON stores "Gary Ablett"
(no jnr/snr) for both Abletts. The established convention across whitelisted pages (games,
contested, brownlow) is: bare "Gary Ablett" INSIDE the regenerated table rows, "Ablett jnr"/"snr"
for disambiguation in PROSE only. When a page has jnr/snr inside the table row (disposals/goals did),
that's the anomaly — regen normalizes it to bare, which is correct, not a regression.

Tests: `tests/unit/test_update_hof_pages.py`. Pattern uses `_page_with_table_sentinels` helper +
`replace_table_body` on tmp pages. Unrelated pre-existing failures live in
`test_drawn_gf_integrity.py` (Sidebottom 366 vs 367 games_played row-count mismatch).
