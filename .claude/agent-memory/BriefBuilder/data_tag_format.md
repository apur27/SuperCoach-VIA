---
name: Structured [data] tag format
description: All new [data] tags must use the 4-field structured form; spec lives at docs/data-tag-spec.md
type: feedback
---

All new `**[data]**` tags must use the structured 4-field form:
`<value> **[data: <file> ; <filter> ; <column> ; <aggregation>]**`, ` ; ` separated.
Full spec (grammar, aggregation vocabulary, `derived:` escape hatch, parsing regex,
DataSentinel contract) is at `docs/data-tag-spec.md`.

**Why:** bare `**[data]**` only named the source in the methodology paragraph, so
DataSentinel had to guess file + filter + column + aggregation. The structured tag makes
verification deterministic. (Fix 2 of the DataSentinel-friendliness work, 2026-06-03.)

**How to apply:** use it for every new tag. `filter=all` when no filter; `derived:<expr>`
(e.g. `derived:wins`, `derived:percentage`) for season-record values that don't map to a
single column. Values not expressible in the schema use **[unverified]** /
**[historical record]** / **[unavailable — stat not recorded in era]**, never a bare tag.
Existing briefs are NOT retrofitted — new briefs only.
