---
name: news-article-dossier-workflow
description: Workflow pattern for BriefBuilder-drafted news article skeletons from a Scientist dossier — data vs historical record split, large-table tagging, Gaffer restrictions
metadata:
  type: feedback
---

When drafting a news article skeleton from a Gaffer-provided Scientist dossier:

**Medal awards are always [historical record].** The vote count that underlies the award is [data]; the award designation itself (Brownlow Medal, Norm Smith Medal, All-Australian) is [historical record] because no medal field exists in any CSV. Keep these strictly separate — never tag "he won the X Medal" as [data].

**Why:** The Gaffer explicitly separated these for the Dustin Martin article. A [data] tag on a medal award would fail DataSentinel because there is no CSV column to verify against.

**How to apply:** In any feature article touching individual awards, apply this split. Verify the underlying vote/count from CSV ([data]), then frame the award itself as [historical record] with a methodology note.

---

**Large per-season tables (15+ rows × 6 cols): use table-level source declaration.** Include a source note above the table declaring file/filter/column/aggregation pattern. Tag individual cells in the narrative text with full structured [data] tags when those specific cells are cited directly. For the full grid, DataSentinel relies on the source note + methodology paragraph combination.

**Why:** 90-cell tables with full structured tags per cell are unreadable and impractical. The source note + methodology approach gives DataSentinel the same traceability.

**How to apply:** Any per-season or per-year summary table with more than ~10 data cells. Always include the aggregation declaration in the note (sum vs count vs derived:mean).

---

**Associational caveat framing for career-split comparisons.** When showing H&A vs finals stat splits, the comparison table must include a blockquote or note: "X is associated with Y — this is observational across N games and does not establish causation." The FOOTYSTRATEGY INSERT then handles interpretation.

**Why:** The Gaffer specified "ASSOCIATIONAL, not causal — state it that way" for the Martin finals goals/game lift (+55%). FootyStrategy may be tempted to editorialize; the [data] layer must pre-anchor the framing.

---

**No council-pipeline provenance stamp on news articles.** The Gaffer applies that stamp at ship time. BriefBuilder includes a minimal "Data layer by BriefBuilder v1.0" line but not the full pipeline stamp.

**How to apply:** News articles (docs/news/*.md) only. Pre-match briefs (docs/coaches-strategy-corner/*.md) follow the standard brief structure with their own header.

---

**GF round_num encoding in matches CSVs.** Grand Final rows use `round_num = "Grand Final"` (string), not a numeric code. EF/QF/SF/PF rows use their abbreviation strings. This differs from player CSV where the `round` column also uses these string values directly.
