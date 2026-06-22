---
name: feedback-data-block-rounding
description: When a DATA BLOCK supplies pre-rounded values, use those values exactly — do not recompute and write the raw decimal
metadata:
  type: feedback
---

When a pre-match brief or retrospective task supplies a DATA BLOCK with pre-computed, rounded values (e.g., disposal_avg 24.2, tackle_avg 2.8, regular_season_disposal_avg 24.4), use those values **exactly** in the document. Do not recompute from raw column sums and write the fuller decimal (24.23…, 2.765…, 24.36…).

**Why:** A prior BriefBuilder run on the Dustin Martin retrospective computed values directly from the data and wrote 24.24, 2.76, and 24.36 — all differing from the DATA BLOCK's 24.2, 2.8, and 24.4. The DATA BLOCK is the contract; it carries the rounding decision. Writing a different decimal creates a DataSentinel mismatch and a harder diff for Scientist to audit.

**How to apply:** When a task includes a DATA BLOCK or similar pre-supplied verified-number list, treat the listed values as definitive. Note the rounding explicitly in the Methodology paragraph (e.g., "disposal avg = 24.2 (not 24.24)") so DataSentinel can see the rounding choice was intentional. If the task does NOT supply a DATA BLOCK, compute from the raw data and apply standard rounding (1 decimal for averages).

Related: [[nan_counting_stats]] for N-of-M annotation discipline.
