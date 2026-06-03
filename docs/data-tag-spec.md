# Structured `[data]` tag specification — v1

Status: active. Applies to **all new `[data]` tags** written by BriefBuilder (and any
agent producing data-backed docs) from 2026-06-03 onward. Existing briefs are not
retrofitted.

## Why

The old tag was bare: `29.4 **[data]**`. The source file was only named in the
methodology paragraph, so DataSentinel had to guess which file, which rows, and which
column produced the number. The structured tag makes verification deterministic: every
tag carries the exact file + filter + column + aggregation needed to recompute the value.

## Format

```
<value> **[data: <file> ; <filter> ; <column> ; <aggregation>]**
```

Four fields, fixed positional order, separated by ` ; ` (space-semicolon-space).

| Field | Meaning | Notes |
|---|---|---|
| `file` | Source CSV basename only, no directory | e.g. `sinclair_jack_12021995_performance_details.csv`. The directory is implied by file-naming convention and recorded in the methodology paragraph. |
| `filter` | Row-selection predicate, or `all` | Multiple conditions joined with ` & `. Use `all` when every row is used. |
| `column` | Source column name, or `derived:<expr>` | Must be a literal column from the file's header, OR a `derived:` formula when the value is computed from multiple columns / does not map to one column. |
| `aggregation` | How rows collapse to the value | Closed vocabulary, see below. |

### Why `;` and not `,`

Filters routinely contain commas-in-spirit (multiple conditions) and ranges. The
separator must be a character that never appears inside a field. Semicolons do not
appear in column names, filenames, filter predicates, or aggregation keywords, so a
regex can split on ` ; ` safely. Within the `filter` field, conjunction is ` & `.

## Filter grammar

- Equality: `column==value` — e.g. `year==2026`, `player==Sinclair Jack`, `team==Sydney`
- Range on rounds: `round>=8 & round<=12` (preferred), or the shorthand `R8-R12`
- Conjunction: ` & ` between conditions — e.g. `year==2026 & round>=8 & round<=12`
- No filter: `all`

String values are written bare (no quotes). Match them against the file exactly as
stored. If a value itself contains ` ; ` or ` & ` the tag cannot be expressed — fall
back to `**[unverified]**` and explain in methodology (this has not yet occurred in
practice).

## Aggregation vocabulary (closed set)

| Keyword | Meaning |
|---|---|
| `mean` | Arithmetic mean over filtered rows, skipna |
| `last5-mean` | Mean over the 5 most recent filtered rows by date/round |
| `lastN-mean` | Mean over the N most recent filtered rows (state N, e.g. `last8-mean`) |
| `sum` | Sum over filtered rows, skipna |
| `count` | Number of filtered rows (or rows meeting a boolean derived condition) |
| `max` / `min` | Extreme value over filtered rows |
| `value` | Single-cell read; the filter MUST select exactly one row |
| `streak` | Trailing run length (for W/L streaks) |

For counting-stat means (tackles, marks, hit-outs) the existing `(N of M games
recorded)` annotation rule still applies and sits **after** the tag, e.g.
`4.2 **[data: ... ; tackles ; mean]** (12 of 13 games recorded)`.

## The `derived:` column escape hatch

Several brief values do not map to a single CSV column. The clearest case is the
**season records** table: `matches_2026.csv` stores `team_1_team_name`,
`team_2_team_name`, and quarter/final goals+behinds — there is no `wins` column and no
`team` column. For these, the `column` field uses a `derived:` prefix naming the
computation so DataSentinel checks a formula, not a literal column.

- `derived:wins` — count of matches where the named team's final score > opponent's
- `derived:percentage` — points-for / points-against × 100
- `derived:margin` — winning final score − losing final score

The `filter` for a derived value still names the team, e.g.
`team_1_team_name==Sydney | team_2_team_name==Sydney`. Keep derived expressions to a
named, well-known quantity — if a value needs a bespoke multi-step computation, it is
Scientist's territory, not a `[data]` tag.

## Parsing regex (reference)

A single tag is matched by:

```
\*\*\[data:\s*([^;]+?)\s*;\s*([^;]+?)\s*;\s*([^;]+?)\s*;\s*([^\]]+?)\s*\]\*\*
```

Capture groups, in order: `file`, `filter`, `column`, `aggregation`. Each is then
trimmed and parsed per the grammar above. Because the separator is ` ; ` and never
appears inside a field, the four groups are unambiguous.

## Worked examples

| Rendered tag | Reads as |
|---|---|
| `29.4 **[data: sinclair_jack_12021995_performance_details.csv ; year==2026 ; disposals ; mean]**` | mean of `disposals` over 2026 rows |
| `27.8 **[data: sinclair_jack_12021995_performance_details.csv ; year==2026 & round>=8 & round<=12 ; disposals ; mean]**` | mean of `disposals` over 2026 R8–R12 |
| `27.8 **[data: sinclair_jack_12021995_performance_details.csv ; year==2026 ; disposals ; last5-mean]**` | mean of `disposals` over the 5 most recent 2026 rows |
| `10 **[data: matches_2026.csv ; team_1_team_name==Sydney \| team_2_team_name==Sydney ; derived:wins ; count]**` | count of 2026 matches Sydney won |
| `27 **[data: next_round_13_prediction_20260525_1929.csv ; player==Sinclair Jack ; predicted_disposals ; value]**` | the single predicted-disposals cell for that player |
| `−2.64 **[data: backtest_by_team_20260525_1929.csv ; team==Hawthorn ; bias ; value]**` | the Hawthorn per-team bias cell |

## DataSentinel contract

For every structured `[data]` tag, DataSentinel:
1. Parses the four fields with the regex above.
2. Opens `<file>` (resolving the directory from naming convention / methodology paragraph).
3. Applies `<filter>` and counts the surviving rows (≥1; exactly 1 for `value`).
4. Computes `<aggregation>` on `<column>` (or evaluates the `derived:` formula).
5. Compares the result to the rendered `<value>` within rounding tolerance (one decimal place as displayed).

A tag that fails to parse, names a missing file/column, selects zero rows, or whose
recomputed value disagrees, fails the gate. Numbers that genuinely cannot be expressed
in this schema use `**[unverified]**`, `**[historical record]**`, or
`**[unavailable — stat not recorded in era]**` instead, with an explanation in the
methodology paragraph.
