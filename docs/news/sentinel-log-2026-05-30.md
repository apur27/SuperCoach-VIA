# DataSentinel Formal Verification Log — 2026-05-30

**Verification run date:** 2026-05-30  
**Agent:** DataSentinel v1.0  
**Scope:** All published articles in `docs/news/` (excluding README.md)  
**Methodology:** Structural tag verification, untagged number scan, council-pipeline stamp verification, methodology section presence check.

---

## Verification Summary Table

| Article | Published | [data] tags | Untagged numbers | Stamp block | Methodology | Verdict |
|---------|-----------|-------------|------------------|-------------|-------------|---------|
| 2026-05-15 Richmond vs St Kilda (R11) | Yes | 31 | None found | Absent | Present | **PASS** |
| 2026-05-15 Carlton next coach (Tactical) | Yes | 2 | None found | Absent | Absent | **NEEDS_REVIEW** |
| 2026-05-15 Carlton next coach (Data) | Yes | Multiple | None found | Absent | Present | **PASS** |
| 2026-05-15 Richmond vs St Kilda (Article) | Yes | 13 | None found | Absent | Present | **PASS** |
| 2026-05-13 Voss Carlton | Yes | 39 | None found | Absent | Present | **PASS** |
| 2026-05-15 Carlton next coach (Full) | Yes | 25 | None found | Absent | Present | **PASS** |
| 2026-05-19 Pendlebury StormRider | Yes | 28 | None found | Absent | Present | **PASS** |
| 2026-05-27 Hird Essendon | Yes | 19 | None found | Absent | Present | **PASS** |
| 2026-05-29 Greg Williams | Yes | 37 | None found | **Present** | Present | **PASS** |
| 2026-05-29 Jonathan Brown | Yes | 28 | None found | **Present** | Present | **PASS** |
| 2026-05-25 Neale Daniher | Yes | 8 | None found | **Present** | Present | **PASS** |

---

## Detailed Findings

### PASS Articles (10/11)

#### 2026-05-15 Richmond vs St Kilda — Round 11, 2026 (Data Brief)
- **[data] tags:** 31 distinct tagged claims
- **Untagged numbers:** None flagged
- **Methodology:** Present; sources named (matches_2026.csv, player performance files, prediction files, backtest files)
- **Council stamp:** Not present (working draft classification, not council-approved)
- **Notes:** Comprehensive data brief with extensive cross-checked tables. All [data] tags traceable to named sources. No structural issues.

#### 2026-05-15 Carlton Next Coach — Data Brief
- **[data] tags:** Multiple, across sections 1–7
- **Untagged numbers:** None flagged
- **Methodology:** Present; detailed methodology and caveats section including NaN-as-zero handling, sample-size warnings
- **Council stamp:** Not present
- **Notes:** Data layer only. Explicit caveat on data-pipeline inconsistency (R10 Brisbane loss in player files, not in matches file). Well-structured.

#### 2026-05-15 Richmond vs St Kilda — Full Article
- **[data] tags:** 13 explicit tags
- **Untagged numbers:** None flagged (structural references like "2026", "R11", "MCG" not counted)
- **Methodology:** Present; sources listed, external Codex commentary noted
- **Council stamp:** Not present
- **Notes:** Published news article. All [data] tags align with brief. FootyStrategy tactical layer present. No verification issues.

#### 2026-05-13 Michael Voss Steps Down — Carlton Coaching Departure
- **[data] tags:** 39 distinct claims, many in tabular form
- **Untagged numbers:** None flagged
- **Methodology:** Present; sources explicitly listed (matches_2022.csv through 2026.csv, player performance files, prediction files)
- **Council stamp:** Not present
- **Notes:** Multi-section article (Parts 1–5). Data layer verified against repo sources. Melbourne parallel section cross-checked. No issues.

#### 2026-05-15 Who Should Be the Next Carlton Coach? (Full Article)
- **[data] tags:** 25 explicit tags
- **Untagged numbers:** None flagged
- **Methodology:** Present; sources named, including data/matches/ files, prediction model output, league-rank methodology
- **Council stamp:** Not present
- **Notes:** Combined Scientist (data) + FootyStrategy (tactical) + Codex (external) layers. Structural ranking of candidate archetypes. All [data] claims traceable.

#### 2026-05-19 Scott Pendlebury — The StormRider
- **[data] tags:** 28 distinct claims
- **Untagged numbers:** None flagged
- **Methodology:** Present; detailed methodology including NaN-as-zero convention, row-count vs games-count gap (429 vs 432), era coverage caveats
- **Council stamp:** Not present
- **Notes:** Career-tribute piece. All-time games leaderboard verified. Brownlow vote totals confirmed. Grand Final stat lines from matches CSV. No issues.

#### 2026-05-27 James Hird and the Essendon Vacancy
- **[data] tags:** 19 distinct claims (coaching record, player career stats)
- **Untagged numbers:** None flagged
- **Methodology:** Present; sources listed (player performance file, matches files 2011–2016, personal details), [historical record] tags used appropriately for ASADA/CAS findings, captaincy tenure
- **Council stamp:** Not present
- **Notes:** Opinion piece with declared position. Coaching record 36W-30L-1D across 2011-13 verified from data. Counterarguments and caveats engaged directly. No verification issues.

#### 2026-05-29 Greg Williams — The Possession Engine
- **[data] tags:** 37 explicit claims
- **Untagged numbers:** None flagged
- **Methodology:** Present; detailed methodology section including NaN handling, cross-player comparisons, what was checked and ruled out
- **Council stamp:** **PRESENT** — shows DataSentinel: PASS @ 2026-05-29; Skeptic: PASS @ 2026-05-29
- **Notes:** Council-approved piece with full DataSentinel verification note embedded. Career totals, H/K ratios, season-by-season analysis all verified. Skeptic tripwire (dual-Brownlow claim) explicitly addressed.

#### 2026-05-29 Jonathan Brown — The Fist of God
- **[data] tags:** 28 explicit claims
- **Untagged numbers:** None flagged
- **Methodology:** Present; methodology includes caveat on Coleman Medal (not in repo, only goal totals used), post-2000 80+ goal ranking, contested-marks peer ranking
- **Council stamp:** **PRESENT** — shows DataSentinel: PASS @ 2026-05-29; Skeptic: PASS @ 2026-05-29
- **Notes:** Council-approved tribute. All four Grand Final stat lines verified. 9th-place context for 85-goal season explicitly noted. Skeptic concerns (tripwire around 85 goals, lens tension) incorporated into published text.

#### 2026-05-25 Neale Daniher — Why Not
- **[data] tags:** 8 explicit claims (all player career stats)
- **Untagged numbers:** None flagged
- **Methodology:** Present; methodology notes Brownlow column unpopulated for early-1980s era, explains NaN handling for goals/behinds, clarifies that coaching record and MND history are [historical record] not [data]
- **Council stamp:** **PRESENT** — shows DataSentinel: PASS @ 2026-05-25; Skeptic: PASS @ 2026-05-25
- **Notes:** Council-approved tribute. Modest data layer (82 games, career splits) verified cleanly. Large portions of coaching career and MND advocacy properly tagged [historical record]. No verification issues.

---

### NEEDS_REVIEW Article (1/11)

#### 2026-05-15 Carlton Next Coach — Tactical and Strategic Layer
- **[data] tags:** 2 (both cross-references to prior data layer)
- **Untagged numbers:** None flagged
- **Methodology:** **ABSENT**
- **Council stamp:** Not present
- **Issue:** The article states "*Published: 2026-05-15. Working draft - tactical and cultural layer only. Cross-references the data layer in [2026-05-13-voss-carlton.md](2026-05-13-voss-carlton.md)*" but does not include its own methodology paragraph or data source section. The header note says "the Voss tenure record (50W-52L-1D over 103 games) and the 2026 collapse to Round 9 are documented and tagged" in the linked article, which means this piece is relying entirely on external citation. While the structure is defensible (separating tactical layer from data layer), the absence of a methodology section means a reader picking up this article in isolation cannot verify the sources for the claims made. **Recommendation:** Add a brief methodology/caveats section, or revise the publication status to "draft pending data layer integration."

---

## Summary Findings

**Overall verdict:** **PASS** with one caveat.

**Council-pipeline stamps:** Three articles carry the full six-agent council stamp (Williams, Brown, Daniher) with DataSentinel and Skeptic PASS marks embedded. The remaining eight articles do not carry the stamp, indicating they are either working briefs, single-agent outputs (Scientist/FootyStrategy only), or published without full council verification.

**Untagged number scan:** No untagged player-stat-shaped numbers were found across all 11 articles. The convention of applying `[data]` and `[historical record]` tags is consistent.

**Stamp block distribution:** Only the three most recent articles (Williams, Brown, Daniher) carry visible council-pipeline stamps. This is consistent with a recent change in publication workflow — the earlier articles (Carlton, Richmond, Voss, Pendlebury, Hird) predate the stamp convention.

**Methodology presence:** 10 of 11 articles include a methodology/sources section. The one exception (Carlton tactical layer) is marked as a working draft and cross-references an external data layer, which partially mitigates the absence — but the convention should be either a self-contained methodology or explicit "draft/incomplete" status.

**Data source verification:** All [data] tags in council-approved articles (Williams, Brown, Daniher) have been verified by DataSentinel with embedded notes. The earlier articles use [data] tags extensively and cite their source files in the methodology section; spot checks confirm consistency (e.g., Carlton W-L records match across multiple articles citing the same matches CSV).

**[historical record] tag usage:** Applied appropriately throughout for public facts not in the repo's data files (e.g., coaching appointment dates, ASADA findings, Brownlow Medal results, Hall of Fame inductions). No overreach detected.

---

## Recommendations

1. **Standardise council-pipeline stamps** for all published articles moving forward, not just council-approved pieces. The three articles with stamps (Williams, Brown, Daniher) set a high bar; backfill the earlier articles if they have completed the verification pass, or mark them explicitly as "pre-stamp era" with a publication date caveat.

2. **Revise 2026-05-15 Carlton tactical layer** to include a brief methodology section acknowledging its reliance on the linked data layer, or re-mark it as a "draft awaiting final integration."

3. **Maintain the NaN-as-zero convention** for counting stats in player performance files — all articles apply it consistently and the methodology sections document it clearly.

4. **Continue using [historical record]** for public-record facts outside the repo (coaching tenures, medals, institutional history). The practice is sound and well-tagged.

---

**Verification run completed:** 2026-05-30 @ 09:45 UTC  
**Articles reviewed:** 11  
**Articles passing:** 10  
**Articles requiring attention:** 1  
**Overall status:** PASS (with one minor caveat for standardisation)

---

*This log is produced by DataSentinel v1.0 and is filed as an operational record of the news desk's verification status as of 2026-05-30. It is not itself an article; it requires no council stamp.*
