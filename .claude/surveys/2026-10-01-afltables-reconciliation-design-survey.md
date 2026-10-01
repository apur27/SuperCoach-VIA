# Survey — 2026-10-01 — scope: DEEP (design review: AFL Tables all-player reconciliation)

- **Surveyor model (session metadata):** Opus 5.5, model ID `claude-opus-5-5`. This is the
  model requested by the user via Gaffer; no substitution occurred.
- **Commissioned by:** Gaffer, per `docs/rewrite/afltables-reconciliation/ARCHITECT_PROMPT.md` (design review mode).
- **Repo state:** `main` @ `fec9cd3d87cc7c8c823176a75f33dfe90246eeeb`, clean tree.
  `.claude/audit/last_refresh_status.json` = `{"phase":"4","exit_code":0,"round":"25","ts":"2026-08-29T13:49:38+1000"}`,
  so there is no active cycle and CLAUDE.md §6.1 does not apply.
- **Reviewed design:** `docs/rewrite/afltables-reconciliation/DESIGN.md` sha256 `d368d9d32d3da4ad045f9f4267946fc0191fa1621ff6dab094bd445d6aa2c2af`.
- **Read-only:** no edits to the design, prompts, code, data or other reviews. I made 27 live GETs
  to afltables.com (cap 30, at least 2.5 s between request starts). Inventory is in §8.

---

## Executive read

The design is sound in its bones. The census can be built from source links, every sampled
profile row in every era carries an explicit match link, the drawn GF and its replay have
distinct match URLs, and the existing independent reader's 28-column layout holds from 1897 to 2026.
**No finding is BLOCKING**: none needs the human owner, and none makes the requirement set infeasible.
However, five HIGH gaps would, as currently written, make the full-audit PASS unreachable by construction or produce false
SOURCE_CONFLICT/UNKNOWN results:
- pre-1984 Brownlow votes exist only as season-level summaries;
- 2021–22 unused medical subs are credited games with fully blank rows;
- printed averages use an unprinted, source-specific denominator;
- the existing HTTP client retries 429s early and does not expose `Retry-After`;
- identity rule 3 depends on exact names that diverge for multiword surnames.

All five have evidence-backed, routine resolutions Gaffer can write into DESIGN.md.
**Highest-leverage move:** add a "source-convention" section to DESIGN §8 that defines the
summary-only, credited-did-not-take-field and printed-average-denominator states, using the
fixtures captured in this survey.

**Surveyor recommends: APPROVE_WITH_CHANGES.** There are zero BLOCKING findings. S-01 to S-05 must
be folded into DESIGN.md before Phase B, and approval must bind to the amended design hash, not `d368d9d3…`.

---

## 1. What each page type can and cannot establish (measured)

| Page | Establishes | Cannot establish / caveats | Evidence |
|---|---|---|---|
| `stats_idx.html` | Links the All Players directory (`playersA_idx.html`), notes, per-season player lists (`/afl/stats/YYYY.html` for 1897–1964 and `YYYYs.html` for 1965–2026), and a status string "2026 … Status: End of season (Complete)" | Nothing about individual players | sample #2 |
| `players[A-Z]_idx.html` (directory) | H1 "All Players - A"; nav to all 26 letters; one link per profile; display name "Surname, First"; title attribute with club stints and years, e.g. `title="Collingwood (1907), Melbourne (1912)"` | **No DOB.** Club names in titles are lineage-modern ("Western Bulldogs (1925-1932)", "Sydney (1973-1976)"). Each letter regenerates independently (Last-Modified: A 2026-08-17, M 2026-08-24). A late debutant (Tobyn Murray, first game 2026-08-22) **is** present in M. | #3, #21 |
| Directory homonyms | Fully suffixed groups only (`Gary_Ablett0/1`, `Bill_Ahern0/1`). A: 400 unique links, 15 duplicated names, 31 suffixed URLs. M: 1,542 unique links, 101 suffixed, max suffix 3. No group mixes an unsuffixed base with suffixed URLs, and `Gary_Ablett.html` returns **404** | Suffix order is not an identity key. `Bill_Ahern0` (St Kilda 1897) and `Bill_Ahern1` (Carlton 1897) share a name **and** a season | #3, #19, #21 |
| Profile `players/<FirstInitial>/<First_Last[N]>.html` | `Born:` DOB on most sampled pages, e.g. "20-Jun-1915" and "25-Jun-1874". One game table per club-season, header `colspan=28` (5 meta columns + 23 stats) in every sampled era (1897, 1933, 1984–93, 2006–26, 2021). **Every game row has exactly one match link in the `Rd` cell**: 1,370 rows across 8 profiles, 0 without and 0 with more than one. Counter tokens carry ↑/↓. Season Totals footer combines GM and W-D-L in one cell, e.g. "26 (20-2-4)". Year "By totals" and "By averages" tables have Totals and Averages footers. | **DOB is absent** on some profiles: Kelly Robinson (Fitzroy 1897–1901) has no Born line. A placeholder-like "1-Jan-1959" (Allan Sidebottom) carries no precision marker. **Rows have no match date**; dates come from match or season pages. The Year table has **no %P column**, and %P is blank in season Totals. **No rowspans.** | #5–10, #16, #17 |
| Profile aggregates | Season BR totals exist **pre-1984** even when per-game BR cells are blank. Reynolds 1935: season BR 13 with no per-game BR; 1933–34 do have per-game BR. **The career Totals BR cell is blank.** | Season-summary-only statistics cannot be reconciled per game (S-01) | #6 |
| Printed averages | All 1,028 season-level averages in 8 profiles are explained: non-BR = total/GM; BR = total/home-and-away games (24 of 24 cases). Rounding is **ROUND_HALF_UP**, proven by exact ties: 28.125→"28.13", 0.625→"0.63", 3.625→"3.63". Career GM average = GM / distinct seasons (McMullin 49/8 = 6.125 → "6.13"; 9 rows over 8 seasons). W-D-L average is a win percentage ("65.31%"). | **Denominators are not printed.** The career TK average is 27/29 (McMullin's pre-1987 games excluded). Fully blank credited rows are **included** (Sexton 2021 KI 167/21 = "7.95"). The notes say exclusion applies "in most cases" (S-03). | #5–10 |
| `_gm.html` "Games Played" | Per game: match link, scores, venue, crowd, date | Redundant for link discovery; not needed (S-13) | #18 |
| Season `afl/seas/YYYY.html` | Fixtures with stage headings, scores, dates and a match link for every fixture: 2010 = 186, 2026 = 218, 1948 = 119, 1975 = 138, all with links. **Drawn GF + replay appear as two "Grand Final" fixtures** with distinct dates and URLs (2010: `041520100925`/`041520101002`; 1948: `051119481002`/`051119481009`) | **No "Replay" token anywhere.** The 2026 opening matches are labelled "Round 1" (5 games from 2026-03-05). The committed fixture `seas_2026_captured_20260925.html` predates the 2026 GF. | #14, #22–24 |
| Match `afl/stats/games/YYYY/<id>.html` | Header with Round/Venue/local Date/Attendance; per-team stats table (`#`, `Player` + 23 labels); a Totals row per team; a second per-team table with age and career-games-to-date, e.g. "139 (35-0-104 25.18%)". Sub arrows appear in the jersey token ("16 ↑", "29 ↓"). | **rowspan appears only in the header's prev/next navigation cells.** No explicit "unused sub" marker. The drawn GF page says "Drawn" in the margin row and links forward (→) to the replay. | #11–13, #15, #25, #26 |
| Match: availability | A blank team Totals cell means the category was not recorded for that team-match (1975 R11: only GL Totals printed; 1897 and the 1948 GF replay are goals-only). A non-blank Totals plus a blank player cell means zero (2026 sum identity, annotations.json). | A team column whose Totals is "0" was **not observed**; the encoding of an all-zero column is unknown. | #11, #15, #25, #26; `tests/scvia/fixtures/zero_semantics/annotations.json` |
| `notes.html` | Availability table 1965–2010 by column, with "From 2011, all categories are complete". Exception list by year/round/team. "In most cases, games where the stat is missing will not count when calculating average". "These statistics are *not official*". | **Nothing about pre-1965** (goals only must be inferred from match structure). Labels differ from the stats tables (`I5`, `OP` vs `IF`, `1%`). Team names are anachronistic ("1975 R14 Sydney" means South Melbourne). Pairs are listed in either order ("Footscray v Carlton" vs the season page's Carlton-home fixture). The SU column is never marked. | #4 |
| `robots.txt` | **HTTP 404**, `content-type: text/html`, 651-byte custom "Broked!" page, no validators. Under DESIGN §6 (an actual 404 permits the normal policy), acquisition may proceed. | Identical body to a missing profile (`Gary_Ablett.html`), so page identity must come from status plus H1/title, not body heuristics | #1, #19 |

## 2. Transport-reuse claims (ARCHITECT_PROMPT step 4) vs code today

| Claim to check | Code reality | Design adequate? |
|---|---|---|
| URL policy lacks census/notes paths | Confirmed. `config/source_policies.toml:37-42` (identical to the packaged copy) allows only `/afl/seas/YYYY.html`, `/afl/stats/games/YYYY/<6-20 digits>.html` and `/afl/stats/players/[A-Z]/[A-Za-z0-9_\-]{1,80}\.html`. `validate_url` rejects any fragment or query (`http.py:199-200`). All 1,942 sampled A/M profile URLs fit the player grammar, including hyphens (`Alf_Andrew-Street`). | **Mostly.** DESIGN §6 (l.247-250) requires a separate exact policy. It omits `/robots.txt` and does not say links are resolved against the page's final URL before fragment-stripping (S-12). |
| Retries cap Retry-After | Confirmed. `retry_after_max_s = 60.0` (`source_policies.toml:26`; `http.py:71`); `_backoff` uses `min(retry_after, cap)` (`http.py:501-503`); `test_http.py:199-204` asserts a 9999 s header sleeps 60 s **and then retries internally**. `FetchResult` has no Retry-After field, and failure results drop headers (`http.py:340-359, 694-703`). Defaults: `requests_per_second = 2.0` (0.5 s) and `max_concurrent_per_host = 2`. | **No.** DESIGN §6 (l.260-268) says to reuse the three-attempt bound and also that the coordinator must not retry before the server deadline. With today's client both cannot hold, and the coordinator cannot read the deadline (S-04). |
| Archives hold mutable validator files | Confirmed. `RawArchive` keeps `validators/<sha(url)>.json` beside write-once objects (`http.py:286-332`). `fetch()` reads them automatically (`624-628`) and rewrites them on every 200 that carries an ETag or Last-Modified (`645-646`). Live directory/notes/season responses send `Last-Modified`. | **Partially** (l.273-275 "keep mutable validators … separate"). Simpler and safer: disable conditional requests in reconciliation capture (S-09). |
| Player-source check is optional and observation-driven | Confirmed. `check_player_pages` selects pages from `source_observations` with adapter `afltables.player_page` (`checks_source.py:959-967`) and returns NOT_APPLICABLE when there are none (`:966-967`). The candidate snapshot has 3 such observations. Only 782 of 13,368 local players carry `source_urls`. | **Yes.** DESIGN §3 l.136 states this correctly and does not claim the existing check does the job. |
| Legacy `audit_player_career_totals` | Confirmed. It derives URLs from names (`game_scraper.py` `_player_url_from_csv_path`), uses `max(games_played)` for GM and `fillna(0).sum()`, has an era gate only for TK/CL, and returns `[]` on an unavailable page. Name-derived URLs fail for all-suffixed homonyms (`Gary_Ablett.html` → 404) and for initials outside the first-name rule. | **Yes.** DESIGN §3 l.134 is accurate. |

## 3. Ranked findings

### S-01 — Season-summary-only statistics (pre-1984 Brownlow) are undefined and would be misread as SOURCE_CONFLICT  [HIGH] [class 10, unwritten convention]
- **Location:** DESIGN §8 l.348-360 (state table), l.390-391 (compare game sums with printed season/career rows), l.397-402 (SOURCE_CONFLICT → UNKNOWN), §2 l.64-65 ("every supported statistic").
- **Evidence (observation, sample #6):**
  - Dick Reynolds' season rows print BR 1933=12, 1934=19, 1935=13 … 1950=3.
  - Per-game BR cells are populated only for 1933–34. The game-cell sums are {1933: 12, 1934: 19}, with no BR in any later season.
  - The career Totals BR cell is blank.
  - `notes.html` marks BR from 1984.
  - Local memory `data_stat_coverage_eras.md` says local BR is null for 1935–1983.
- **Reason:** Under §8.4, every pre-1984 vote-getter yields "game sum ≠ printed season row", which §8 turns into SOURCE_CONFLICT/UNKNOWN. The PASS of D16 becomes unreachable through a misclassification rather than a real conflict. The local per-game schema also cannot represent season-only votes, and the design does not say whether that counts as a local mismatch, an unsupported value or an exclusion.
- **Recommended outcome:**
  - Add an availability state such as `SOURCE_SUMMARY_ONLY`, assigned only by an evidence-backed rule: season row value present, per-game cells blank, and team Totals blank in the season's matches.
  - Account for it with exact counts per player and season. It is never a source conflict.
  - **Default that preserves strictness (Gaffer may adopt):** keep it in scope and report the local absence as its own finding category. That category is a data finding, so a data FAIL can coexist with accepted software.
  - Excluding it from the claim would narrow scope and needs the owner's explicit choice. Trade-off: exclusion keeps PASS reachable, but the attestation must name the exclusion.
- **Owner:** Gaffer (design text); Scientist (rule + fixture).
- **Required test:** `tests/scvia/unit/test_reconciliation_aggregate.py::test_summary_only_brownlow_is_not_source_conflict` and `::test_summary_only_statistic_counted_separately` (fixture: trimmed Reynolds profile + one 1935 Essendon match).
- **Effort:** S · Impact-per-day rank 1

### S-02 — Credited appearances with fully blank rows (2021–22 unused medical subs) have no state  [HIGH] [class 5/10]
- **Location:** DESIGN §8 state table l.353-360; §7 l.342-344 ("named-but-did-not-play/substitute roles"); T11.
- **Evidence:**
  - *Observation, #9 and #11:* Alex Sexton 2021 rows 139–141 (R6–R8) carry a counter and a result with every cell blank, %P included. The season GM "21 (7-0-14)" counts them, and the season KI average "7.95" = 167/21 includes them in the denominator.
  - *Observation:* on match page 2021 R6 (`162020210424`), Gold Coast lists 23 players, and Sexton is all-blank with jersey token "6" and no arrow. Sydney's sub pair carries arrows ("16 ↑" Campbell %P 55, "29 ↓" Hewett %P 14). Gold Coast's kicks Totals is 246, equal to the player sum.
  - *Inference:* Sexton was Gold Coast's unused medical sub, credited with a game.
  - *Local scale:* rows with no kicks, handballs or %P number 202 (2021), 183 (2022) and 5 each in 2023–25, all with counters.
  - `domain/blanks.py:12-18` and annotations.json keep such rows null.
- **Reason:**
  - The team-Totals rule ("non-blank Totals plus a blank cell means zero") would classify these cells as RECORDED_ZERO.
  - The participation context says the player did not take the field.
  - The source's averages treat the rows as games.
  - Without a stated rule, the implementation will either fabricate zeros or leave about 400 rows UNRESOLVED_BLANK, which means permanent UNKNOWN.
- **Recommended outcome:**
  - Add a participation state, e.g. `CREDITED_DID_NOT_TAKE_FIELD`, assigned by a deterministic rule: all 23 cells blank in a match where the team's %P is recorded, the player is listed in that match, and the profile credits the game.
  - Count the appearance for membership and counters. Treat counting cells as not-applicable for per-game equality, but include them as zero when reproducing source printed averages.
  - Keep local null vs source credited-blank as a visible "representation" category, not a numeric contradiction.
  - State whether the rule also covers 2011–2015 substitutes; verify that in the Phase E pilot.
- **Owner:** Scientist (rule, fixtures); Gaffer (design text).
- **Required test:** `test_reconciliation_cells.py::test_credited_unused_sub_row_is_appearance_not_zero_stats` (fixtures: samples #9 and #11 trimmed) and `::test_used_sub_arrow_tokens_preserved_raw`.
- **Effort:** S · Impact-per-day rank 2

### S-03 — Printed-average verification as specified is infeasible ("verify denominator exactly") and would generate false conflicts  [HIGH] [class 4/10]
- **Location:** DESIGN §8 l.388-395; §9 l.436 (PASS requires no source conflict); T13.
- **Evidence:**
  - *Measured, 8 profiles:* 1,028 of 1,028 season averages are reproduced exactly by total/GM, or total/H&A games for BR, with ROUND_HALF_UP.
  - Ties prove half-up rather than half-even: Pendlebury 2017 DI 450/16 = 28.125 → "28.13"; Reynolds 1935 GL 10/16 = 0.625 → "0.63".
  - Career McMullin TK "0.93" = 27/29 excludes 1984–86; HO "0.02" = 1/49 includes seasons with blank HO.
  - Denominators are not printed anywhere.
  - The notes hedge exclusion as "In most cases".
- **Reason:**
  - The source's denominator is its own convention: GM; BR over home-and-away games; era- or match-missing games excluded.
  - That convention differs from the design's "observed" count, which excludes blank rows.
  - "Verify the underlying … denominator exactly" (l.395) cannot be done against a value the page does not print.
  - Treating every printed-average disagreement as SOURCE_CONFLICT (UNKNOWN) would let a self-admittedly inconsistent derived figure block PASS. It also invites convention-shopping, which l.393-394 forbids.
- **Recommended outcome:**
  - Specify the source-average model in DESIGN: ROUND_HALF_UP; season denominator GM, BR uses home-and-away games; career denominator excludes era- and match-unrecorded games per notes and match Totals; GM average uses distinct seasons; W-D-L average is a win percentage.
  - Classify printed averages as a **source-consistency** check (`source_consistent` dimension) that never decides a local-layer verdict. Local totals and per-game cells remain the verdict basis.
  - Record a model miss with its candidate denominators; never pick the one that passes.
  - Report local aggregate denominators separately, as §8.3 already requires.
- **Owner:** Gaffer (design); Scientist (model + fixtures).
- **Required test:** `test_reconciliation_aggregate.py::test_display_rounding_half_up_proven_by_ties`, `::test_brownlow_average_uses_home_and_away_denominator`, `::test_career_average_excludes_unrecorded_era_games`, `::test_average_model_miss_is_source_consistency_not_local_fail`.
- **Effort:** M · Impact-per-day rank 3

### S-04 — Retry-After and spacing cannot be honoured through today's HttpClient; design text is self-contradictory  [HIGH] [class NEW / transport]
- **Location:** DESIGN §6 l.257-268; code `ingest/http.py:501-503, 632-703`; `config/source_policies.toml:19-27`; `tests/scvia/unit/test_http.py:199-204`.
- **Evidence:** See §2 row 2. The client sleeps `min(Retry-After, 60)` and retries in-process. `FetchResult` exposes neither the header nor the server deadline. The limiter state lives in memory (`HostRateLimiter._next`, `http.py:251-278`), so spacing is not enforced across a resume. Defaults are 0.5 s spacing and concurrency 2.
- **Reason:** "Reuse the … three-attempt bound" (l.260) and "must not retry earlier than a longer server deadline" (l.266-267) cannot both hold with the current client. A coordinator built on it would make up to two early retries on a long 429. That is exactly the impolite behaviour §6 forbids.
- **Recommended outcome:**
  - The reconciliation policy pins `requests_per_second = 0.5`, `max_concurrent_per_host = 1` and client-level `max_attempts = 1`. The coordinator owns retries, with a persisted next-eligible timestamp per host that survives resume.
  - `FetchResult` gains an **additive**, default-neutral field carrying the parsed Retry-After, or an equivalent hook, so the coordinator can honour the full deadline.
  - Default production behaviour stays byte-identical. Scientist records the CLAUDE.md §6.2 scope determination in writing: `http.py` is invoked by `scvia refresh` via `scripts/scvia_weekly.sh`, and test 2 applies only if production behaviour changes.
- **Owner:** Scientist.
- **Required test:** `test_reconciliation_capture.py::test_429_long_retry_after_persists_deadline_no_early_retry`, `::test_spacing_two_seconds_includes_retries_redirects_robots`, `::test_spacing_persists_across_resume`; plus an `ingest/http` regression proving default behaviour is unchanged (`tests/scvia/unit/test_http.py::test_retry_after_field_is_additive`).
- **Effort:** S · Impact-per-day rank 4

### S-05 — Identity rule 3 needs exact names that diverge from source; rule 1 covers only 5.8% of players  [HIGH] [class NEW / identity]
- **Location:** DESIGN §7 l.309-321.
- **Evidence:**
  - *Measured:* 782 of 13,368 local players have `source_urls`; 13,363 have a source-quality DOB and 5 unknown.
  - Name divergences against source:
    - local "Brendon Chee" (`last_name` "Chee") vs source "Ah Chee, Brendon" (`players/B/Brendon_Ah_Chee.html`);
    - local "Fred Abel" vs source "De Abel, Fred" (`players/F/Fred_De_Abel.html`, sample #27);
    - local `last_name` "Achkar" vs source "El Achkar".
  - Kelly Robinson's profile has no DOB (#16).
  - Sidebottom's DOB is "1-Jan-1959" (#17); there are 44 local Jan-1 DOBs.
  - In the A+M sample, 4 of 1,944 local display names have no exact source-name match.
  - Same-season homonyms exist (Bill Ahern0/1, both 1897).
- **Reason:** "Unique normalized-name plus full-DOB" (l.312) fails on surname-splitting variance, which is exactly the multiword-surname case l.317 lists. Those players would fall to "reviewed override", which means manual work per player and UNKNOWN meanwhile, although deterministic, non-fuzzy evidence exists.
- **Recommended outcome:** Add tested deterministic rules, with exact-set equality only and no similarity scores:
  - **(3b)** a globally unique full-DOB match plus an identical (season, club) membership set, with the name difference recorded as an identity-variance finding;
  - **(4)** for a missing or low-precision DOB, global uniqueness of an *exact* equality between the source profile's appearance set (match URL, club) and the local player's appearance set.
  - Any partial overlap stays UNKNOWN. An identity accepted through set equality cannot hide an appearance mismatch, because equality implies none.
  - Treat Jan-1 DOBs as low-precision, so they corroborate but do not form a sole key.
- **Owner:** Scientist (rules); Gaffer (design text; the architect approves rule changes per l.325).
- **Required test:** `test_reconciliation_identity.py::test_multiword_surname_resolved_by_dob_and_membership`, `::test_missing_dob_resolved_only_by_exact_unique_appearance_set`, `::test_partial_overlap_stays_unknown`, `::test_same_season_homonyms_resolved_by_club`.
- **Effort:** M · Impact-per-day rank 5

### S-06 — Census closure identity is implied but not stated  [MEDIUM] [class 1]
- **Location:** DESIGN §6 l.229-245; T02.
- **Evidence:**
  - Every match page links each listed player's profile (#11).
  - Directories regenerate per letter, with independent Last-Modified headers (#3, #21).
  - Per-season player lists `/afl/stats/YYYY.html` list every player by club with a profile link: 1992 shows 552 unique players, McMullin twice for his two clubs (#20).
- **Reason:** A letter page that is stale or omits a player is detectable at no extra cost. The design reconciles profile appearances with lineups but does not name this as a completeness identity.
- **Recommended outcome:**
  - Require the census closure identity: every profile URL linked from any in-scope match page is in the directory census, or is a census gap that blocks `capture_complete`.
  - Optionally capture the 130 per-season player lists as a second census, at under 0.5% extra requests. If adopted, classify their extra columns (`DA`).
- **Owner:** Scientist.
- **Required test:** `test_reconciliation_inventory.py::test_lineup_profile_not_in_directory_breaks_census_completeness`.
- **Effort:** S · Impact-per-day rank 6

### S-07 — Which source inconsistencies block PASS is ambiguous  [MEDIUM] [class 10]
- **Location:** DESIGN §8 l.397-402 ("source-internal consistency is a separate measured dimension") vs §9 l.436 (PASS requires "no … source conflict").
- **Reason:** It is unclear whether a source-internal disagreement between derived figures (printed average, season row) blocks the local-layer PASS. Given the source's own disclaimer, a strict reading makes PASS rare. A loose reading risks waving through real per-game conflicts.
- **Recommended outcome:** State it in one sentence:
  - A per-appearance cell conflict between profile and match page blocks PASS (UNKNOWN) for the affected cells.
  - A derived-figure inconsistency (S-03) is recorded under `source_consistent=false` and does not change the local-layer verdict.
- **Owner:** Gaffer.
- **Required test:** `test_reconciliation_report.py::test_profile_match_cell_conflict_blocks_pass`, `::test_derived_figure_inconsistency_reported_not_blocking`.
- **Effort:** S · Impact-per-day rank 7

### S-08 — Notes exceptions need source-specific, season-scoped name handling  [MEDIUM] [class NEW]
- **Location:** DESIGN §8 l.362-366; §7 l.338-340.
- **Evidence (#4):**
  - Notes list "1975 R14 Sydney, Hawthorn, Essendon"; Sydney did not exist in 1975 (South Melbourne).
  - "Footscray v Carlton" vs the season page's "Carlton v Footscray" (#24).
  - Notes labels `I5` and `OP` vs table labels `IF` and `1%`.
  - Notes are silent on pre-1965, and "From 2011, all categories are complete" is implicit.
- **Reason:** The design's season-valid alias rule is correct for joins, but the notes use lineage names. A naive parser will drop or mis-assign exceptions.
- **Recommended outcome:**
  - Add a versioned notes-exception map from lineage name to the season-valid club, plus order-insensitive team pairs and a notes-label map.
  - Derive pre-1965 availability from match Totals structure, since notes give no evidence.
  - Treat any unmapped exception row as a schema gap.
- **Owner:** Scientist.
- **Required test:** `test_reconciliation_source.py::test_notes_exception_lineage_name_maps_to_season_club`, `::test_notes_label_variants_I5_OP`, `::test_unmapped_notes_row_is_schema_gap`.
- **Effort:** S · Impact-per-day rank 8

### S-09 — Drop conditional requests from capture (simplification)  [MEDIUM] [over-engineering]
- **Location:** DESIGN §6 l.258, 273-275; T17.
- **Evidence:** §2 row 3. A one-shot frozen corpus gains nothing from 304s. Validators are mutable state inside the archive root.
- **Recommended outcome:**
  - Capture uses `conditional=False` and writes no `validators/` under `RUN/capture`.
  - End-of-acquisition revalidation is a full GET plus hash comparison (about 156 requests).
  - T17 becomes "an unsolicited 304 is a failure; no conditional headers are ever sent".
- **Owner:** Scientist.
- **Required test:** `test_reconciliation_capture.py::test_capture_sends_no_conditional_headers_and_writes_no_validators`, `::test_unsolicited_304_is_failure`.
- **Effort:** S · Impact-per-day rank 9

### S-10 — Existing reader silently overwrites duplicate header labels  [MEDIUM] [class 4]
- **Location:** `src/supercoach_via/integrity/sourcepages.py:518` (and the same dict pattern at `:284`, `:299`); rowspan is ignored at `:113-117`.
- **Evidence (executed):** A synthetic 28-column table with `MK` relabelled `KI` returns `problems: []`, `kicks = 11` (the second column) and `marks` absent.
- **Reason:** DESIGN T12 requires "duplicate headers → explicit schema gap". Extending the current reader without a fix would fail T12. The same reader backs the existing integrity checker. No real page showed duplicates, so this is a latent defect.
- **Recommended outcome:** The reconciliation reader, and ideally the shared tokenizer path, rejects duplicate labels and any `rowspan` inside a data table as MALFORMED.
- **Owner:** Scientist.
- **Required test:** `test_reconciliation_source.py::test_duplicate_header_label_is_malformed`, `::test_rowspan_in_game_table_is_schema_gap`.
- **Effort:** S · Impact-per-day rank 10

### S-11 — "Replay" is not a source token; define it as derived  [LOW]
- **Location:** DESIGN §2 l.101-102 ("stage/replay"); §7 l.330-332; T08.
- **Evidence:** #12–14 and #23. Two "Grand Final" fixtures have the same pair and distinct dates and URLs; the first is drawn; profile `Rd` = "GF" for both; there is no "Replay" text.
- **Recommended outcome:** A replay ordinal is derived from (season, stage, unordered pair, date order, prior drawn result) and keyed by match URL. It is never parsed from text.
- **Owner:** Scientist.
- **Required test:** `test_reconciliation_identity.py::test_drawn_gf_and_replay_distinguished_by_match_url` (2010 fixtures) and `::test_replay_without_links_is_unknown`.
- **Effort:** S · Rank 11

### S-12 — URL policy completeness and link resolution  [LOW]
- **Location:** DESIGN §6 l.247-253.
- **Recommended outcome:**
  - The reconciliation policy enumerates `^/robots\.txt$`, `^/afl/stats/stats_idx\.html$`, `^/afl/stats/players[A-Z]_idx\.html$` and `^/afl/stats/notes\.html$`, plus the optional per-season list pattern.
  - Relative hrefs (`players/V/…`, `../../games/…`, `../../1992.html#5`) are joined against the *final* URL, then the fragment is stripped, then the result is validated.
- **Owner:** Scientist.
- **Required test:** `test_reconciliation_capture.py::test_policy_paths_exact_and_fragment_stripped_after_join`.
- **Effort:** S · Rank 12

### S-13 — Do not capture `_gm.html` pages  [LOW] [over-engineering guard]
- **Evidence:** #18 duplicates the profile's match links, adds about 13.4k requests (about 7.4 h at 2 s), and profile links exist in all sampled eras.
- **Recommended outcome:** State that it is excluded from the required corpus.
- **Owner:** Gaffer.
- **Test:** none; record the decision.
- **Rank:** 13

### S-14 — Cache stack can be smaller  [LOW] [over-engineering]
- **Location:** DESIGN §10 l.494-509.
- **Reason:** An offline compare of about 1.8 GB (estimate, §5) with a 10-minute target may not need a separate comparison-result cache. A parsed-facts cache keyed by (body sha, parser/rule hash) plus full recomputation can satisfy T20/T21 if "changed-since" is implemented as a full recompute with reuse accounting.
- **Recommended outcome:** Gaffer may permit the smaller design if it measures within target.
- **Owner:** Gaffer.
- **Test:** T20/T21 unchanged.
- **Rank:** 14

### S-15 — Commit interpreter path is stale  [LOW] [class 10]
- **Evidence:**
  - `.githooks/pre-commit:29` defaults to `/home/abhi/sourceCode/python/coding/.venv/bin/python`, which does not exist (`ls`: No such file).
  - CLAUDE.md §5 run commands and `scripts/weekly_refresh.sh` use the same path.
  - `scripts/git_commit_safe.sh` does not set `COUNCIL_PYTHON`.
  - The workaround is documented only in `docs/rewrite/IMPLEMENTATION_STATUS.md:15`.
- **Impact:** Phase H's `.py` commits fail closed unless `COUNCIL_PYTHON` is exported, which tempts `--no-verify`.
- **Recommended outcome:** DESIGN §12 phase H and the README launch steps state the interpreter export explicitly.
- **Owner:** Gaffer.
- **Test:** T29 (`test_reconciliation_cli.py::test_no_machine_specific_paths`).
- **Rank:** 15

### S-16 — Output-alias helper does not inode-check against inputs  [LOW]
- **Evidence:** `integrity/runner.py:541-613`. `samefile` runs only among outputs, and the input check is path containment only. Atomic replace (`atomic_write_bytes`) means a hardlinked input inode is not mutated, so the practical risk is low.
- **Recommended outcome:** T23 asserts that the bytes of the input inode are unchanged when an output path is a hardlink to an input file outside the input roots.
- **Owner:** Scientist.
- **Test:** `test_reconciliation_report.py::test_hardlinked_output_never_mutates_input_inode`.
- **Rank:** 16

**No CRITICAL or BLOCKING findings.** No human-owner policy question blocks approval. The single
owner-level choice (S-01 option "exclude season-summary-only statistics") arises only if
Gaffer prefers exclusion over the strict default.

## 4. T01–T30 challenge table

All modules are under `tests/scvia/unit/` unless marked (I) = `tests/scvia/integration/test_reconciliation_real.py`.

| ID | Feasible as specified? | Gap / underspecification | Proposed test(s) |
|---|---|---|---|
| T01 | Yes: directory + lineup links give an independent census | None | `test_reconciliation_inventory.py::test_source_only_player_is_missing_locally_and_fails` |
| T02 | Yes for a failed letter (nav lists A–Z; H1 "All Players - A") | Needs the closure identity to detect a letter whose page silently omits a player (S-06) | `test_reconciliation_inventory.py::test_failed_letter_makes_denominator_unknown`, `::test_lineup_profile_not_in_directory_breaks_census_completeness` |
| T03 | Yes: real fixtures exist (Gary_Ablett0/1, Bill_Ahern0/1 same season, Ah Chee, De Abel) | Rule 3 depends on exact names (S-05); low-precision DOB undefined | `test_reconciliation_identity.py::test_same_season_homonyms_resolved_by_club`, `::test_multiword_surname_resolved_by_dob_and_membership`, `::test_conflicting_dob_is_identity_conflict`, `::test_no_url_constructed_from_name` |
| T04 | Yes | None | `test_reconciliation_identity.py::test_two_local_ids_one_profile_is_conflict`, `::test_alias_cycle_rejected` |
| T05 | Yes: explicit per-row match links | None | `test_reconciliation_compare.py::test_missing_appearance_detected_despite_final_counter` |
| T06 | Yes | None | `test_reconciliation_compare.py::test_offsetting_cell_swaps_detected_per_game` |
| T07 | Yes | None | `test_reconciliation_compare.py::test_duplicate_local_appearance_detected_before_aggregation` |
| T08 | Yes: distinct URLs (2010 and 1948 pairs captured) | "Replay" must be derived (S-11) | `test_reconciliation_identity.py::test_drawn_gf_and_replay_distinguished_by_match_url`, `::test_replay_without_links_is_unknown` |
| T09 | Yes: McMullin 1992 (Essendon + Collingwood rows), Sidebottom 1987 | Career GM average uses distinct seasons; there is no combined-season row, so "combined Totals rows" means only the career footer | `test_reconciliation_aggregate.py::test_two_club_season_counted_once_in_career`, `::test_career_gm_average_uses_distinct_seasons` |
| T10 | Mostly | All-zero team column encoding not observed (Totals "0" vs blank); must be UNRESOLVED until a fixture proves it. Pre-1965 is not in notes (S-08). | `test_reconciliation_cells.py::test_blank_with_team_total_is_recorded_zero`, `::test_blank_team_total_is_not_recorded` (1975 R11), `::test_pre1965_goals_only_from_match_structure`, `::test_all_zero_team_column_unresolved_without_fixture` |
| T11 | Partly | Unused-sub state missing (S-02); %P never zero-filled already stated | `test_reconciliation_cells.py::test_finals_brownlow_not_applicable`, `::test_credited_unused_sub_row_is_appearance_not_zero_stats`, `::test_arrow_tokens_preserved_raw`, `::test_blank_tog_never_zero` |
| T12 | Yes, after S-10 | Today's reader keeps the last duplicate label silently; rowspans absent in data tables, so a rowspan must be a schema gap | `test_reconciliation_source.py::test_duplicate_header_label_is_malformed`, `::test_rowspan_in_game_table_is_schema_gap`, `::test_permuted_headers_map_by_label`, `::test_alternate_year_table_without_pct_column` |
| T13 | Yes | Rounding proven HALF_UP (S-03); "comma decimal" not seen in samples, keep as negative case | `test_reconciliation_cells.py::test_strict_parse_rejects_nan_inf_bool_comma`, `test_reconciliation_aggregate.py::test_display_rounding_half_up_proven_by_ties` |
| T14 | Partly | Must exclude summary-only BR (S-01) and derived-average model misses (S-03/S-07) from SOURCE_CONFLICT | `test_reconciliation_compare.py::test_profile_vs_match_cell_conflict_is_source_conflict`, `test_reconciliation_aggregate.py::test_summary_only_brownlow_is_not_source_conflict` |
| T15 | Yes | Profile rows lack dates, so the boundary uses match/season dates | `test_reconciliation_inventory.py::test_games_after_through_date_excluded_by_match_date`, `::test_truncated_career_printed_total_marked_out_of_scope` |
| T16 | Not with today's client (S-04) | Custom 404 HTML is identical for robots and missing profiles | `test_reconciliation_capture.py::test_429_long_retry_after_persists_deadline_no_early_retry`, `::test_403_stops_host`, `::test_200_challenge_page_unusable`, `::test_404_profile_is_gap_not_empty_career`, `::test_timeout_and_oversized_are_gaps` |
| T17 | Redefine (S-09) | — | `test_reconciliation_capture.py::test_capture_sends_no_conditional_headers_and_writes_no_validators`, `::test_unsolicited_304_is_failure` |
| T18 | Yes | Add spacing persistence across resume (S-04) | `test_reconciliation_capture.py::test_resume_after_sigterm_exact_queue`, `::test_second_writer_exits_5`, `::test_corrupt_checkpoint_refused`, `::test_changed_plan_refused`, `::test_spacing_persists_across_resume` |
| T19 | Yes | — | `test_reconciliation_cli.py::test_compare_offline_socket_denied_relocated_archive_identical` |
| T20 | Yes | — | `test_reconciliation_report.py::test_outputs_identical_across_workers_shuffles_cold_warm_changed` |
| T21 | Yes | Include notes-map and identity-rule hash in keys | `test_reconciliation_cache.py::test_invalidation_source_cell`, `::test_invalidation_notes_map`, `::test_invalidation_identity_override`, `::test_invalidation_code_hash`, `::test_invalidation_local_row` |
| T22 | Yes | Documented limitation: byte-restored change unobservable | `test_reconciliation_local.py::test_input_drift_marks_incomplete_and_retains_fail` |
| T23 | Yes | Add an inode assertion (S-16) | `test_reconciliation_report.py::test_output_alias_refused_relative_symlink_hardlink`, `::test_hardlinked_output_never_mutates_input_inode` |
| T24 | Yes | — | `test_reconciliation_report.py::test_partial_write_leaves_no_completion_marker_and_exact_receipt` |
| T25 | Yes | — | `test_reconciliation_compare.py::test_historical_mismatch_fails_full_audit` |
| T26 | Yes: 33 local quarantine rows, including 3 malformed finals rows (memory) | — | `test_reconciliation_local.py::test_quarantined_row_matching_source_is_coverage_gap`; (I) `::test_real_quarantine_rows_visible` |
| T27 | Yes | Classify non-statistic columns (`GM`, `W-D-L`, `DA`, `SU`) so they do not trip "unexpected numeric column" | `test_reconciliation_source.py::test_unknown_numeric_column_blocks_schema_completeness`, `::test_known_non_statistic_columns_classified` |
| T28 | Yes: 2026 season page lists GF 2026-09-26 (`081920260926`) | Committed fixture predates the GF; needs a new trimmed fixture | `test_reconciliation_inventory.py::test_latest_final_discovered_without_local_seed` |
| T29 | Yes | S-15 | `test_reconciliation_cli.py::test_installed_cli_uses_packaged_configs`, `::test_no_machine_specific_paths` |
| T30 | Yes | — | `test_reconciliation_report.py::test_per_layer_verdicts_retained_combined_fail` |

Suggested additions, not in DESIGN: T31 census closure (S-06); T32 source-average model (S-03); T33 summary-only state (S-01); T34 credited-unused-sub (S-02); T35 spacing across resume (S-04); and (I) `test_reconciliation_real.py::test_pilot_values_traceable_to_cells`.

## 5. Scale estimate (ESTIMATES — labelled)

| Item | Count | Basis |
|---|---|---|
| Profiles | ~13,370 | Local 13,368 players. Source/local ratio in sampled letters: A 400/401, M 1,542/1,543 (measured). |
| Match pages | ~17,056 | Local matches 1897–2026 (measured). Season-page fixture counts equal local where checked: 2010 = 186 = 186; 2026 = 218 = 218. |
| Season pages | 130 | 1897–2026 |
| Index, notes, robots, directories | 29 | 26 letters + 3 |
| End-of-run revalidation | ~156 | 26 directories + 130 seasons |
| Optional per-season player lists | 130 | S-06 |
| **Total** | **~30,600–30,900 requests** | **≈17.0–17.2 h minimum at 2 s spacing**, excluding retries and 429 pauses. Agrees with DESIGN's 16.7 h capacity arithmetic. |
| Raw bytes | ~1.8 GB | Profiles ≈ 31 KB + 0.79 KB × games, a linear fit over 8 samples, giving ~0.96 GB for 695,508 local player-games. Match pages average 46.5 KB over 6 samples, giving ~0.79 GB. Local career length: median 20 games, p90 154. |

## 6. Anti-pattern list (standing, with additions)

- Never trust an LLM sum; re-measure disputed numbers in pandas before acting.
- Never verify by exit code; re-read the file content after any write.
- Never `git add .`; stage by explicit allowlist.
- Never hand-edit anything under `data/` or a generated table body.
- Never let a Pass-1 PASS stand in for Pass-2 clearance.
- Never soften an upstream caveat when translating numbers into prose.
- Never run the refresh before round settlement.
- Never push to `main` from parallel agents; serialize through one committer.
- Never define an agent's role, model or tool scope in more than one place.
- Never leave a gate-enforced convention unwritten.
- **NEW:** Never treat a source's *derived* figure (printed average, season-only summary) as a peer of its per-game cells. Model its convention explicitly, or keep it in a separate source-consistency dimension (evidence: S-01, S-03).
- **NEW:** Never let a transport layer's internal retry decide politeness. The component that persists the schedule must own every retry and see the server's deadline (evidence: S-04).

## 7. Watch list (speculation, unranked)

- *URL instability on new homonyms.* All-suffixed groups and a 404 for `Gary_Ablett.html` suggest a base URL is renamed when a namesake debuts. No drift was found among 121 local A/M `source_urls`; the only miss was El Achkar, filed under E at source. Worth a pilot check: a stored URL that 404s should be investigated through the directory, never re-guessed.
- *1992 population.* Source list: 552 unique players. Local: 553 distinct player_ids, 555 player-club pairs. Possibly one local duplicate identity or parsing noise; a pilot case for Scientist, not a claim.
- *Local-only names in A/M:* Fred Abel (resolved: source "De Abel"), Joe Medici (1941–44) and Maurie Araugo (1924–28) need the Phase E pilot.
- *2011–2015 substitute era:* whether unused "green vest" subs were credited games is untested (S-02 scope).
- *Legacy harness:* `scripts/weekly_refresh.sh` hard-codes the missing venv path. This is outside this design, but the next weekly cycle will fail closed on it.
- *Courtesy:* robots.txt is absent (404). A roughly 17 h crawl of a volunteer site is permitted by design; consider whether the operator wants to notify the site owner. Not a gate.

## 8. Inventory — files and samples inspected

**Design package (sha256):**
- `DESIGN.md` d368d9d32d3da4ad045f9f4267946fc0191fa1621ff6dab094bd445d6aa2c2af
- `ARCHITECT_PROMPT.md` 2135e6c30e72c92ac713e7866290695a4e99669ff1ceb434e56adccfb234f030
- `ENGINEER_PROMPT.md` c013915cc8ddcde0893542cbd8273b022bc8857325f06a087ed4167600bc7d41
- `README.md` 12feee656c1a8bff93e4c8548a615ff215b6eaedfc5788285042a22064afb991

**Agent definitions (sha256):**
- Surveyor 7cfbf4df8b6a09a57fff70cd37d5acd34c80fd651f06533b29fd5cfee73a0cab (frontmatter `model: fable`; overridden for this task to claude-opus-5-5)
- Gaffer e86766506f4ed4338ff3539bcfd6165ad578a5317aee3caf153c3a844ea641ce
- Scientist d387ada030d0d74e8e3d30ed69ea060d26f8add3d65296b5e48731f2042ebedc
- QA f76d24c61bb40782a1890c329643cecf18678d89e6745f7631c79b075d4897df
- DataSentinel 87a7b0df3a473b92753338d1f87b72dfc18668512c5566ec78eb8cc784c0b6a5

**Code, tests, config and docs (sha256, first 16 hex):**
- `scrapers/game_scraper.py` 0b7f2d8294ba9e0e
- `integrity/sourcepages.py` ac603c242ccb27a7
- `integrity/checks_source.py` f3ed829c2a4eb9c6
- `ingest/http.py` 58141a151c50d948
- `storage/snapshots.py` cc5ce33a76e0de32
- `storage/queries.py` 60a99f6f63488664
- `integrity/capture.py` 40285a7ade7bb584
- `integrity/runner.py` 259599c3b1f6be99
- `integrity/report.py` a4a3d038b5cab518
- `integrity/cache.py` 4ad5d89e74b010e0
- `domain/schemas.py` c93a69e1ac4dc2ff (`PLAYER_STAT_COLUMNS` has **23** entries, executed)
- `domain/blanks.py` 828a74a26b10f282
- `cli.py` cce0a25b9121be0f (Typer `app`; `EXIT` map 0/2/3/4/5/6/7/8/9 matches DESIGN exits; no `add_typer` group exists yet; a Typer sub-app is the natural registration)
- `config/source_policies.toml` 4f390dab331ff659 (packaged copy identical)
- `tests/scvia/fixtures/zero_semantics/annotations.json` 1a307bd7ae8049d1
- `tests/scvia/unit/test_http.py` 68b3c038a5aed54e
- `tests/scvia/unit/test_integrity_sourcepages.py` 5d055dde0919c57e
- `tests/scvia/unit/test_integrity_source.py` 3d77e49a6205167e
- `docs/data-integrity.md` bad837d611427b09
- `docs/reviews/CLAUDE_OPUS55_REVIEW.md` 5cf79f41454495b1
- `pyproject.toml` 5a0d96b259537118 (groups `dev` and `legacy`, extra `ml`: present)
- `uv.lock` d28243bbfc643bef
- `.githooks/pre-commit` 677d8b7869e4446e
- `CLAUDE.md` 56eff3294c5ab200

**Archived fixtures:**
- `tests/scvia/fixtures/raw/afltables/game_2026_091020260406…` e1675c9dc8b2bb6c
- `…game_2026_131820260329…` 08a2bf5968e7ebce
- `…seas_2026_captured_20260925.html` b02240bfd1de7051
- `docs/rewrite/evidence/b1/raw/*.html.gz` (6 objects, content-addressed; 3 modern profiles, 2 matches, 1 season)

No pre-2026, directory, notes, homonym or replay fixtures existed before this survey.

**Scientist memories consulted:**
- `reconciliation_source_afltables_player.md` 32dab361ca7c129d (its career-total-only advice is superseded by per-game requirements, as DESIGN §1 says)
- `afltables_player_profile_url.md` ecfea3256ca1f214 (first-name-initial rule confirmed in 400/400 A links; name-constructed URLs fail for suffixed homonyms)
- `blank_counting_stat_means.md` 49a958a8bde5fb85
- `data_stat_coverage_eras.md` e0e6c79b56a21c2d (BR 1931–34 fragment confirmed on source)
- `rewrite_legacy_import_facts.md` dd4e31f1fde33bec
- also read: `afl_data_source_urls.md`, `dob_source_equivalence.md`, `hof_games_counter_gotcha.md`, `same_team_false_duplicate.md`

**§11 paths and IDs (verified):**
- `var/reviews/opus55/20260929T203757Z-followup/candidate-data` → `current.json` snapshot `sha256:3de6597513b5c6eaca1b102d304a2cf128d8c08f229ee1de384c1d40fe746db0`; `load_snapshot(..., verify=True)` passes, 21 tables.
- `var/finalized/data` → `sha256:aa836549e10e96a9039cc4642bbe563b94247e0bfaf95e745a0971cc0bb2b98f`; verify passes, 21 tables. The candidate's snapshots dir also contains the aa836549 manifest.
- Release dir `…/releases/releases/20260929T213900Z-c8938f4ddb83` exists (`checksums.json`, `seal.json`, `validation.json`, `public/`, `site/`).
- `scvia` resolves to `/home/abhi/git/SuperCoach-VIA/.venv/bin/scvia` (`[project.scripts] scvia = "supercoach_via.cli:main"`).
- `reconcile-afltables` does not exist yet, as the design states.

**Local measurements (candidate snapshot, executed via DuckDB):**
- 13,368 players; 695,508 player_games; 17,056 matches; 130 seasons; max match date 2026-09-26.
- 782 players with `source_urls`; 13,363 with DOB; 18 aliases; 33 quarantine rows.
- 249 multi-club player-seasons.
- Arrow tokens: ↓ 3,780, ↑ 3,776.
- `time_on_ground_pct`: 212,210 non-null, 0 non-integer.

**Live samples:** fetched with User-Agent "SuperCoach-VIA design-review (careful, low-rate)", `curl --max-time 30`, at least 2.5 s between starts. Bodies are at `/tmp/claude-1000/-home-abhi-git-SuperCoach-VIA/74120495-fb82-4a6f-8eff-a9b3f613b8e2/scratchpad/surveyor-samples/` (log `LOG.tsv`).

| # | UTC | Status | Bytes | sha256 | URL |
|---|---|---|---|---|---|
| 1 | 2026-10-01T10:07:57Z | 404 | 651 | 2628a3f2ff134bd7ccc26a2cf4fdfca4388fb8abbe27b9e5bce6de91475f27e5 | https://afltables.com/robots.txt |
| 2 | 10:08:04Z | 200 | 9142 | 84ffcc6df8ca760499884988be94884cd2b282aab98307ad7dd7c8820ed298ff | …/afl/stats/stats_idx.html |
| 3 | 10:08:06Z | 200 | 44113 | 16c16f958655ba890840fe2ca61e66ea410051aa357ed53f51fe7551c57edc9e | …/afl/stats/playersA_idx.html |
| 4 | 10:08:09Z | 200 | 34684 | b5e546e73baca0d51fdea76ac8e816ccc7696e03428c24e1e5caac409ac858ac | …/afl/stats/notes.html |
| 5 | 10:09:14Z | 200 | 399923 | e0bc5998d25e3659d75121f3d230cf0e38b7309f16415440eeb9d0b5e8c4eb39 | …/players/S/Scott_Pendlebury.html |
| 6 | 10:09:17Z | 200 | 240102 | 422f66aba68963290275f80795de26e496fe3d213b47e5f82e59922ccc7c5d3f | …/players/D/Dick_Reynolds.html |
| 7 | 10:09:19Z | 200 | 223661 | 00baafd9718dfd9d537bcdda309dba216d7e33a1fec15c47610b934705c3b165 | …/players/G/Gary_Ablett0.html |
| 8 | 10:09:22Z | 200 | 87781 | 369ed248dadc0b10fe897d6427aaf5c4af835c7d7c3e2330dbda8562c141f6cb | …/players/I/Ian_McMullin.html |
| 9 | 10:09:24Z | 200 | 210841 | 24d02e8109aeb976f6a2c58c78716c16428e2b07e504e5b5bf8791a5714020b1 | …/players/A/Alex_Sexton.html |
| 10 | 10:09:27Z | 200 | 26227 | fad24ff18c6301aa9fe3160816f136225598f1b6f0cdd9c537d7e433f01e54af | …/players/B/Bill_Ahern0.html |
| 11 | 10:10:54Z | 200 | 55516 | 55322c686011dca0b24cd5f6edd9cebf2209d6791de2ff68b4b5f560c487fc15 | …/stats/games/2021/162020210424.html |
| 12 | 10:10:57Z | 200 | 54356 | 589ecb77931c474aa979c4b3447e57b8930ae7ec61c5d488029354f7e6b486f2 | …/stats/games/2010/041520100925.html |
| 13 | 10:10:59Z | 200 | 54913 | bed9d42692257dc2179d81483de9db0f49678e331a9ad92f27fa7ee5895a3345 | …/stats/games/2010/041520101002.html |
| 14 | 10:11:02Z | 200 | 195948 | fc027577c797e25dccea45dc54966a5d6ef9aa52bac6d9d86b05916277f5252c | …/afl/seas/2010.html |
| 15 | 10:11:04Z | 200 | 37724 | 4bfa8dfac27ebbcc83be3483e0876e04e595d83a0c00f8cf98631ebca6a25b75 | …/stats/games/1897/041518970508.html |
| 16 | 10:13:09Z | 200 | 59886 | bafa678e8f2388e2d83377fd584fcc58c92d8a1a827140f282c0381145a50e4b | …/players/K/Kelly_Robinson.html |
| 17 | 10:13:12Z | 200 | 83049 | 8a43aafda7cdd1e808d12fa19b14bb2de27a91a0b3759899016a298d02831325 | …/players/A/Allan_Sidebottom.html |
| 18 | 10:13:14Z | 200 | 189546 | 0887ab6cd9b38e3f77306bdac8b6d7d7794a7dbbb3c3cfe344b133b2ec59905c | …/players/S/Scott_Pendlebury_gm.html |
| 19 | 10:13:17Z | 404 | 651 | 2628a3f2ff134bd7ccc26a2cf4fdfca4388fb8abbe27b9e5bce6de91475f27e5 | …/players/G/Gary_Ablett.html |
| 20 | 10:13:51Z | 200 | 455794 | 57af8eba25e830e7ae834876c1887fb4cf0c464f1f1b1cbaf6ae1d2f8f260aa0 | …/afl/stats/1992.html |
| 21 | 10:14:20Z | 200 | 159680 | b4e2d2a48d74eff870883b8a89cc0473c8617f9830e5a34ea94ff3eb69452f79 | …/afl/stats/playersM_idx.html |
| 22 | 10:15:08Z | 200 | 240511 | 87c99c1a2eb84c4957759e5ec82e02aec97d7daf5264e1ba6db800cdd59bc83a | …/afl/seas/2026.html |
| 23 | 10:15:10Z | 200 | 129135 | 6789b6972c2db43d07e38b9f965791ccf910503cbf160302d4e93631634b2ad3 | …/afl/seas/1948.html |
| 24 | 10:15:13Z | 200 | 148121 | d65e460f038d5461dcda192b6dc9e4ca4621644858bec076cba4913791360eab | …/afl/seas/1975.html |
| 25 | 10:15:29Z | 200 | 38439 | 7cc6c42e458fc8838c9cd3b8c99109e3459ce7522d474a1aef56a23064355290 | …/stats/games/1975/030719750616.html |
| 26 | 10:15:31Z | 200 | 38183 | 449e5123e79108435b15036e50880b4d9b10cabbe649e076e183a62ed5f17682 | …/stats/games/1948/051119481009.html |
| 27 | 10:18:51Z | 200 | 351305 | e86ceeb53a512f3261b12f0cea7ca88ce3468e046646fd00eeff3b0432a064d2 | …/afl/stats/1934.html |

Coverage against the brief:
- early career: #6, #10, #16
- modern career: #5, #9
- identical names / numeric suffix: #3, #7, #10, #19, #21
- club transfer: #8 (1992), #17 (1987)
- substitution markers: #5, #9, #11
- unavailable historical statistics: #4, #15, #25, #26
- drawn/replayed final: #12, #13, #14, #23, #26
- season page: #14, #22–24
- match page: #11–13, #15, #25, #26
- `stats_idx`: #2
- `playersA_idx`: #3
- `notes.html`: #4

No 403, 429 or challenge page was encountered.

## 9. What was checked and found clean

- The DESIGN §3 characterisations of `audit_player_career_totals`, `check_player_pages` and the existing reader's limits are accurate today.
- The 23-column statistic contract matches the code. Live profile and match headers use exactly those labels plus meta columns. Notes use `I5`/`OP` (S-08).
- The 28-column profile game table holds in every sampled era. There are no rowspans in profile or season tables. Match-page rowspans occur only in navigation cells.
- Explicit match links exist on every sampled profile row (1,370/1,370), including 1897.
- The season-page fixture count equals the local count for 2010 and 2026. The 2026 GF is present (`081920260926`), so D14 is checkable.
- §11 snapshot IDs, paths, release directory, uv groups/extra and `scvia` entry point all exist and verify.
- Local `%P` values are all integers, and sampled source %P values are integers, so the Decimal comparison in DESIGN §8 is feasible without precision loss.
- robots.txt is a genuine 404 (no inference about its contents needed).
- No active weekly cycle. CLAUDE.md §6.1 freeze does not apply to the design review. §6.2 applies to engineering only if shared transport behaviour changes (S-04).

## 10. Routing summary

| ID | Severity | Owner |
|---|---|---|
| S-01 | HIGH | Gaffer (design) + Scientist |
| S-02 | HIGH | Scientist + Gaffer |
| S-03 | HIGH | Gaffer + Scientist |
| S-04 | HIGH | Scientist |
| S-05 | HIGH | Scientist + Gaffer |
| S-06 | MEDIUM | Scientist |
| S-07 | MEDIUM | Gaffer |
| S-08 | MEDIUM | Scientist |
| S-09 | MEDIUM | Scientist |
| S-10 | MEDIUM | Scientist |
| S-11 | LOW | Scientist |
| S-12 | LOW | Scientist |
| S-13 | LOW | Gaffer |
| S-14 | LOW | Gaffer |
| S-15 | LOW | Gaffer |
| S-16 | LOW | Scientist |

Escalation to the human: **none required.** No gate is defective, no two prompts conflict over one
artifact, and no published number is implicated. The only owner-level choice is optional (S-01
exclusion variant) and does not block approval under the strict default.

**Surveyor recommends: APPROVE_WITH_CHANGES** (0 BLOCKING; fold S-01 to S-05 into DESIGN.md and bind
approval to the amended hash before the Phase B launch).
