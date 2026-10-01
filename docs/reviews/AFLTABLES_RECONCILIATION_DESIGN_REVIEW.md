# AFL Tables reconciliation: architecture review

**Mode:** design review (`docs/rewrite/afltables-reconciliation/ARCHITECT_PROMPT.md`)
**Decision:** **APPROVED**
**Date:** 2026-10-01
**Blocking findings:** none (`blocking_findings: []`)
**Approved design:** `docs/rewrite/afltables-reconciliation/DESIGN.md`, SHA-256 `da64c4cf5974d9ae4e5a638ef218de247a7dc42508262145778b2e802150797f`
**Machine-readable record:** [`afltables-reconciliation-design-review.json`](afltables-reconciliation-design-review.json)

This approval is a model review of an engineering design. It is not the owner's consent to
modify statistics or activate anything in production, and it makes no claim about whether
the repository's data agrees with AFL Tables. Nothing has been implemented or audited yet.

## 1 Who reviewed, on what

| Role | Agent definition (SHA-256) | Requested model | Resolved model | Ran? |
|---|---|---|---|---|
| Gaffer (coordinator, design edits, this record) | `.claude/agents/Gaffer.md` `e86766506f4ed4338ff3539bcfd6165ad578a5317aee3caf153c3a844ea641ce` | Opus 5.5 `claude-opus-5-5` | `claude-opus-5-5`, from this session's metadata | Yes |
| Surveyor run 1 (deep survey) | `.claude/agents/Surveyor.md` `7cfbf4df8b6a09a57fff70cd37d5acd34c80fd651f06533b29fd5cfee73a0cab` (stored default `fable`) | Opus 5.5, dispatched with the Agent tool's `opus` model override | `claude-opus-5-5`, self-reported from the subagent's session metadata. The dispatch tool does not return a resolved model ID. | Yes |
| Surveyor run 2 (confirmation of the amended design) | same | same | `claude-opus-5-5`, self-reported | Yes |
| Surveyor run 3 (confirmation of the N-findings fixes) | same | same | `claude-opus-5-5`, self-reported | Yes |
| Scientist | `.claude/agents/Scientist.md` `d387ada030d0d74e8e3d30ed69ea060d26f8add3d65296b5e48731f2042ebedc` (stored default `opus`) | Sonnet 5.5 `claude-sonnet-5-5`, for implementation | — | **No.** Not consulted in this review; see §6 |
| QA | `.claude/agents/QA.md` `f76d24c61bb40782a1890c329643cecf18678d89e6745f7631c79b075d4897df` | — | — | No; acceptance stage |
| DataSentinel | `.claude/agents/DataSentinel.md` `87a7b0df3a473b92753338d1f87b72dfc18668512c5566ec78eb8cc784c0b6a5` | — | — | No; no tagged statistical document produced |

The Agent dispatch tool was available in this session and listed Surveyor, Scientist, QA
and DataSentinel; the availability check followed `project_no_agent_dispatch_tool.md`.
No agent definition was edited. Model choices were passed per invocation.

**Repository state at review:** `main` @ `fec9cd3d87cc7c8c823176a75f33dfe90246eeeb`, clean
working tree at start. `.claude/audit/last_refresh_status.json` =
`{"phase":"4","exit_code":0,"round":"25","ts":"2026-08-29T13:49:38+1000"}`, so no weekly
cycle was active (CLAUDE.md §6.1). Claude Code `2.1.286`; `--agent`, `--model` and
`--effort` are present in its help. Unrelated pre-existing state was preserved.

## 2 Evidence

### Surveyor reports (exact preserved copies)

| Run | Live path | Preserved copy | SHA-256 |
|---|---|---|---|
| 1 | `.claude/surveys/2026-10-01-afltables-reconciliation-design-survey.md` | [`afltables-reconciliation/surveyor-design-review-2026-10-01-run1.md`](afltables-reconciliation/surveyor-design-review-2026-10-01-run1.md) | `0e19d9673e371295d15bfc8ba2205f08a650b65757d1fa43c60744ff4ba76311` |
| 2 | `.claude/surveys/2026-10-01-afltables-reconciliation-design-survey-run2.md` | [`afltables-reconciliation/surveyor-design-review-2026-10-01-run2.md`](afltables-reconciliation/surveyor-design-review-2026-10-01-run2.md) | `a0262a3172b82710399b371d365f4bd5816da4c95aa0c31bdcecf07559f7e9d9` |
| 3 | `.claude/surveys/2026-10-01-afltables-reconciliation-design-survey-run3.md` | [`afltables-reconciliation/surveyor-design-review-2026-10-01-run3.md`](afltables-reconciliation/surveyor-design-review-2026-10-01-run3.md) | `2d57408d9f4d65bdf98f5b0c4bde6651e9941e4b986ef28d135bc34d0852934f` |

Intermediate design hashes: `2e69b86d6d1abb43fc599a5adc7ef8edbddb66539c6b12390b04065d27d59ff0` (after the S-findings, reviewed by run 2) and `da64c4cf5974d9ae4e5a638ef218de247a7dc42508262145778b2e802150797f` (after the N-findings, reviewed by run 3).

### Design package hashes

| File | Reviewed (before edits) | Final |
|---|---|---|
| `DESIGN.md` | `d368d9d32d3da4ad045f9f4267946fc0191fa1621ff6dab094bd445d6aa2c2af` | `da64c4cf5974d9ae4e5a638ef218de247a7dc42508262145778b2e802150797f` |
| `ARCHITECT_PROMPT.md` (role payload) | `2135e6c30e72c92ac713e7866290695a4e99669ff1ceb434e56adccfb234f030` | unchanged |
| `ENGINEER_PROMPT.md` (role payload) | `c013915cc8ddcde0893542cbd8273b022bc8857325f06a087ed4167600bc7d41` | unchanged |
| `README.md` (launch guide) | `12feee656c1a8bff93e4c8548a615ff215b6eaedfc5788285042a22064afb991` | unchanged |

### Code paths inspected (SHA-256 at `fec9cd3d8`)

| Path | SHA-256 |
|---|---|
| `scrapers/game_scraper.py` | `0b7f2d8294ba9e0e42b438d0a435798e1532357c4cbd00871791473317d56c29` |
| `src/supercoach_via/integrity/sourcepages.py` | `ac603c242ccb27a7c21109630924cebe048185ee53552dc1a3bdba9038ef35e6` |
| `src/supercoach_via/integrity/checks_source.py` | `f3ed829c2a4eb9c6eddcafa8431bdf7eb24b86e903bc8e73733b9ecc91ca8adc` |
| `src/supercoach_via/ingest/http.py` | `58141a151c50d9481d9e3968a68376268a3033668ff70191318274f818936cb2` |
| `src/supercoach_via/storage/snapshots.py` | `cc5ce33a76e0de32d0b3698ddd4b70d0656993471a9bf4044a3080d2b3878782` |
| `src/supercoach_via/storage/queries.py` | `60a99f6f6348866486c5360816a27dce7ba18edf5e3192d4bb23aa74ed08a3c7` |
| `src/supercoach_via/integrity/capture.py` | `40285a7ade7bb5842dc94a43c4db21175475da35df586a7ca2b0a397d106a349` |
| `src/supercoach_via/integrity/runner.py` | `259599c3b1f6be994ed0c3947455813b7b94b37998a2472796a16452ea5a2e67` |
| `src/supercoach_via/integrity/report.py` | `a4a3d038b5cab5186932355af3daf0ea0f844150631912caa8def89fc99de8f1` |
| `src/supercoach_via/integrity/cache.py` | `4ad5d89e74b010e02af4658a8359d87ca3ba2529a0385e2d78ced8954bda3eea` |
| `src/supercoach_via/domain/schemas.py` | `c93a69e1ac4dc2ff0816a9e614ea1657adcad41c900faed50916297a23b36e6a` |
| `src/supercoach_via/domain/blanks.py` | `828a74a26b10f282265c8715ee14c215f965e7cea899991f9a1824ed2a9c7c01` |
| `src/supercoach_via/cli.py` | `cce0a25b9121be0f33c473fc3d58753e135a0b5e51d865857cb99ac2db3b52aa` |
| `config/source_policies.toml` (packaged copy identical) | `4f390dab331ff6595cd6d6fbfcfdaf31fce62623ed7e562886bf28f5c9300b45` |
| `tests/scvia/unit/test_http.py` | `68b3c038a5aed54ebecf8b78bf1ae84dcc898ee3e0f2b3c12e02b0498c39214e` |
| `tests/scvia/unit/test_integrity_sourcepages.py` | `5d055dde0919c57e0b49ea2963e5975e3cb15fc98390384c2f3bbbfe77b6bab9` |
| `tests/scvia/unit/test_integrity_source.py` | `3d77e49a6205167e728d7d10d5b877d613c9047f5f11f23fa6c179cce1b88799` |
| `tests/scvia/fixtures/zero_semantics/annotations.json` | `1a307bd7ae8049d18df494b20db1a2b81bc8fbee6316e37dce0efd33b0addeae` |
| `pyproject.toml` | `5a0d96b2595371181d83aa2a725466e03a8197801d66bdb3d9512abf8fade452` |
| `uv.lock` | `d28243bbfc643befbf60a05c9c8a44431f73303ee3802a41e0d16a634555f873` |
| `.githooks/pre-commit` | `677d8b7869e4446e8cfb33a92cbb28868919367e4093d6a2742e280382fe57fc` |
| `scripts/git_commit_safe.sh` | `99f808dcc16d951ef95e058b6c0e3b3a741708c0e42ca6c1c52bfe590261dfe6` |
| `CLAUDE.md` | `56eff3294c5ab200fc8b86ea639402c7fe30bf5d86d531b82e39293a3484bae2` |

Also read: `docs/data-integrity.md` (`bad837d6…`) and `docs/reviews/CLAUDE_OPUS55_REVIEW.md`
(`5cf79f41…`). The design §11 inputs were verified by Surveyor with `load_snapshot(..., verify=True)`:
candidate `sha256:3de65975…` at `var/reviews/opus55/20260929T203757Z-followup/candidate-data`,
control `sha256:aa836549…` at `var/finalized/data`, and the release directory
`…/releases/releases/20260929T213900Z-c8938f4ddb83`. All exist. Gaffer re-loaded the
candidate snapshot independently.

### Memories consulted

- Gaffer: `.claude/agent-memory/Gaffer/MEMORY.md` (`c8357fe3…`), `project_no_agent_dispatch_tool.md` (`5f67fe2a…`), `feedback_harness_change_discipline.md` (`0706675e…`), `feedback_consult_surveyor.md` (`bb769c21…`).
- Scientist: `reconciliation_source_afltables_player.md` (`32dab361…`); its career-total-only approach is superseded by the per-game requirement. `afltables_player_profile_url.md` (`ecfea325…`); name-constructed URLs fail for suffixed homonyms. `blank_counting_stat_means.md` (`49a958a8…`); its "blank means zero in the modern era" convention is **not** adopted, and the design requires per-match evidence instead. `hof_games_counter_gotcha.md` (`47456378…`); counter-based game counts are not used.
- Surveyor (its own reads): additionally `data_stat_coverage_eras.md`, `rewrite_legacy_import_facts.md`, `afl_data_source_urls.md`, `dob_source_equivalence.md`, `same_team_false_duplicate.md`.

### Source samples

Surveyor run 1 made 27 live GET requests to `afltables.com` with a descriptive User-Agent,
at least 2.5 s between request starts. No 403, 429 or challenge page appeared. The bodies,
headers, `LOG.tsv` and `SHA256SUMS` are retained locally, outside Git, at
`var/reconciliations/afltables/design-review-samples-20261001/`. Gaffer re-hashed the copies
and they match the survey. The full URL/status/hash table is in survey §8 and in the JSON
record. Coverage:

| Case | Samples |
|---|---|
| Early career | Dick Reynolds (1933–51), Bill Ahern0 (1897), Kelly Robinson (1897–1901, no DOB) |
| Modern career | Scott Pendlebury, Alex Sexton |
| Identical names / numeric suffix | `playersA_idx`, `playersM_idx`, Gary Ablett0, Bill Ahern0, unsuffixed `Gary_Ablett.html` (404) |
| Club transfer | Ian McMullin (1992, two clubs), Allan Sidebottom (1987) |
| Substitution markers | Pendlebury, Sexton, 2021 match `162020210424` |
| Unavailable historical statistics | `notes.html`, matches from 1897, 1975 and 1948 |
| Drawn/replayed final | 2010 `041520100925` / `041520101002`, `seas/2010`, `seas/1948`, 1948 replay `051119481009` |
| Season / match / index | `seas/1948`, `seas/1975`, `seas/2010`, `seas/2026`; six match pages; `stats_idx`; per-season lists 1934 and 1992 |
| robots.txt | genuine HTTP 404 (651-byte custom HTML) |

### Gaffer's independent checks of Surveyor's claims

- **S-10 confirmed:** `sourcepages.py` builds the stat-cell dict from `zip(labels[5:], cells[5:])`, so a duplicated label silently keeps the last value.
- **S-15 confirmed:** the `.githooks/pre-commit` default interpreter `/home/abhi/sourceCode/python/coding/.venv/bin/python` does not exist; `.venv/bin/python` does.
- **S-02 scale confirmed:** a DuckDB count over the candidate snapshot's `player_games` found rows with null `kicks`, `handballs` and `time_on_ground_pct`: 2021: 202, 2022: 183, and 5 each in 2023, 2024 and 2025. This matches the survey.
- **S-01 confirmed, with one correction (G-01):** Reynolds' 1935 season row prints BR with blank per-game cells. However, the career "Totals" row of the by-totals table **does** print a BR value; the survey said it was blank. The finding and its resolution are unchanged, and the design text states the corrected fact. Surveyor run 2 re-parsed the page and agreed. Its run-1 parser had ignored the footer label's `colspan=3`, shifting the footer columns. Reynolds' per-game BR (1933–34) plus his season-only BR (1935–50) sum exactly to the printed career total. No other run-1 measurement depended on that parse.
- **S-05 confirmed:** Kelly Robinson's profile has no `Born:` line.
- **Transport claims confirmed:** `retry_after_max_s = 60.0`; `_backoff` uses `min(retry_after, cap)`; `RawArchive` writes `validators/<sha(url)>.json`; the policy has only season, match and player path grammars; defaults are `requests_per_second = 2.0` and `max_concurrent_per_host = 2`.

## 3 Findings and dispositions

Severity and owner come from Surveyor. Disposition is Gaffer's resolution in the final design.

| ID | Sev | Location | Reason | Disposition (final DESIGN §) | Required test |
|---|---|---|---|---|---|
| S-01 | HIGH | DESIGN §8 state table, aggregates, §9 | Pre-1984 Brownlow exists only as season summaries; as originally written every vote-getter became a SOURCE_CONFLICT, so PASS was unreachable by misclassification | **Resolved.** New `SOURCE_SUMMARY_ONLY` availability with an evidence rule. It is never a SOURCE_CONFLICT. Kept **in scope**; local absence is `LOCAL_MISSING_SUMMARY_VALUE`, which makes the layer FAIL. Excluding it is reserved to the owner (§8) | T33: `test_reconciliation_aggregate.py::test_summary_only_brownlow_is_not_source_conflict`, `::test_summary_only_statistic_counted_separately`, `::test_local_missing_summary_value_fails_layer` |
| S-02 | HIGH | §8 states, §7 roles, T11 | Unused 2021–22 medical subs are credited games with all-blank rows; there was no state for them, so they would have been fabricated zeros or a permanent UNKNOWN (~400 local rows) | **Resolved.** `CREDITED_DID_NOT_TAKE_FIELD` participation state with a four-condition rule; cells take `NOT_APPLICABLE_DNTF`; local null or zero agrees with a representation note; a positive local value is a mismatch. Whether 2011–15 substitutes follow the same convention is deferred to the Phase E pilot (§8) | T34: `test_reconciliation_cells.py::test_credited_unused_sub_row_is_appearance_not_zero_stats`, `::test_used_sub_arrow_tokens_preserved_raw` |
| S-03 | HIGH | §8 aggregation step 5 | "Verify the denominator exactly" was impossible because the source does not print denominators | **Resolved.** The measured source-average model is explicit (ROUND_HALF_UP; GM; BR over home-and-away games; career era exclusion; GM average over distinct seasons; W-D-L as a percentage). A miss is recorded with all candidate denominators considered (§8) | T32: `test_reconciliation_aggregate.py::test_display_rounding_half_up_proven_by_ties`, `::test_brownlow_average_uses_home_and_away_denominator`, `::test_career_average_excludes_unrecorded_era_games`, `::test_average_model_miss_is_source_consistency_not_local_fail` |
| S-04 | HIGH | §6 network; `ingest/http.py:501-503`, `632-703` | The client retries internally after at most 60 s, hides `Retry-After`, has in-memory spacing only, and defaults to 0.5 s spacing with concurrency 2 | **Resolved.** Reconciliation policy pins 0.5 rps, concurrency 1 and client `max_attempts = 1`; the coordinator owns retries with a persisted next-eligible timestamp; the fetch result gains an additive `Retry-After` field; default production behavior is proven byte-identical; Scientist records the CLAUDE.md §6.2 scope (§6) | T16, T35: `test_reconciliation_capture.py::test_429_long_retry_after_persists_deadline_no_early_retry`, `::test_spacing_two_seconds_includes_retries_redirects_robots`, `::test_spacing_persists_across_resume`; `test_http.py::test_retry_after_field_is_additive` |
| S-05 | HIGH | §7 identity rules 3–4 | Exact-name rule fails on split surnames (Ah Chee, De Abel, El Achkar); some profiles have no DOB; only 782 of 13,368 local players carry a source URL | **Resolved.** Rule 3b: unique full DOB plus an identical (season, club) set. Rule 4: a globally unique exact appearance-set equality when DOB is missing or low precision. A 1 January DOB is low precision. Set equality only; partial overlap stays UNKNOWN (§7) | T03: `test_reconciliation_identity.py::test_multiword_surname_resolved_by_dob_and_membership`, `::test_missing_dob_resolved_only_by_exact_unique_appearance_set`, `::test_partial_overlap_stays_unknown`, `::test_same_season_homonyms_resolved_by_club` |
| S-06 | MED | §6 discovery | Census closure was implied but not stated | **Resolved.** Closure identity added: a lineup-linked profile missing from the directory blocks `capture_complete`; the per-season lists are an optional second census (§6 step 6) | T02, T31: `test_reconciliation_inventory.py::test_lineup_profile_not_in_directory_breaks_census_completeness` |
| S-07 | MED | §8 vs §9 PASS row | It was ambiguous which source inconsistencies block PASS | **Resolved.** A per-appearance evidence disagreement is a `SOURCE_CONFLICT` and blocks PASS. A printed-figure disagreement is a `SOURCE_DERIVED_INCONSISTENCY`: `source_consistent=false`, and it does not block (§8, §9 table) | `test_reconciliation_report.py::test_profile_match_cell_conflict_blocks_pass`, `::test_derived_figure_inconsistency_reported_not_blocking` |
| S-08 | MED | §8 availability | Notes use lineage club names, pairs in either order, and `I5`/`OP` labels | **Resolved.** Versioned notes-exception map, order-insensitive pairs, label map; an unmapped row is a schema gap; maps are hashed into the plan and caches; pre-1965 availability comes from match structure (§8) | T10: `test_reconciliation_source.py::test_notes_exception_lineage_name_maps_to_season_club`, `::test_notes_label_variants_I5_OP`, `::test_unmapped_notes_row_is_schema_gap` |
| S-09 | MED | §6, T17 | Conditional requests add mutable state for no benefit to a one-shot corpus | **Resolved (simplification).** `conditional=False`, no validators written, revalidation by full GET and hash, an unsolicited 304 is a failure; T17 redefined (§6, §13) | T17: `test_reconciliation_capture.py::test_capture_sends_no_conditional_headers_and_writes_no_validators`, `::test_unsolicited_304_is_failure` |
| S-10 | MED | `sourcepages.py` stat-cell dict; T12 | A duplicated header label silently overwrites | **Resolved.** Duplicate labels and data-table rowspans are `MALFORMED` (§3, §8) | T12: `test_reconciliation_source.py::test_duplicate_header_label_is_malformed`, `::test_rowspan_in_game_table_is_schema_gap` |
| S-11 | LOW | §7 | No "Replay" token exists in the source | **Resolved.** Replay ordinal derived; appearance keyed by match URL (§7) | T08: `test_reconciliation_identity.py::test_drawn_gf_and_replay_distinguished_by_match_url`, `::test_replay_without_links_is_unknown` |
| S-12 | LOW | §6 URL policy | Policy omitted robots/index/notes paths; link-resolution order was unstated | **Resolved.** Paths enumerated; join against the final URL, then strip the fragment, then validate (§6) | `test_reconciliation_capture.py::test_policy_paths_exact_and_fragment_stripped_after_join` |
| S-13 | LOW | §6 | `_gm.html` pages would add ~13k redundant requests | **Resolved.** Excluded from the required corpus (§6 step 7) | none; decision recorded |
| S-14 | LOW | §10 | The comparison-result cache may be unnecessary | **Resolved.** Permitted simplification, conditional on measured targets and unchanged T20/T21 (§10) | T20/T21 |
| S-15 | LOW | `.githooks/pre-commit:29` | The default commit interpreter does not exist | **Resolved.** Phase H states `COUNCIL_PYTHON` explicitly and forbids `--no-verify` (§12). The hook itself is unchanged (out of scope) | T29: `test_reconciliation_cli.py::test_no_machine_specific_paths` |
| S-16 | LOW | `integrity/runner.py:541-613` | The output-alias check does not compare inodes against inputs | **Resolved.** T23 extended to hardlinks to input files outside the input roots (§13) | T23: `test_reconciliation_report.py::test_hardlinked_output_never_mutates_input_inode` |
| G-01 | LOW | Survey §3 S-01 evidence | Gaffer found the survey's "career Totals BR blank" statement is wrong | Evidence corrected in DESIGN §8; no design change | covered by T33 fixtures |
| N-01 | HIGH | DESIGN T14 | T14 still called a summary disagreement a SOURCE_CONFLICT, contradicting §8 and T32, so D06 could not be satisfied | **Resolved.** T14 now covers only same-appearance disagreements and single-sourced totals; printed vs double-sourced figures are T32 (§13) | T14 tests below |
| N-02 | HIGH | §9 aggregate partition; UNKNOWN row | The aggregate partition had no source-unavailable or not-applicable bucket, so pre-1965 aggregates would have been forced into unresolved (PASS unreachable) or equal (overstated) | **Resolved.** Aggregate partition gains source-unavailable and not-applicable; an aggregate whose source values are all `NOT_RECORDED` is source-unavailable; the UNKNOWN row reads "no FAIL condition" (§9) | `test_reconciliation_report.py::test_aggregate_partition_exhaustive_with_source_unavailable` |
| N-03 | MED | §8 step 5 | The career BR average model was wrong: the source divides by home-and-away games in seasons where the award was conducted, which reproduced 5/5 sampled careers vs at most 2/5 under the old text | **Resolved.** Separate BR career rule; the no-award seasons come from a captured, versioned evidence file, never inferred from blanks (§8). Scientist captures that evidence in Phase E | `test_reconciliation_aggregate.py::test_brownlow_career_average_excludes_no_award_seasons_only` |
| N-04 | MED | §8 summary-only rule | Career references across mixed per-game and summary-only seasons were undefined; rule conditions lacked "all"/"every" | **Resolved.** Composed reference = sum of per-club-season references; rule conditions say "all" and "every … for the player's team" (§8) | `test_reconciliation_aggregate.py::test_career_reference_composes_pergame_and_summary_only_seasons` |
| N-05 | MED | §8 "What blocks PASS" | The SOURCE_CONFLICT class omitted counter vs games-to-date, result, jumper, and single-sourced cells (one seam where strictness had loosened) | **Resolved.** Any disagreement between two source facts indexing one appearance is a SOURCE_CONFLICT; a printed total that disagrees with single-sourced cells leaves that aggregate unresolved; only multi-appearance figures are "derived" (§8) | `test_reconciliation_compare.py::test_counter_vs_match_games_to_date_conflict_blocks_pass`, `::test_single_sourced_cells_with_disagreeing_total_unresolved` |
| N-06 | LOW | §3 `http.py` row; §6 spacing bullet | Stale retry and conditional-request wording | **Resolved** (§3, §6) | none |
| N-07 | LOW | §8 DNTF | DNTF bucket placement and era scope were stated two ways | **Resolved.** Agreeing DNTF cells are not-applicable, never equal and never in `verified_numeric_fraction`; positive values are mismatches; the rule is condition-scoped, with a pilot-review count outside 2021–22 (§8) | T34 tests |
| N-08 | LOW | §12 Phase E | Samples cannot fully fixture T33 or settle 1931–34 BR on match pages | **Resolved.** The Phase E pilot captures a 1935–83 match page for a vote-getter, a 1933/34 Essendon match page, the no-award evidence and a 2011–15 substitute; states are decided before Phase F (§12) | pilot integration test |
| R3-01 | LOW | DESIGN §9 aggregate partition (l.568-571); §8 l.443-447 vs l.522 | Bucket placement unstated for mixed aggregates; "printed season totals are derived" vs summary-only season value as reference | **Resolved by binding clarification C-1** (§3a). Counts only; no verdict effect | `test_reconciliation_report.py::test_mixed_recorded_unrecorded_aggregate_bucket`, `::test_local_missing_summary_precedes_mismatch`; `test_reconciliation_aggregate.py::test_summary_only_season_value_is_reference_not_derived` |
| R3-02 | LOW | DESIGN §8 l.536-538 | No default for a cell printed on one source page and blank on the other, pending the pilot | **Resolved by binding clarification C-2** (§3a) | `test_reconciliation_compare.py::test_one_page_printed_other_blank_defaults_to_source_conflict`, `::test_pilot_rule_id_hashed_into_plan` |

### 3a Binding clarifications (part of this approval)

Surveyor run 3 approved `DESIGN.md` at `da64c4cf…` and raised two LOW items that affect
counts, not verdicts. To keep approval bound to the exact text the independent reviewer
read, they are recorded here rather than by editing the design. Scientist must treat them
as part of the approved design; QA and Surveyor check them at acceptance.

- **C-1 (R3-01).** In a composed aggregate, `NOT_RECORDED` club-seasons contribute nothing.
  The aggregate is source-unavailable only if every contribution is `NOT_RECORDED`;
  otherwise it is compared over the recorded contributions, and the recorded/unrecorded
  season counts are reported. When an aggregate both lacks a local summary value and
  differs, it is counted once, as local-missing-summary (which takes precedence over
  mismatch); both facts appear in the finding. A `SOURCE_SUMMARY_ONLY` season value is a
  *reference*, not a derived figure. Only printed totals and averages computed over
  multiple appearances or seasons are derived.
- **C-2 (R3-02).** Until a versioned rule covers it, a cell printed on one source page and
  blank on the other for the same appearance is a `SOURCE_CONFLICT` (UNKNOWN, blocks
  PASS). A Phase E pilot decision, for example on 1931–34 match-page BR, becomes a
  versioned rule ID with its evidence locator, hashed into the plan and cache keys like
  every other rule. Adding that rule is routine within this approved design.

## 4 Test matrix: T01–T35

All modules are under `tests/scvia/unit/` unless marked (I), meaning
`tests/scvia/integration/test_reconciliation_real.py`. Owner for every row: Scientist
(implementation and tests); QA verifies at acceptance.

| ID | Test(s) |
|---|---|
| T01 | `test_reconciliation_inventory.py::test_source_only_player_is_missing_locally_and_fails` |
| T02 | `test_reconciliation_inventory.py::test_failed_letter_makes_denominator_unknown`, `::test_lineup_profile_not_in_directory_breaks_census_completeness` |
| T03 | `test_reconciliation_identity.py::test_same_season_homonyms_resolved_by_club`, `::test_multiword_surname_resolved_by_dob_and_membership`, `::test_conflicting_dob_is_identity_conflict`, `::test_no_url_constructed_from_name` |
| T04 | `test_reconciliation_identity.py::test_two_local_ids_one_profile_is_conflict`, `::test_alias_cycle_rejected` |
| T05 | `test_reconciliation_compare.py::test_missing_appearance_detected_despite_final_counter` |
| T06 | `test_reconciliation_compare.py::test_offsetting_cell_swaps_detected_per_game` |
| T07 | `test_reconciliation_compare.py::test_duplicate_local_appearance_detected_before_aggregation` |
| T08 | `test_reconciliation_identity.py::test_drawn_gf_and_replay_distinguished_by_match_url`, `::test_replay_without_links_is_unknown` |
| T09 | `test_reconciliation_aggregate.py::test_two_club_season_counted_once_in_career`, `::test_career_gm_average_uses_distinct_seasons` |
| T10 | `test_reconciliation_cells.py::test_blank_with_team_total_is_recorded_zero`, `::test_blank_team_total_is_not_recorded`, `::test_pre1965_goals_only_from_match_structure`, `::test_all_zero_team_column_unresolved_without_fixture`; `test_reconciliation_source.py::test_notes_exception_lineage_name_maps_to_season_club` |
| T11 | `test_reconciliation_cells.py::test_finals_brownlow_not_applicable`, `::test_credited_unused_sub_row_is_appearance_not_zero_stats`, `::test_arrow_tokens_preserved_raw`, `::test_blank_tog_never_zero` |
| T12 | `test_reconciliation_source.py::test_duplicate_header_label_is_malformed`, `::test_rowspan_in_game_table_is_schema_gap`, `::test_permuted_headers_map_by_label`, `::test_alternate_year_table_without_pct_column` |
| T13 | `test_reconciliation_cells.py::test_strict_parse_rejects_nan_inf_bool_comma`; `test_reconciliation_aggregate.py::test_display_rounding_half_up_proven_by_ties` |
| T14 | `test_reconciliation_compare.py::test_profile_vs_match_cell_conflict_is_source_conflict`, `::test_counter_vs_match_games_to_date_conflict_blocks_pass`, `::test_result_or_jumper_conflict_blocks_pass`, `::test_single_sourced_cells_with_disagreeing_total_unresolved`, `::test_one_page_printed_other_blank_defaults_to_source_conflict`, `::test_pilot_rule_id_hashed_into_plan` |
| T15 | `test_reconciliation_inventory.py::test_games_after_through_date_excluded_by_match_date`, `::test_truncated_career_printed_total_marked_out_of_scope` |
| T16 | `test_reconciliation_capture.py::test_429_long_retry_after_persists_deadline_no_early_retry`, `::test_403_stops_host`, `::test_200_challenge_page_unusable`, `::test_404_profile_is_gap_not_empty_career`, `::test_timeout_and_oversized_are_gaps` |
| T17 | `test_reconciliation_capture.py::test_capture_sends_no_conditional_headers_and_writes_no_validators`, `::test_unsolicited_304_is_failure`, `::test_corrupt_reused_object_refused_and_refetched` |
| T18 | `test_reconciliation_capture.py::test_resume_after_sigterm_exact_queue`, `::test_second_writer_exits_5`, `::test_corrupt_checkpoint_refused`, `::test_changed_plan_refused` |
| T19 | `test_reconciliation_cli.py::test_compare_offline_socket_denied_relocated_archive_identical` |
| T20 | `test_reconciliation_report.py::test_outputs_identical_across_workers_shuffles_cold_warm_changed` |
| T21 | `test_reconciliation_cache.py::test_invalidation_source_cell`, `::test_invalidation_notes_map`, `::test_invalidation_identity_override`, `::test_invalidation_code_hash`, `::test_invalidation_local_row` |
| T22 | `test_reconciliation_local.py::test_input_drift_marks_incomplete_and_retains_fail` |
| T23 | `test_reconciliation_report.py::test_output_alias_refused_relative_symlink_hardlink`, `::test_hardlinked_output_never_mutates_input_inode` |
| T24 | `test_reconciliation_report.py::test_partial_write_leaves_no_completion_marker_and_exact_receipt` |
| T25 | `test_reconciliation_compare.py::test_historical_mismatch_fails_full_audit` |
| T26 | `test_reconciliation_local.py::test_quarantined_row_matching_source_is_coverage_gap`; (I) `::test_real_quarantine_rows_visible` |
| T27 | `test_reconciliation_source.py::test_unknown_numeric_column_blocks_schema_completeness`, `::test_known_non_statistic_columns_classified` |
| T28 | `test_reconciliation_inventory.py::test_latest_final_discovered_without_local_seed` (needs a new trimmed `seas/2026` fixture that includes the GF) |
| T29 | `test_reconciliation_cli.py::test_installed_cli_uses_packaged_configs`, `::test_no_machine_specific_paths` |
| T30 | `test_reconciliation_report.py::test_per_layer_verdicts_retained_combined_fail` |
| T31 | `test_reconciliation_inventory.py::test_lineup_profile_not_in_directory_breaks_census_completeness` |
| T32 | `test_reconciliation_aggregate.py::test_brownlow_average_uses_home_and_away_denominator`, `::test_career_average_excludes_unrecorded_era_games`, `::test_brownlow_career_average_excludes_no_award_seasons_only`, `::test_average_model_miss_is_source_consistency_not_local_fail`; `test_reconciliation_report.py::test_derived_figure_inconsistency_reported_not_blocking` |
| T33 | `test_reconciliation_aggregate.py::test_summary_only_brownlow_is_not_source_conflict`, `::test_summary_only_statistic_counted_separately`, `::test_local_missing_summary_value_fails_layer`, `::test_career_reference_composes_pergame_and_summary_only_seasons`; `test_reconciliation_report.py::test_aggregate_partition_exhaustive_with_source_unavailable`, `::test_mixed_recorded_unrecorded_aggregate_bucket`, `::test_local_missing_summary_precedes_mismatch`; `test_reconciliation_aggregate.py::test_summary_only_season_value_is_reference_not_derived` |
| T34 | `test_reconciliation_cells.py::test_credited_unused_sub_row_is_appearance_not_zero_stats`, `::test_dntf_local_positive_is_mismatch`, `::test_dntf_cells_never_counted_as_verified`, `::test_dntf_outside_2021_22_reported_for_pilot_review` |
| T35 | `test_reconciliation_capture.py::test_spacing_persists_across_resume`, `::test_spacing_two_seconds_includes_retries_redirects_robots`; `test_http.py::test_retry_after_field_is_additive` |
| Pilot | (I) `test_reconciliation_real.py::test_pilot_values_traceable_to_cells` |

Test names are the required minimum. Scientist may rename them, provided the mapping is
kept in the run record. Fixtures are trimmed from the retained review samples.

## 5 Explicit decisions

- **Scope.** Men's senior VFL/AFL premiership matches from 1897 through `--through-date 2026-09-30`, including finals, drawn finals and their replays. Every supported statistic is in scope, including pre-1984 season-only Brownlow (`SOURCE_SUMMARY_ONLY`). Consequence (Surveyor run 2): the candidate holds no BR for 1935–83, so a primary-snapshot PASS (D16) is unreachable for it until local data stores season-level award values. That is an honest data FAIL, not a software defect. Narrowing that is an owner choice that must be named in the attestation; it is not taken here. `_gm.html` pages are not required.
- **Identity fallback.** In order: verified local source URL; evidence-backed override; unique name + full DOB + membership; unique full DOB + identical (season, club) set (3b); a globally unique exact appearance-set equality when DOB is missing or low precision (4). Otherwise UNKNOWN. No fuzzy authorisation, no URL built from a name, global uniqueness enforced.
- **Blank evidence.** Six source states plus `NOT_APPLICABLE_DNTF`. A blank team Totals cell means not recorded; a non-blank team Totals with a blank player cell means zero; the all-zero column encoding is unresolved until proven; pre-1965 availability comes from match structure; `%P` is never zero-filled; the repository's "modern blank = zero" convention is not adopted.
- **Source conflicts.** Any disagreement between two source facts indexing the same appearance is a `SOURCE_CONFLICT` (cell, membership, counter vs games-to-date, result, jumper): affected cells are UNKNOWN and PASS is blocked. A printed multi-appearance figure that disagrees with the composed reference is a `SOURCE_DERIVED_INCONSISTENCY` (`source_consistent=false`; does not block) when every contributing cell is double-sourced. Otherwise that aggregate is unresolved. A page is never chosen because it agrees with the repository.
- **Performance targets.** Unchanged and to be measured, not fabricated: full offline compare within 10 min and 2 GiB peak process-tree RSS with four workers; warm compare within 2 min; at most +10 s on the fast tier. Acquisition is measured separately; the estimate is at least 17 h. A missed target stays open; coverage is never reduced to meet it. The smaller cache stack is permitted if it meets the targets.
- **CLI contracts.** `scvia reconcile-afltables plan|capture|compare` as in DESIGN §11, registered as a Typer sub-app. Exit codes 0/2/4/5/8/9 reuse the existing `EXIT` map in `cli.py`. `plan` and `compare` refuse network access; `capture --resume` resumes with an unchanged plan.
- **Completeness accounting.** Census closure identity added. Appearance, cell and aggregate partitions are exhaustive and exclusive. The aggregate partition includes source-unavailable and not-applicable (N-02). Recorded-zero and malformed are sub-counts, and agreeing DNTF cells are not-applicable; duplicates and conflicts are orthogonal flags. A failed census makes the denominator unknown. Five separate fields: `execution_complete`, `capture_complete`, `identity_complete`, `comparison_complete` and `source_consistent`.
- **Harness scope.** The audit is opt-in and is not wired into any schedule, hook or gate. The only shared-code change anticipated is the additive `http.py` result field. If default behavior is byte-identical (proven by test), CLAUDE.md §6.2 test 2 is not triggered; Scientist records this in writing, and any behavior change requires the scratch smoke run.

## 6 What was not done

- **Scientist was not consulted during this review.** The source-semantics evidence came from Surveyor's measured samples, and Gaffer re-checked four of its claims (§2). Phase B red tests and the Phase E pilot are where Scientist validates each new rule against fixtures. A rule that the pilot contradicts is a material change and returns to architecture review.
- No implementation, bulk capture, data, release, hook, harness or schedule change was made. QA and DataSentinel did not run; neither is applicable at the design stage.
- Whether 2011–2015 substitutes follow the credited-did-not-take-field convention is unverified (pilot item).
- The all-zero team-column encoding is unverified (stays UNRESOLVED until a fixture proves it).
- The candidate is expected, not established, to FAIL on pre-1984 season-only Brownlow; Scientist's audit decides.
- The no-award Brownlow seasons (inferred 1942–45 from samples) are unverified until Scientist captures the evidence file in Phase E.
- 1931–34 match-page BR (whether it is printed per game) is unverified; decided in Phase E.

## 7 Implementation order (if approved)

1. **Phase B.** Contracts and red tests: strict schemas, outcome rules, output safety, fixtures trimmed from the retained samples; failing tests first (T01–T35).
2. **Phase C.** Offline comparison: local adapters, the independent reader (rejecting duplicate labels and rowspans), identity rules 1–4, appearance, cell and aggregate comparison, the source-average model, report accounting.
3. **Phase D.** Acquisition: reconciliation policy, coordinator-owned retries with a persisted deadline, the additive `http.py` field with a default-unchanged regression, census closure, checkpoint and resume, the mock-network adversarial suite.
4. **Phase E.** Bounded live pilot over the review's sample cases plus the 2011–15 substitute check; every value traced by hand.
5. **Phase F.** Full capture (at least 17 h), four deterministic comparisons, mutation copies, the existing full integrity checker, the legacy-CSV layer.
6. **Phase G.** Gaffer final acceptance with a fresh Surveyor review and QA.

## 8 Next role: Scientist on Sonnet 5.5

```bash
cd /home/abhi/git/SuperCoach-VIA
claude --agent Scientist --model claude-sonnet-5-5 --effort high \
  "Follow docs/rewrite/afltables-reconciliation/ENGINEER_PROMPT.md. Verify Opus approval, implement the approved design with tests, and execute the full all-player reconciliation. Produce reproducible reports and an honest PASS, FAIL or UNKNOWN verdict."
```

Not launched by this review.
