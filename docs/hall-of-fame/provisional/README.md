# PROVISIONAL — Hall of Fame operator tables

**PROVISIONAL · as of 2026-10-07.** Snapshot `sha256:b9830cbf1093544e234f26159440a2cc84eb834a223da13d05e931ae8f5815c4`. These are data-based ranks, not official AFL Hall of Fame selections.

**Source agreement UNKNOWN: zero confirmed discrepancies; 63 unresolved source cells per layer.** Full parent audit: `sha256:f1abd8c2b7f8d6b3812e73f91a4403add1844f3ad9037139e7dc373a0ed91703`, report SHA-256 `81ac70ba8aa25110119a91779d2c2df91088b2f0f87408db3546de8109a4509c`, capture 2026-10-01T11:46:00Z to 2026-10-02T04:44:55Z. Candidate audit: `sha256:b9830cbf1093544e234f26159440a2cc84eb834a223da13d05e931ae8f5815c4`, seasons 2026 only, report SHA-256 `6fecac8e1c8eb804774eb70a5f9d03bd5448ee86d43d1253472ca0f1ea7d9c2a`, capture 2026-10-05T20:05:26Z to 2026-10-05T20:35:27Z. Verified unchanged historical partitions carry the parent's unresolved evidence forward. This composition is not a new full audit or fresh historical capture. Legacy counts describe the cited reports; the fragment continuity proof applies to the snapshot. Historical Brownlow, all-zero and percentage evidence remains unresolved. Totals sum recorded values; all-missing totals are absent. Means divide by stat observed_games. Recorded appearances count rows; career_games uses max(rows, source counter). legacy_v1 retains its pinned historical blank-as-zero scoring imputation and era adjustments.

Source: queried immutable snapshot manifest at `var/candidates/20261005-afltables-corrected/snapshots/b9830cbf1093544e234f26159440a2cc84eb834a223da13d05e931ae8f5815c4.json`. Audit report: `docs/reviews/evidence/afltables-candidate-source-coverage-20261007.json`. Raw inputs are retained locally. [Verified provenance](provenance.json).

Ranking method hash: `191aa99b83f618d3e90d8ce68b8e467dcc6562993e81706f5c3f44e12332d563`. All additive current stats are included; time-on-ground percentage is not additive.

These separate operator reports do not promote the candidate. Rebuild them when the pinned source evidence or snapshot changes.

Repeat offline from repository root:

```bash
.venv/bin/python scripts/refresh_provisional_hof.py --data-root var/candidates/20261005-afltables-corrected --snapshot sha256:b9830cbf1093544e234f26159440a2cc84eb834a223da13d05e931ae8f5815c4 --coverage-receipt docs/reviews/evidence/afltables-candidate-source-coverage-20261007.json --audit-sha256 54fdc52c35721abdad30adf15981f7f7c901f7cb073eb601949621e87d761689 --as-of 2026-10-07
```

Add `--check` to verify existing output bytes without writing. All manifest fragments and the pinned evidence hash are verified before querying.

- [legacy_v1 top100 — score is a method output](top100.md) ([CSV](top100.csv), [JSON](top100.json))
- [career kicks — recorded totals](career/kicks.md) ([CSV](career/kicks.csv), [JSON](career/kicks.json))
- [single-season kicks — recorded totals](single-season/kicks.md) ([CSV](single-season/kicks.csv), [JSON](single-season/kicks.json))
- [career marks — recorded totals](career/marks.md) ([CSV](career/marks.csv), [JSON](career/marks.json))
- [single-season marks — recorded totals](single-season/marks.md) ([CSV](single-season/marks.csv), [JSON](single-season/marks.json))
- [career handballs — recorded totals](career/handballs.md) ([CSV](career/handballs.csv), [JSON](career/handballs.json))
- [single-season handballs — recorded totals](single-season/handballs.md) ([CSV](single-season/handballs.csv), [JSON](single-season/handballs.json))
- [career disposals — recorded totals](career/disposals.md) ([CSV](career/disposals.csv), [JSON](career/disposals.json))
- [single-season disposals — recorded totals](single-season/disposals.md) ([CSV](single-season/disposals.csv), [JSON](single-season/disposals.json))
- [career goals — recorded totals](career/goals.md) ([CSV](career/goals.csv), [JSON](career/goals.json))
- [single-season goals — recorded totals](single-season/goals.md) ([CSV](single-season/goals.csv), [JSON](single-season/goals.json))
- [career behinds — recorded totals](career/behinds.md) ([CSV](career/behinds.csv), [JSON](career/behinds.json))
- [single-season behinds — recorded totals](single-season/behinds.md) ([CSV](single-season/behinds.csv), [JSON](single-season/behinds.json))
- [career hitouts — recorded totals](career/hitouts.md) ([CSV](career/hitouts.csv), [JSON](career/hitouts.json))
- [single-season hitouts — recorded totals](single-season/hitouts.md) ([CSV](single-season/hitouts.csv), [JSON](single-season/hitouts.json))
- [career tackles — recorded totals](career/tackles.md) ([CSV](career/tackles.csv), [JSON](career/tackles.json))
- [single-season tackles — recorded totals](single-season/tackles.md) ([CSV](single-season/tackles.csv), [JSON](single-season/tackles.json))
- [career rebound_50s — recorded totals](career/rebound_50s.md) ([CSV](career/rebound_50s.csv), [JSON](career/rebound_50s.json))
- [single-season rebound_50s — recorded totals](single-season/rebound_50s.md) ([CSV](single-season/rebound_50s.csv), [JSON](single-season/rebound_50s.json))
- [career inside_50s — recorded totals](career/inside_50s.md) ([CSV](career/inside_50s.csv), [JSON](career/inside_50s.json))
- [single-season inside_50s — recorded totals](single-season/inside_50s.md) ([CSV](single-season/inside_50s.csv), [JSON](single-season/inside_50s.json))
- [career clearances — recorded totals](career/clearances.md) ([CSV](career/clearances.csv), [JSON](career/clearances.json))
- [single-season clearances — recorded totals](single-season/clearances.md) ([CSV](single-season/clearances.csv), [JSON](single-season/clearances.json))
- [career clangers — recorded totals](career/clangers.md) ([CSV](career/clangers.csv), [JSON](career/clangers.json))
- [single-season clangers — recorded totals](single-season/clangers.md) ([CSV](single-season/clangers.csv), [JSON](single-season/clangers.json))
- [career frees_for — recorded totals](career/frees_for.md) ([CSV](career/frees_for.csv), [JSON](career/frees_for.json))
- [single-season frees_for — recorded totals](single-season/frees_for.md) ([CSV](single-season/frees_for.csv), [JSON](single-season/frees_for.json))
- [career frees_against — recorded totals](career/frees_against.md) ([CSV](career/frees_against.csv), [JSON](career/frees_against.json))
- [single-season frees_against — recorded totals](single-season/frees_against.md) ([CSV](single-season/frees_against.csv), [JSON](single-season/frees_against.json))
- [career brownlow_votes — recorded totals](career/brownlow_votes.md) ([CSV](career/brownlow_votes.csv), [JSON](career/brownlow_votes.json))
- [single-season brownlow_votes — recorded totals](single-season/brownlow_votes.md) ([CSV](single-season/brownlow_votes.csv), [JSON](single-season/brownlow_votes.json))
- [career contested_possessions — recorded totals](career/contested_possessions.md) ([CSV](career/contested_possessions.csv), [JSON](career/contested_possessions.json))
- [single-season contested_possessions — recorded totals](single-season/contested_possessions.md) ([CSV](single-season/contested_possessions.csv), [JSON](single-season/contested_possessions.json))
- [career uncontested_possessions — recorded totals](career/uncontested_possessions.md) ([CSV](career/uncontested_possessions.csv), [JSON](career/uncontested_possessions.json))
- [single-season uncontested_possessions — recorded totals](single-season/uncontested_possessions.md) ([CSV](single-season/uncontested_possessions.csv), [JSON](single-season/uncontested_possessions.json))
- [career contested_marks — recorded totals](career/contested_marks.md) ([CSV](career/contested_marks.csv), [JSON](career/contested_marks.json))
- [single-season contested_marks — recorded totals](single-season/contested_marks.md) ([CSV](single-season/contested_marks.csv), [JSON](single-season/contested_marks.json))
- [career marks_inside_50 — recorded totals](career/marks_inside_50.md) ([CSV](career/marks_inside_50.csv), [JSON](career/marks_inside_50.json))
- [single-season marks_inside_50 — recorded totals](single-season/marks_inside_50.md) ([CSV](single-season/marks_inside_50.csv), [JSON](single-season/marks_inside_50.json))
- [career one_percenters — recorded totals](career/one_percenters.md) ([CSV](career/one_percenters.csv), [JSON](career/one_percenters.json))
- [single-season one_percenters — recorded totals](single-season/one_percenters.md) ([CSV](single-season/one_percenters.csv), [JSON](single-season/one_percenters.json))
- [career bounces — recorded totals](career/bounces.md) ([CSV](career/bounces.csv), [JSON](career/bounces.json))
- [single-season bounces — recorded totals](single-season/bounces.md) ([CSV](single-season/bounces.csv), [JSON](single-season/bounces.json))
- [career goal_assists — recorded totals](career/goal_assists.md) ([CSV](career/goal_assists.csv), [JSON](career/goal_assists.json))
- [single-season goal_assists — recorded totals](single-season/goal_assists.md) ([CSV](single-season/goal_assists.csv), [JSON](single-season/goal_assists.json))
- [Recorded appearances](career/recorded-appearances.md) ([CSV](career/recorded-appearances.csv), [JSON](career/recorded-appearances.json))
