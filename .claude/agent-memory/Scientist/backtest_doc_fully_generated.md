---
name: backtest-doc-fully-generated
description: docs/afl-backtest-2026.md is now fully auto-generated from 6 marker blocks with ONE deliberate frozen exemption (the R18 decision record) — never hand-edit a figure into it, and never "refresh" the frozen block.
metadata:
  type: project
---

As of 2026-07-28 `docs/afl-backtest-2026.md` carries **no hand-maintained figures**.
It had failed the DataSentinel gate on stale prose twice in two days
([[project_backtest_partial_regen_stale_blocks]],
[[project_backtest_known_coverage_limitation_stale]] in DataSentinel's memory),
so the user's decision was to convert the whole surface rather than keep patching it.

**Why:** hand-written prose citing a specific figure goes stale by construction every
round. The only durable fix is that no figure has an author.

**How to apply:**

Six marker pairs, two generators:

| Block | Generator | Content |
|---|---|---|
| `2026-BACKTEST` | `update_team_analysis.py` | per-round table, top-30 table, threshold disclosure |
| `CUMULATIVE` | `scripts/update_eval_surface.sh` | player-weighted pool + "Read:" callout |
| `TEAMBIAS` | same | per-club signed bias |
| `MISSES` | same | top-5 over/under per round |
| `TRAINCORPUS` | same | season span, loading population, `prediction.py` line refs |
| `VINTAGEPATH` | same | round -> vintage -> retrain/archive path -> attestation |

- **Never hand-type a number into this doc.** If a figure is wrong, fix the generator.
- `TRAINCORPUS` derives the `supercoach/prediction.py` line numbers by *source anchor*,
  not by literal. The doc had cited `:589` long after the anchor moved to `:598`. The
  anchors are `historical_data = df[df['year'] < self.target_year]` and
  `birth_year_threshold = self.target_year - 40`. If you rename either, the script
  **hard-exits** — that is intentional, not a bug.
- The season span must be computed over the files the **birth-year filter admits**
  (filename DOB token > `target_year - 40`), not over all of `data/player_data/`.
  Scanning everything reports a span decades too early — the model can never see
  those rows. See [[model_training_corpus_scope]].
- `VINTAGEPATH` attestation compares the forward CSV's first git commit against the
  round's first bounce read at **UTC+8** (earliest Australian venue offset), so it
  fails closed regardless of which state the game was in. No git / never committed /
  committed late all render "not attested".
- `update_eval_surface.sh` now requires `supercoach/prediction.py` to exist. Any tmp-repo
  test fixture driving the real script must create it **and** a player file with at
  least one pre-target-year row, or the script fails closed on the corpus scan.

**The one exemption — do not "fix" it.** The "Known coverage limitation — Round 18 2026"
section is a *dated decision record* ("as at R1–R20, 2026-07-26"), not a live figure. It
is deliberately outside every marker pair, and
`test_r18_coverage_limitation_record_is_not_regenerated` pins that. Re-pointing it at the
latest round would misrepresent the evidence the decision was taken on. It cannot go
stale because it makes no claim about current state and cross-references the live
Cumulative block for that. Verified 2026-07-28: every cell still reproduces exactly from
a pinned R1–R20 pool (7,153/7,281, 3.958/3.960, −0.110/−0.105, 74.36/74.40, 95.78/95.78,
St Kilda −0.584→−0.733).

It was left frozen rather than generated-over-a-pinned-window because reproducing the
"with the fuller R18 vintage" column requires deliberately loading a **superseded**
artifact (`prediction_vs_actual_round_18_2026_20260707_154033.csv`) that the keep-last
rule says to ignore — a generator that reaches around its own vintage discipline, and
one whose output would silently change if any pre-R20 round were ever re-scored.

See also [[backtest_doc_verification]], [[backtest_artifact_vintage_selection]].
