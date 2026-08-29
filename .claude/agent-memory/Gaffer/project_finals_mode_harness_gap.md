---
name: finals-mode-harness-gap
description: The weekly harness has no finals mode — during finals it fabricates a phantom "round 26" prediction that poisons cheat-sheet selection into 2027; never launch the full harness during finals until a flag lands
metadata:
  type: project
---

**ANTI-PATTERN: never launch `scripts/weekly_refresh.sh` during the finals window
until a finals-mode flag lands.** It will fabricate a prediction for a round that
does not exist, and the phantom CSV outlives the mistake by more than a year.

Established 2026-08-29 (verified in files, Surveyor DEEP survey concurring).

**Season shape.** AFL 2026 H&A is **25 rounds / 23 games per team** (207 matches,
9 per round) — same as 2024 and 2025. Finals are labelled as STRINGS:
`Qualifying Final`/`Elimination Final`/`Semi Final`/`Preliminary Final`/`Grand Final`
in `data/matches/`, and `QF`/`EF`/`SF`/`PF`/`GF` in `data/player_data/`.

**Why the harness breaks.** `supercoach/prediction.py::extract_round_number`
returns NaN for every finals label; `get_next_round` (:999-1028) is
`max(non-NaN round_number) + 1`. Once round 25 is scraped it returns **26**.
The model cannot predict finals by construction: `round_number` is a model
feature (:557) and NaN-round rows are dropped at :1110.

**The round number is derived from the prediction ARTIFACT, not from the data** —
that is the root fault, and it cuts both ways:
- `scripts/weekly_refresh.sh:140-156` reads `ROUND` from the newest
  `next_round_*.csv` and hard-aborts if its mtime predates the run. So you cannot
  simply skip prediction.
- `refresh_and_rank.sh:93-100` sets `END_SCORE_ROUND = UPCOMING_ROUND - 1`. Skip
  prediction and `seq START END` goes empty — **the backtest is silently skipped
  with no error.**

**Worst blast radius — `scripts/generate_weekly_cheat_sheet.py:40-58`.**
`find_latest_prediction` keys on `(int(round), timestamp)` parsed from the
FILENAME, and filenames carry **no year**. A phantom `next_round_26` file would
outrank every round 1-25 of 2027 *forever*, until someone deletes it. One
accidental full-harness run poisons cheat-sheet selection for a season.
(Related: `next_round_25_*` already exists for both 2025 and 2026 — year-less
namespace is a latent collision, Surveyor F5, Scientist-owned.)

**Both scrape-side gates fail OPEN on finals rows, for the whole series:**
`scripts/check_round_settled.py:52-59` drops non-integer rounds, so it pins to
round 25 and passes without inspecting a single finals game;
`audit_match_rounds` (`scrapers/game_scraper.py:248-252`) restricts to integer
H&A rounds by explicit design. Only `phantom_row_validator.py` is finals-literate
(`_FINALS_ROUND_CODES`). A dropped Grand Final row is invisible to the
completeness gate.

**How to apply.** During finals, do not run the full harness. The fix shape
(agreed with Surveyor, pending human sign-off): derive the round and the backtest
bound from the last settled integer round in `matches_<year>.csv` — the
computation `check_round_settled.py` already does — and gate everything behind one
flag so default-path behaviour is unchanged. Note the semantic flip: `ROUND`
currently means *upcoming*; in finals mode it means *completed*. CLAUDE.md 6.2 is
triggered on the bright line (both entry-point .sh files) and needs TWO smoke runs
via `scripts/smoke_harness.sh`: flag-off proving nothing changed, and flag-on with
a real scrape.

See [[weekly-r24-r25-retro]] (BL-19, the recap mislabel this makes structural) and
[[backtest-completion-manifest]].
