"""Integration tier — QA's data-touching checks, made automatic.

These assertions used to live only inside the QA agent's manual checklist
(`.claude/agents/QA.md`), which means they ran only when a human remembered to
invoke QA. Every check ported here is deterministic and genuinely blocking: it
fails on a defect, not on legitimate week-to-week variation.

Deliberately NOT ported (left to QA's judgement, see the report and QA's
memory under `.claude/agent-memory/QA/`):

  * the `assets/charts/hall/` "at least 6 chart files" mandatory-artifact rule —
    `generate_records_charts.py` emits exactly 4 by design and always has, so
    the rule is chronically unmet by spec drift, not by regression. The
    non-noisy half of that intent IS ported, as
    `test_every_chart_referenced_by_a_published_doc_exists`.
  * "top-5 career-games players must have current-season rows" — a legitimate
    retirement makes this fail with no defect present.
  * "round gaps in the current season" as a whole — the TOP of the range moves
    every week (the current round is not scraped yet). Only the deterministic
    half, an INTERIOR gap, is ported.
  * "new .py files must have tests" — QA.md itself says "use judgment".

Already covered elsewhere, so not duplicated here: `_stat_leaders.json` schema
and the rank-1 games cross-check (tests/unit/test_qa_stat_leaders_schema.py),
`all_time_top_100.csv` row count and ordering (tests/unit/test_top100_sync.py),
`check_hof_numbers.py`'s own logic (tests/unit/test_check_hof_numbers.py),
phantom/counter-gap player rows (tests/unit/test_phantom_row_validator.py plus
the blocking phantom-row gate the harness already runs).

Every test here is marked `integration` and is excluded from the pre-commit tier.
"""
import glob
import importlib.util
import os
import re
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

pytestmark = pytest.mark.integration

PREDICTION_GLOB = os.path.join(
    _REPO_ROOT, "data", "prediction", "next_round_*_prediction_*.csv"
)
BACKTEST_GLOB = os.path.join(
    _REPO_ROOT, "data", "prediction", "backtest", "backtest_summary_*.csv"
)
MATCHES_GLOB = os.path.join(_REPO_ROOT, "data", "matches", "matches_*.csv")


def _newest(pattern):
    """Newest by MTIME, never by lexicographic sort.

    A plain `sorted(...)[-1]` ranks `next_round_9_*` above `next_round_18_*`
    because "9" > "1" as a string. That bug published weekly recaps under the
    wrong round label for several cycles (Surveyor CR-1 / F01). Do not
    reintroduce it here.
    """
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None


# --------------------------------------------------------------- predictions


def test_newest_prediction_csv_has_the_schema_downstream_docs_assume():
    """The weekly cheat sheet, the insights recap and the eval banner all read
    this file positionally. A column rename or an out-of-range value ships as a
    wrong published number rather than as a crash."""
    import pandas as pd

    path = _newest(PREDICTION_GLOB)
    assert path, f"no prediction CSV matches {PREDICTION_GLOB}"
    df = pd.read_csv(path)
    rel = os.path.relpath(path, _REPO_ROOT)

    required = {"player", "team", "predicted_disposals"}
    assert required.issubset(df.columns), (
        f"{rel} missing column(s) {sorted(required - set(df.columns))}; "
        f"has {list(df.columns)}"
    )
    assert len(df) > 10, f"{rel} has only {len(df)} rows — a truncated prediction run"

    pred = pd.to_numeric(df["predicted_disposals"], errors="coerce")
    assert pred.notna().all(), (
        f"{rel} has {int(pred.isna().sum())} non-numeric/NaN predicted_disposals"
    )
    out_of_range = df[~pred.between(0, 80)]
    assert out_of_range.empty, (
        f"{rel} has {len(out_of_range)} prediction(s) outside 0-80 disposals — "
        f"the model or the feature frame is broken:\n{out_of_range.head().to_string()}"
    )
    assert df["player"].notna().all() and df["team"].notna().all(), (
        f"{rel} has null player/team labels"
    )


def test_newest_prediction_csv_has_no_duplicate_player_rows():
    """One row per player per team. Duplicates here are the published-side
    symptom of the dtype-blind dedup incident that doubled 833 player CSVs."""
    import pandas as pd

    path = _newest(PREDICTION_GLOB)
    assert path, f"no prediction CSV matches {PREDICTION_GLOB}"
    df = pd.read_csv(path)
    dupes = df[df.duplicated(subset=["player", "team"], keep=False)]
    assert dupes.empty, (
        f"{os.path.relpath(path, _REPO_ROOT)} has {len(dupes)} duplicated "
        f"player/team rows:\n{dupes.head(10).to_string()}"
    )


# ------------------------------------------------------------------ backtest


def test_backtest_summary_exists_and_is_well_formed():
    """The eval banner's MAE / within-5 pills are computed from these files.

    Only structure is asserted, not values: the accuracy numbers legitimately
    move every round, so pinning them would fail on a good week.
    """
    import pandas as pd

    path = _newest(BACKTEST_GLOB)
    assert path, f"no backtest summary matches {BACKTEST_GLOB}"
    df = pd.read_csv(path)
    rel = os.path.relpath(path, _REPO_ROOT)

    assert len(df) > 0, f"{rel} is empty"
    required = {"round", "year", "n_players", "mae", "pct_within_5"}
    assert required.issubset(df.columns), (
        f"{rel} missing column(s) {sorted(required - set(df.columns))}; "
        f"has {list(df.columns)}"
    )
    mae = pd.to_numeric(df["mae"], errors="coerce")
    assert mae.notna().all(), f"{rel} has non-numeric mae values"
    assert (mae > 0).all() and (mae < 30).all(), (
        f"{rel} has implausible MAE values: {mae.tolist()}"
    )
    pct = pd.to_numeric(df["pct_within_5"], errors="coerce")
    assert pct.between(0, 100).all(), f"{rel} pct_within_5 outside 0-100: {pct.tolist()}"


# ----------------------------------------------------------------------- HOF


def test_check_hof_numbers_passes_on_the_published_docs():
    """QA checklist item 4, run automatically.

    `scripts/check_hof_numbers.py` compares each rendered HOF stat page (and the
    hub) against `_stat_leaders.json`. It is the authoritative regression gate
    for those pages — the JSON itself is gitignored, so `git diff` cannot catch
    a drift here.
    """
    json_path = os.path.join(_REPO_ROOT, "docs", "hall-of-fame", "_stat_leaders.json")
    if not os.path.exists(json_path):
        pytest.skip(
            "_stat_leaders.json not present (gitignored; generated by the HOF "
            "pipeline) — nothing to verify on this checkout"
        )
    spec = importlib.util.spec_from_file_location(
        "check_hof_numbers", os.path.join(_REPO_ROOT, "scripts", "check_hof_numbers.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.check_hof_numbers() == 0, (
        "check_hof_numbers.py reported a mismatch between the published HOF "
        "pages and _stat_leaders.json (see captured stdout for the offending rows)"
    )


# ----------------------------------------------------------------- match data


def test_current_season_match_file_has_no_interior_round_gaps():
    """A missing round BELOW the highest scraped round is a dropped-game defect.

    QA treats round gaps as a warning because the newest round may not be
    scraped yet — but that only excuses a gap at the TOP of the range. A hole in
    the middle (round 7 absent while round 20 is present) is the incremental-
    scrape blind spot that has cost this repo real games, and no legitimate
    seasonal variation produces it.

    Finals labels ("Grand Final", …) are non-numeric and are ignored.
    """
    import pandas as pd

    files = sorted(glob.glob(MATCHES_GLOB))
    assert files, f"no match files match {MATCHES_GLOB}"
    path = files[-1]  # matches_<year>.csv sorts chronologically
    df = pd.read_csv(path)
    rel = os.path.relpath(path, _REPO_ROOT)

    rounds = pd.to_numeric(df["round_num"], errors="coerce").dropna().astype(int)
    assert not rounds.empty, f"{rel} has no numeric home-and-away rounds"

    present = set(rounds.unique())
    gaps = sorted(set(range(min(present), max(present) + 1)) - present)
    assert not gaps, (
        f"{rel} is missing interior round(s) {gaps} (present range "
        f"{min(present)}-{max(present)}) — an incremental scrape cannot backfill "
        f"a past round, so this needs a targeted re-scrape"
    )


# ------------------------------------------------------------ published charts


def test_every_chart_referenced_by_a_published_doc_exists():
    """No broken image links in the shipped docs.

    This is the non-noisy half of QA's chart-artifact check: rather than
    asserting a chart COUNT (which is chronically wrong by spec and grows
    legitimately), assert that every chart a published page points at is
    actually on disk. A generator that stops emitting a chart, or a rename,
    shows up here as a broken image on a live page.
    """
    docs = sorted(
        glob.glob(os.path.join(_REPO_ROOT, "docs", "**", "*.md"), recursive=True)
        + glob.glob(os.path.join(_REPO_ROOT, "*.md"))
    )
    assert docs, "no markdown docs found — fixture/env problem"

    missing = []
    checked = 0
    for doc in docs:
        with open(doc, encoding="utf-8") as f:
            text = f.read()
        for m in re.finditer(r"!\[[^\]]*\]\(([^)\s]+)", text):
            target = m.group(1)
            if target.startswith(("http://", "https://", "data:")):
                continue
            checked += 1
            resolved = os.path.normpath(os.path.join(os.path.dirname(doc), target))
            if not os.path.exists(resolved):
                missing.append(f"{os.path.relpath(doc, _REPO_ROOT)} -> {target}")

    assert checked > 0, "no local image references found — the check would be vacuous"
    assert not missing, (
        f"{len(missing)} published doc(s) reference a chart that does not exist:\n"
        + "\n".join(missing[:20])
    )
