"""Schema + integrity tests for the conceded-stats file and its repair writer.

The conceded-stats file (data/conceded_stats/team_stats_conceded_YEAR.csv) holds,
for every team in every match, the statistical totals it *conceded* to its
opponent (i.e. the sum of the opponent's player stats in that match).

Two classes of test live here:

1. Pure-logic unit tests for the repair function `fix_conceded_columns`
   (synthetic data, no real files) -- the TDD core.
2. Data-integrity tests that assert the real regenerated file obeys the schema
   and the physical invariants. These FAIL against the corrupt file and PASS
   after the writer repairs it.

Key physical invariant: disposals = kicks + handballs. In the corrupt file this
identity does NOT hold for `*_conceded`; instead `marks == kicks + disposals`,
the fingerprint of a 3-column value rotation among disposals/handballs/marks.
"""

import os

import pandas as pd
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONCEDED_FILE = os.path.join(
    REPO, "data", "conceded_stats", "team_stats_conceded_2025.csv"
)
MATCHES_FILE = os.path.join(REPO, "data", "matches", "matches_2025.csv")

EXPECTED_COLUMNS = [
    "year", "round", "team", "opponent",
    "disposals_conceded", "kicks_conceded", "handballs_conceded",
    "marks_conceded", "goals_conceded", "behinds_conceded",
    "tackles_conceded", "hitouts_conceded", "inside_50s_conceded",
    "clearances_conceded",
]

# Full-team conceded plausibility ranges (a whole team's totals in one game).
VALUE_RANGES = {
    "disposals_conceded": (250, 520),
    "kicks_conceded": (140, 300),
    "handballs_conceded": (40, 210),
    "marks_conceded": (40, 170),
    "goals_conceded": (2, 35),
    "behinds_conceded": (2, 35),
}


# ---------------------------------------------------------------------------
# 1. Pure-logic unit tests for the repair function (no real files)
# ---------------------------------------------------------------------------

def _make_corrupt_row():
    """One synthetic corrupt row: values rotated among disposals/handballs/marks.

    True stats: disposals=350, kicks=210, handballs=140, marks=90.
    Corrupt layout (what the buggy writer emitted):
      disposals_conceded <- handballs (140)
      handballs_conceded <- marks     (90)
      marks_conceded     <- disposals (350)
    """
    return {
        "year": 2025, "round": 1, "team": "A", "opponent": "B",
        "disposals_conceded": 140,   # actually handballs
        "kicks_conceded": 210,       # correct
        "handballs_conceded": 90,    # actually marks
        "marks_conceded": 350,       # actually disposals
        "goals_conceded": 12, "behinds_conceded": 10,
        "tackles_conceded": 55, "hitouts_conceded": 38,
        "inside_50s_conceded": 52, "clearances_conceded": 40,
    }


def test_fix_corrects_rotated_columns():
    from scripts.build_conceded_stats import fix_conceded_columns
    df = pd.DataFrame([_make_corrupt_row()])
    out = fix_conceded_columns(df)
    r = out.iloc[0]
    assert r["disposals_conceded"] == 350
    assert r["kicks_conceded"] == 210
    assert r["handballs_conceded"] == 140
    assert r["marks_conceded"] == 90
    # Untouched columns preserved
    assert r["goals_conceded"] == 12
    assert r["hitouts_conceded"] == 38
    # Invariant now holds
    assert r["disposals_conceded"] == r["kicks_conceded"] + r["handballs_conceded"]


def test_fix_is_idempotent_on_correct_data():
    """Running the fix on an already-correct row must be a no-op."""
    from scripts.build_conceded_stats import fix_conceded_columns
    correct = {
        "year": 2025, "round": 1, "team": "A", "opponent": "B",
        "disposals_conceded": 350, "kicks_conceded": 210,
        "handballs_conceded": 140, "marks_conceded": 90,
        "goals_conceded": 12, "behinds_conceded": 10,
        "tackles_conceded": 55, "hitouts_conceded": 38,
        "inside_50s_conceded": 52, "clearances_conceded": 40,
    }
    df = pd.DataFrame([correct])
    out = fix_conceded_columns(df)
    for k, v in correct.items():
        assert out.iloc[0][k] == v


def test_fix_preserves_column_order():
    from scripts.build_conceded_stats import fix_conceded_columns
    df = pd.DataFrame([_make_corrupt_row()])
    out = fix_conceded_columns(df)
    assert list(out.columns) == EXPECTED_COLUMNS


def test_fix_raises_on_unexplained_row():
    """A row matching neither invariant must raise, not be silently guessed."""
    from scripts.build_conceded_stats import fix_conceded_columns
    bad = _make_corrupt_row()
    bad["disposals_conceded"] = 999  # breaks both invariants
    with pytest.raises(ValueError):
        fix_conceded_columns(pd.DataFrame([bad]))


def test_main_repairs_file_and_is_idempotent(tmp_path):
    """main() on a synthetic corrupt CSV fixes it; second run is a no-op."""
    from scripts.build_conceded_stats import main
    p = tmp_path / "conceded.csv"
    pd.DataFrame([_make_corrupt_row(), _make_corrupt_row()]).to_csv(p, index=False)
    assert main(str(p)) == 0
    fixed = pd.read_csv(p)
    assert (
        fixed["disposals_conceded"]
        == fixed["kicks_conceded"] + fixed["handballs_conceded"]
    ).all()
    # second run: still valid, unchanged
    assert main(str(p)) == 0
    again = pd.read_csv(p)
    pd.testing.assert_frame_equal(fixed, again)


# ---------------------------------------------------------------------------
# 2. Data-integrity tests on the real file
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def conceded():
    return pd.read_csv(CONCEDED_FILE)


def test_columns_exact(conceded):
    assert list(conceded.columns) == EXPECTED_COLUMNS


def test_dtypes(conceded):
    assert conceded["year"].dtype.kind == "i"
    assert conceded["round"].dtype.kind == "i"
    assert conceded["team"].dtype == object
    assert conceded["opponent"].dtype == object
    for c in EXPECTED_COLUMNS[4:]:
        assert conceded[c].dtype.kind == "i", f"{c} should be integer"


def test_no_missing(conceded):
    assert conceded.isna().sum().sum() == 0


def test_disposals_invariant(conceded):
    """disposals = kicks + handballs must hold for every row (physical law)."""
    ok = (
        conceded["disposals_conceded"]
        == conceded["kicks_conceded"] + conceded["handballs_conceded"]
    )
    assert ok.all(), f"invariant fails on {(~ok).sum()} rows"


def test_value_ranges(conceded):
    for col, (lo, hi) in VALUE_RANGES.items():
        assert conceded[col].between(lo, hi).all(), (
            f"{col} outside [{lo},{hi}]: "
            f"min={conceded[col].min()} max={conceded[col].max()}"
        )


def test_goals_behinds_match_official_scores(conceded):
    """goals/behinds conceded must equal the opponent's final score in matches.

    matches_2025 is the authoritative, complete source for scores.
    """
    mt = pd.read_csv(MATCHES_FILE)
    rows = []
    for _, r in mt.iterrows():
        try:
            rnd = int(r["round_num"])
        except (ValueError, TypeError):
            continue  # finals rounds with non-numeric labels
        rows.append((int(r["year"]), rnd, r["team_1_team_name"], r["team_2_team_name"],
                     int(r["team_2_final_goals"]), int(r["team_2_final_behinds"])))
        rows.append((int(r["year"]), rnd, r["team_2_team_name"], r["team_1_team_name"],
                     int(r["team_1_final_goals"]), int(r["team_1_final_behinds"])))
    official = pd.DataFrame(
        rows, columns=["year", "round", "team", "opponent", "g", "b"]
    )
    c = conceded.copy()
    c["round"] = c["round"].astype(int)
    m = c.merge(official, on=["year", "round", "team", "opponent"], how="inner")
    assert len(m) > 50, "too few rows matched to matches -- key mismatch"
    assert (m["goals_conceded"] == m["g"]).all(), "goals_conceded != official"
    assert (m["behinds_conceded"] == m["b"]).all(), "behinds_conceded != official"
