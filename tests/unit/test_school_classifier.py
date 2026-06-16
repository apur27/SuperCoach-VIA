"""
Unit tests for scrapers/school_classifier.py.

Pure string + CSV logic -- no network. Covers extract_school (pathway-chain
parsing), classify_school (school -> affiliation lookup), and
build_school_affiliation (end-to-end DataFrame build over a tmp CSV).
"""

import pandas as pd

from scrapers.school_classifier import (
    build_school_affiliation,
    classify_school,
    extract_school,
)


# ---------------------------------------------------------------------------
# extract_school
# ---------------------------------------------------------------------------

def test_extract_school_simple():
    assert extract_school("Haileybury College / Dandenong U18") == "Haileybury College"


def test_extract_school_first_match():
    assert extract_school("Wesley College (WA) / Perth") == "Wesley College (WA)"


def test_extract_school_multi_segment():
    assert extract_school("Dowerin / Wesley College (WA) / Perth") == "Wesley College (WA)"


def test_extract_school_no_school():
    assert extract_school("Kyabram / Murray U18") is None


def test_extract_school_none_input():
    assert extract_school(None) is None


def test_extract_school_empty_string():
    assert extract_school("") is None


# ---------------------------------------------------------------------------
# classify_school
# ---------------------------------------------------------------------------

def test_classify_aps():
    assert classify_school("Haileybury College") == "APS"
    assert classify_school("Carey Grammar") == "APS"


def test_classify_gps_wa():
    assert classify_school("Wesley College (WA)") == "GPS_WA"


def test_classify_gps_qld():
    assert classify_school("Brisbane Grammar") == "GPS_QLD"


def test_classify_other_private():
    assert classify_school("Emmanuel College") == "other_private"


def test_classify_unknown():
    # "Wonthaggi High School" contains "High School" -> state
    assert classify_school("Wonthaggi High School") == "state"


def test_classify_unknown_school():
    assert classify_school("Foobar Academy") == "unknown"


def test_classify_case_insensitive():
    assert classify_school("haileybury college") == "APS"


# ---------------------------------------------------------------------------
# build_school_affiliation
# ---------------------------------------------------------------------------

def test_build_affiliation_dataframe(tmp_path):
    enrichment = tmp_path / "enrichment.csv"
    out = tmp_path / "schools.csv"

    pd.DataFrame(
        {
            "year": [2004, 2005, 2006],
            "pick": [1, 2, 3],
            "player_name": ["Alpha One", "Beta Two", "Gamma Three"],
            "original_club": [
                "Haileybury College / Dandenong U18",   # APS
                "Wesley College (WA) / Perth",          # GPS_WA
                "Kyabram / Murray U18",                 # no school
            ],
            "grade": ["A", "B", "C"],
            "games": [100, 50, 10],
            "goals": [20, 5, 1],
        }
    ).to_csv(enrichment, index=False)

    df = build_school_affiliation(str(enrichment), str(out))

    expected_cols = [
        "year", "pick", "player_name", "original_club",
        "extracted_school", "school_type", "grade", "games", "goals",
    ]
    assert list(df.columns) == expected_cols
    # File written with the same columns.
    written = pd.read_csv(out)
    assert list(written.columns) == expected_cols

    assert df.loc[0, "extracted_school"] == "Haileybury College"
    assert df.loc[0, "school_type"] == "APS"
    assert df.loc[1, "extracted_school"] == "Wesley College (WA)"
    assert df.loc[1, "school_type"] == "GPS_WA"
    # No school in the chain -> extracted is null, type is "unknown".
    assert pd.isna(df.loc[2, "extracted_school"])
    assert df.loc[2, "school_type"] == "unknown"
