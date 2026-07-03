"""Validate the read-only gate config files promoted out of agent memory (Q7).

These files are consulted by DataSentinel/Skeptic as the canonical, reviewed source
for the coach-anonymity list and the FanFooty schema facts. If they are malformed or
empty, the gate silently loses its teeth — so guard their shape here.
"""
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
COACH_NAMES = REPO / "config" / "coach_names.txt"
FANFOOTY = REPO / "config" / "fanfooty_schema.yaml"
STAT_COVERAGE = REPO / "config" / "stat_coverage_eras.yaml"


def test_coach_names_exists_and_has_entries():
    assert COACH_NAMES.is_file()
    names = [
        ln.strip() for ln in COACH_NAMES.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    assert len(names) >= 5, "coach-anonymity list must carry real names to be load-bearing"
    # names carried over from the anonymity-lint memory must be present
    assert "Hardwick" in names
    assert "Lyon" in names


def test_fanfooty_schema_parses_and_has_required_keys():
    data = yaml.safe_load(FANFOOTY.read_text())
    for key in ("unreliable_fields", "unavailable_fields", "reliable_fields"):
        assert key in data, f"fanfooty_schema.yaml missing '{key}'"
    # the three known-bad fields must be flagged unreliable
    assert set(data["unreliable_fields"]) == {"goals", "behinds", "clangers"}
    # unavailable fields must include the three that are absent from the snapshot
    assert {"inside_50s", "clearances", "contested_possessions"} <= set(data["unavailable_fields"])
    # reliable fields must include core possession stats that agree with afltables
    assert {"kicks", "handballs", "marks", "tackles"} <= set(data["reliable_fields"])


def test_unreliable_and_reliable_fields_do_not_overlap():
    data = yaml.safe_load(FANFOOTY.read_text())
    assert not (set(data["unreliable_fields"]) & set(data["reliable_fields"]))
    assert not (set(data["unavailable_fields"]) & set(data["reliable_fields"]))


def test_stat_coverage_eras_parses_and_is_well_formed():
    assert STAT_COVERAGE.is_file()
    data = yaml.safe_load(STAT_COVERAGE.read_text())
    assert "stats" in data, "stat_coverage_eras.yaml missing top-level 'stats' key"
    stats = data["stats"]
    assert len(stats) >= 12, "coverage table must carry the core metrics to be load-bearing"
    for name, entry in stats.items():
        assert "recorded_from" in entry, f"{name} missing 'recorded_from'"
        year = entry["recorded_from"]
        assert isinstance(year, int), f"{name} recorded_from must be an int year"
        assert 1897 <= year <= 2100, f"{name} recorded_from {year} out of plausible range"


def test_stat_coverage_eras_known_anchors():
    # Anchors verified in the Scientist coverage memory (aggregated across all
    # player_data CSVs): the AFL-stats era starts 1965; goals span the whole set.
    stats = yaml.safe_load(STAT_COVERAGE.read_text())["stats"]
    assert stats["kicks"]["recorded_from"] == 1965
    assert stats["goals"]["recorded_from"] == 1897
