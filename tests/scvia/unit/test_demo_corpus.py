"""Demo corpus: deterministic, DEMO-labelled, contains the required edge cases."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from supercoach_via.demo import write_demo_corpus


def _digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.csv")):
        h.update(p.relative_to(root).as_posix().encode() + p.read_bytes())
    return h.hexdigest()


def test_deterministic(tmp_path: Path) -> None:
    write_demo_corpus(tmp_path / "a")
    write_demo_corpus(tmp_path / "b")
    assert _digest(tmp_path / "a") == _digest(tmp_path / "b")


def test_edge_cases_present_and_labelled(tmp_path: Path) -> None:
    counts = write_demo_corpus(tmp_path)
    assert counts["scheduled"] >= 3 and counts["matches"] > 60
    data = tmp_path / "data"
    names = {p.name for p in (data / "player_data").iterdir()}
    same = [n for n in names if n.startswith("samename_demo_") and n.endswith("performance_details.csv")]
    assert len(same) == 2
    with (data / "matches" / "matches_2025.csv").open() as fh:
        gfs = [r for r in csv.DictReader(fh) if r["round_num"] == "Grand Final"]
    assert len(gfs) == 2 and gfs[0]["team_1_final_goals"] == gfs[0]["team_2_final_goals"]
    with (data / "matches" / "matches_2026.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert any(r["team_1_final_goals"] == "" for r in rows)
    for r in rows:
        assert r["team_1_team_name"].startswith("Demo") and r["team_2_team_name"].startswith("Demo")
    with (data / "matches" / "matches_2024.csv").open() as fh:
        r24 = list(csv.DictReader(fh))
    r2 = min(r["date"] for r in r24 if r["round_num"] == "2")
    r4 = min(r["date"] for r in r24 if r["round_num"] == "4")
    assert r2 > r4  # postponed low-numbered round


def test_demo_validates_under_the_default_policy(tmp_path: Path) -> None:
    """An honest demo: team goal totals equal the sum of player goals, so the default
    (current-season-blocking) validation passes without a relaxed policy."""
    from supercoach_via.ingest.legacy import import_legacy
    from supercoach_via.ingest.reconcile import load_policy, validate_dataset
    from supercoach_via.settings import RunContext, Settings

    write_demo_corpus(tmp_path / "src")
    ctx = RunContext(settings=Settings(data_root=tmp_path / "var", source_root=tmp_path / "src"))
    report = validate_dataset(import_legacy(tmp_path / "src", ctx), load_policy())
    blocking = [i for i in report.issues if i["severity"] == "blocking"]
    assert report.ok, blocking[:3]
    assert not [i for i in report.issues if i["rule_id"] == "match_player_goals_mismatch"]
