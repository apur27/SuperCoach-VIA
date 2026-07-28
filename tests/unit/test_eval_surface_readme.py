"""update_eval_surface.sh — the README stat surfaces it silently failed to update.

Three defects, all found by reconciling README figures against the backtest CSVs:

1. STALE TEAM VINTAGE. The per-team merge deduped on (year, round, team) keeping
   last. That does not supersede a round when the newer run covers FEWER teams:
   R18 2026 was re-scored from an archived forward CSV that predated the full
   fixture (14 teams, not 18), so four teams survived from the older file and
   injected 92 phantom player-rounds. One of them carried bias -3.26 over n=23,
   which alone dragged St Kilda's published season figure from -0.583 to -0.733.
   The published sentence was not merely stale, it was an incoherent hybrid: the
   headline came from one vintage and four teams of the table from another.

2. UNICODE MINUS. The aggregate-bias row is written with U+2212 MINUS SIGN, but
   the pattern accepted only ASCII `[-+]`, so re.sub matched nothing, returned the
   input unchanged, and raised no error — freezing that cell at its R1-R13 value
   while the same metric updated elsewhere in the file. Same silent-no-op class as
   the banner aria-label dash.

3. UNTEMPLATED ROUND COUNT. "all N rounds" was never regenerated at all.

The n-reconciliation assertion is the cheap detector for (1): the per-team table's
n must sum to the summary's n_players. When it does not, a vintage is crossed.
"""
import os
import re
import shutil
import subprocess

import pytest

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "update_eval_surface.sh"
VENV_PYTHON = Path("/home/abhi/sourceCode/python/coding/.venv/bin/python")

pytestmark = pytest.mark.skipif(
    not VENV_PYTHON.exists(), reason="repo venv python not available"
)

SUMMARY_HEADER = "year,round,n_players,mae,rmse,pct_within_5,pct_within_10,bias\n"
TEAM_HEADER = "year,round,team,n,bias\n"


def _write(path, text, mtime):
    path.write_text(text)
    os.utime(path, (mtime, mtime))


def _make_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "docs").mkdir(parents=True)
    bt = repo / "data" / "prediction" / "backtest"
    bt.mkdir(parents=True)
    (repo / "data" / "player_data").mkdir(parents=True)
    # Needs at least one pre-target-year row: the script derives the training-corpus
    # season span from the loaded files and fails closed if it finds none.
    (repo / "data" / "player_data" / "a_b_01011990_performance_details.csv").write_text(
        "year,round,disposals\n2025,1,19\n2026,1,20\n"
    )
    # The script locates the two corpus filters in prediction.py by source anchor
    # rather than by hard-coded line number, so the file has to exist.
    (repo / "supercoach").mkdir(parents=True)
    (repo / "supercoach" / "prediction.py").write_text(
        "historical_data = df[df['year'] < self.target_year].copy()\n"
        "birth_year_threshold = self.target_year - 40\n"
    )
    shutil.copy(SCRIPT, repo / "scripts" / "update_eval_surface.sh")
    # Plant the U+2212 form rather than relying on the live README to carry it:
    # the fix rewrites that cell in ASCII, so the shipped file no longer reproduces
    # the bug and a fixture that copied it verbatim would silently stop testing it.
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    readme = re.sub(
        r"(\| Aggregate bias \| \*\*\[data\]\*\* )[-+−]?[\d.]+",
        "\\g<1>\u22120.093", readme, count=1)
    readme = re.sub(r"(Full per-round table \(all )\d+( rounds\))",
                    r"\g<1>13\g<2>", readme, count=1)
    (repo / "README.md").write_text(readme, encoding="utf-8")
    shutil.copy(REPO / "docs" / "banner.svg", repo / "docs" / "banner.svg")

    # Summary: round 1 (n=100) and round 2 (n=80). Weighted bias = -0.022.
    _write(
        bt / "backtest_summary_20260101_000000.csv",
        SUMMARY_HEADER + "2026,1,100,4.0,5.0,70.0,94.0,0.2\n2026,2,80,3.5,4.5,78.0,96.0,-0.3\n",
        mtime=1_000_000,
    )

    # Round-1 per-team rows (n sums to 100).
    _write(
        bt / "backtest_by_team_20260101_000000.csv",
        TEAM_HEADER + "2026,1,Sydney,50,0.2\n2026,1,Geelong,50,0.2\n",
        mtime=1_000_000,
    )
    # OLDER round-2 vintage: 4 teams, n=100. Carlton carries an extreme bias.
    _write(
        bt / "backtest_by_team_20260102_000000.csv",
        TEAM_HEADER
        + "2026,2,Sydney,25,0.1\n2026,2,Geelong,25,-0.5\n"
        + "2026,2,Carlton,25,-3.0\n2026,2,Melbourne,25,0.0\n",
        mtime=2_000_000,
    )
    # NEWER round-2 vintage: only 2 teams, n=80 — matches the summary.
    _write(
        bt / "backtest_by_team_20260103_000000.csv",
        TEAM_HEADER + "2026,2,Sydney,40,0.1\n2026,2,Geelong,40,-0.5\n",
        mtime=3_000_000,
    )
    return repo


def _run(repo):
    return subprocess.run(
        ["bash", str(repo / "scripts" / "update_eval_surface.sh")],
        cwd=repo, capture_output=True, text=True,
    )


def _readme(repo):
    return (repo / "README.md").read_text(encoding="utf-8")


def _team_sentence(md):
    m = re.search(r"Team-level signed bias spans .*?disposals\.", md, re.DOTALL)
    assert m, "team-bias sentence not found"
    return m.group(0)


# --------------------------------------------------- 1. stale team vintage


def test_superseded_team_vintage_is_not_merged(tmp_path):
    """Only the newest FILE per (year, round) may contribute team rows.

    Carlton exists solely in the superseded round-2 vintage. If it reaches the
    output it becomes "most under-predicted" on a -3.0 bias that the summary does
    not know about — exactly the St Kilda -0.733 defect.
    """
    repo = _make_repo(tmp_path)
    res = _run(repo)
    assert res.returncode == 0, res.stderr

    sentence = _team_sentence(_readme(repo))
    assert "Carlton" not in sentence, (
        f"superseded vintage leaked into the team table: {sentence}"
    )
    assert "Melbourne" not in sentence, f"superseded vintage leaked: {sentence}"
    # Geelong (-0.111) is the true most-under-predicted of the surviving teams.
    assert "Geelong" in sentence, sentence


def test_team_rows_reconcile_with_the_summary(tmp_path):
    """The detector: per-team n must sum to the summary's n_players."""
    repo = _make_repo(tmp_path)
    res = _run(repo)
    assert res.returncode == 0, res.stderr
    assert "team-n reconciliation" not in res.stderr


def test_unreconcilable_team_rows_fail_closed(tmp_path):
    """A genuine vintage crossing must abort, not publish a hybrid sentence."""
    repo = _make_repo(tmp_path)
    bt = repo / "data" / "prediction" / "backtest"
    # A round present per-team but absent from the summary: unreconcilable.
    _write(
        bt / "backtest_by_team_20260104_000000.csv",
        TEAM_HEADER + "2026,7,Hawthorn,30,0.4\n",
        mtime=4_000_000,
    )
    res = _run(repo)
    assert res.returncode != 0
    assert "reconcil" in (res.stdout + res.stderr).lower()


# ------------------------------------------------------- 2. unicode minus


def test_aggregate_bias_row_with_unicode_minus_is_updated(tmp_path):
    """The row is authored with U+2212; the pattern must not require ASCII."""
    repo = _make_repo(tmp_path)
    assert "−" in _readme(repo), "fixture should start with a U+2212 bias row"
    res = _run(repo)
    assert res.returncode == 0, res.stderr

    md = _readme(repo)
    row = re.search(r"\| Aggregate bias \| \*\*\[data\]\*\* ([^|]+)\|", md)
    assert row, "aggregate-bias row not found"
    value = row.group(1).strip()
    assert "0.093" not in value, f"bias row still frozen at its old value: {value!r}"
    # Weighted bias over the fixture = (100*0.2 + 80*-0.3) / 180 = -0.0222.
    assert "0.022" in value, f"unexpected bias value: {value!r}"


# -------------------------------------------------------- 3. round count


def test_round_count_phrase_is_regenerated(tmp_path):
    repo = _make_repo(tmp_path)
    res = _run(repo)
    assert res.returncode == 0, res.stderr
    md = _readme(repo)
    assert "all 13 rounds" not in md, "round count still frozen at 13"
    assert "all 2 rounds" in md, "round count not regenerated from the data"
