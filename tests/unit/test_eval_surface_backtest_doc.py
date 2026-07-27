"""update_eval_surface.sh must regenerate the frozen blocks in docs/afl-backtest-2026.md.

That doc published five mutually inconsistent figure sets because nothing regenerated
them: a cumulative summary frozen at R1-R13 (whose values could no longer be
reproduced from any vintage on disk — they came from artifacts since deleted), a
team-bias table frozen at R1-R13, and a notable-misses table frozen at R1-R11, all
sitting under an auto-generated per-round table that tracked R1-R20.

Three traps this file pins, each of which produces a plausible-looking wrong answer:

  * RMSE must be pooled in SQUARED space. Feeding it through the same n-weighted
    mean used for MAE/bias/within-5/within-10 gives 5.0843 where the correct pooled
    value is 5.0937 — close enough to look right and be wrong.
  * The per-round table's closing line is an UNWEIGHTED mean across rounds and is
    correct as labelled. The cumulative block is player-weighted. They are different
    statistics; "fixing" the doc by making them agree would be the actual defect.
  * Per-team rows must supersede by whole FILE per (year, round). A per-team dedupe
    keeps rows from a superseded vintage covering fewer teams.

Hermetic: the real script is copied into tmp_path and driven against fixture CSVs.
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
DETAIL_HEADER = ("player,team,round,year,predicted_disposals,actual_disposals,"
                 "error,abs_error\n")


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
    (repo / "data" / "player_data" / "a_b_01011990_performance_details.csv").write_text(
        "year,round,disposals\n2026,1,20\n"
    )
    shutil.copy(SCRIPT, repo / "scripts" / "update_eval_surface.sh")
    shutil.copy(REPO / "README.md", repo / "README.md")
    shutil.copy(REPO / "docs" / "banner.svg", repo / "docs" / "banner.svg")
    shutil.copy(REPO / "docs" / "afl-backtest-2026.md", repo / "docs" / "afl-backtest-2026.md")

    # Two rounds, n=10 each so the per-player detail below reconciles exactly.
    # n-weighted MAE  = (10*4 + 10*3)/20 = 3.500
    # n-weighted bias = (10*0.2 + 10*-0.4)/20 = -0.100
    # RMSE pooled in squared space = sqrt((10*6^2 + 10*4^2)/20) = sqrt(26) = 5.0990.
    # The WRONG linear mean would be (6+4)/2 = 5.0000 — which is why 5.099 is asserted.
    _write(bt / "backtest_summary_20260101_000000.csv",
           SUMMARY_HEADER
           + "2026,1,10,4.0,6.0,70.0,90.0,0.2\n"
           + "2026,2,10,3.0,4.0,80.0,96.0,-0.4\n",
           mtime=1_000_000)

    _write(bt / "backtest_by_team_20260101_000000.csv",
           TEAM_HEADER + "2026,1,Carlton,5,-0.9\n2026,1,Sydney,5,1.3\n",
           mtime=1_000_000)
    # OLDER round-2 vintage: 3 teams. Hawthorn exists ONLY here.
    _write(bt / "backtest_by_team_20260102_000000.csv",
           TEAM_HEADER
           + "2026,2,Carlton,4,-0.5\n2026,2,Sydney,3,0.4\n2026,2,Hawthorn,3,-9.0\n",
           mtime=2_000_000)
    # NEWER round-2 vintage: 2 teams, n=100, reconciles with the summary.
    _write(bt / "backtest_by_team_20260103_000000.csv",
           TEAM_HEADER + "2026,2,Carlton,5,-0.5\n2026,2,Sydney,5,0.4\n",
           mtime=3_000_000)

    # Per-player detail for the notable-misses table.
    rows = []
    for i, (pred, act) in enumerate([(10, 30), (12, 28), (14, 27), (11, 23), (13, 24),
                                     (30, 10), (28, 12), (27, 14), (23, 11), (24, 13)]):
        nm = "Ah Chee Callum" if i == 0 else f"Surname{i} First{i}"
        rows.append(f"{nm},Carlton,1,2026,{pred},{act},{pred-act},{abs(pred-act)}")
    _write(bt / "prediction_vs_actual_round_1_2026_20260101_000000.csv",
           DETAIL_HEADER + "\n".join(rows) + "\n", mtime=1_000_000)
    rows2 = [f"Other{i},Sydney,2,2026,20,{20 - i},{i},{abs(i)}" for i in range(10)]
    _write(bt / "prediction_vs_actual_round_2_2026_20260101_000000.csv",
           DETAIL_HEADER + "\n".join(rows2) + "\n", mtime=1_000_000)
    return repo


def _run(repo):
    return subprocess.run(["bash", str(repo / "scripts" / "update_eval_surface.sh")],
                          cwd=repo, capture_output=True, text=True)


def _doc(repo):
    return (repo / "docs" / "afl-backtest-2026.md").read_text(encoding="utf-8")


def _block(md, name):
    m = re.search(rf"<!-- {name}-START -->(.*?)<!-- {name}-END -->", md, re.DOTALL)
    assert m, f"{name} block not found"
    return m.group(1)


# ------------------------------------------------------------- cumulative


def test_cumulative_block_is_regenerated(tmp_path):
    repo = _make_repo(tmp_path)
    res = _run(repo)
    assert res.returncode == 0, res.stderr
    block = _block(_doc(repo), "CUMULATIVE")
    assert "4,806" not in block, "still frozen at the R1-R13 pool"
    assert "7,153" not in block, "hard-coded a real-repo figure into the generator"
    assert "| **20** |" in block, f"player count not regenerated: {block}"
    assert "3.500" in block, f"MAE not n-weighted: {block}"
    assert "-0.100" in block or "−0.100" in block, f"bias not n-weighted: {block}"


def test_rmse_is_pooled_in_squared_space(tmp_path):
    """THE TRAP: a linear weighted mean gives 5.000; the correct pooled RMSE is 5.099."""
    repo = _make_repo(tmp_path)
    assert _run(repo).returncode == 0
    block = _block(_doc(repo), "CUMULATIVE")
    assert "5.099" in block, (
        f"RMSE appears to be linearly averaged rather than pooled in squared space: {block}"
    )
    assert "5.000" not in block


def test_round_window_and_count_are_regenerated(tmp_path):
    repo = _make_repo(tmp_path)
    assert _run(repo).returncode == 0
    block = _block(_doc(repo), "CUMULATIVE")
    assert "13 (R1–R13)" not in block
    assert "2 (R1–R2)" in block, block


def test_per_round_mean_line_is_left_alone(tmp_path):
    """The auto-block's unweighted mean is correct as labelled and must not be
    'reconciled' into agreement with the player-weighted figures."""
    repo = _make_repo(tmp_path)
    before = _doc(repo)
    keep = re.search(r"\*\*Overall \(mean across \d+ rounds\).*", before).group(0)
    assert _run(repo).returncode == 0
    assert keep in _doc(repo), "the mean-of-rounds line was rewritten"


# -------------------------------------------------------------- team bias


def test_team_block_supersedes_by_file(tmp_path):
    repo = _make_repo(tmp_path)
    assert _run(repo).returncode == 0
    block = _block(_doc(repo), "TEAMBIAS")
    assert "Hawthorn" not in block, (
        f"a superseded vintage leaked into the team table: {block}"
    )
    assert "Carlton" in block and "Sydney" in block


def test_causal_prose_is_not_regenerated(tmp_path):
    """Deleted deliberately: it made causal claims from associational data."""
    repo = _make_repo(tmp_path)
    assert _run(repo).returncode == 0
    md = _doc(repo)
    assert "outperforming the model's pre-2026 expectations" not in md
    assert "Recurring names worth noting" not in md


# ---------------------------------------------------------- notable misses


def test_misses_block_is_regenerated_from_detail_csvs(tmp_path):
    repo = _make_repo(tmp_path)
    assert _run(repo).returncode == 0
    block = _block(_doc(repo), "MISSES")
    assert "Nick Daicos" not in block, "still the frozen R1-R11 table"
    assert "Callum Ah Chee" in block, f"under-predictions not regenerated: {block}"
    assert "First5 Surname5" in block, f"over-predictions not regenerated: {block}"


# ------------------------------------------------------------ reconciliation


def test_unreconcilable_pool_fails_closed(tmp_path):
    """The three-way invariant is the gate: summary, per-team and pooled detail
    must agree, or nothing is written."""
    repo = _make_repo(tmp_path)
    bt = repo / "data" / "prediction" / "backtest"
    _write(bt / "backtest_by_team_20260104_000000.csv",
           TEAM_HEADER + "2026,9,Geelong,40,0.1\n", mtime=4_000_000)
    res = _run(repo)
    assert res.returncode != 0
    assert "reconcil" in (res.stdout + res.stderr).lower()


def test_misses_table_uses_natural_name_order(tmp_path):
    """The CSVs store "Surname Firstname"; readers expect "Firstname Surname".

    The hand-written table this replaced read naturally, so a regenerator that
    emits raw CSV order would be a visible presentation regression. Firstname is
    the LAST token, which keeps multi-token surnames ("Ah Chee", "Wanganeen-Milera")
    intact rather than splitting on the first space.
    """
    repo = _make_repo(tmp_path)
    assert _run(repo).returncode == 0
    block = _block(_doc(repo), "MISSES")
    assert "Callum Ah Chee" in block, f"multi-token surname mangled: {block}"
    assert "Ah Chee Callum" not in block, "raw CSV name order leaked into the doc"
