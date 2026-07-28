"""update_eval_surface.sh must keep docs/banner.svg's aria-label in sync with the pills.

The banner carries the same three figures twice: once as visible <text> pills and
once inside the root aria-label (what a screen reader announces). The pill regexes
matched the HTML entity form `&#8211;`; the aria-label in the shipped file held a
LITERAL en-dash, so its regex never matched and the aria-label silently froze at
R1-R13 / MAE 4.020 while the pills tracked the real figures. A screen-reader user
was being read stale accuracy claims.

The invariant under test is agreement, not a hard-coded number: whatever window,
MAE and within-5 the pills show, the aria-label must announce the same. That holds
regardless of which dash form the file happens to use.

Hermetic: the real script + real README/banner are COPIED into tmp_path and driven
against fixture backtest CSVs, so nothing touches the repo's working tree.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "update_eval_surface.sh"
VENV_PYTHON = Path("/home/abhi/sourceCode/python/coding/.venv/bin/python")

# A deliberately stale aria-label; the script must overwrite every part of it.
STALE_MAE = "9.999"
STALE_W5 = "11.1"

pytestmark = pytest.mark.skipif(
    not VENV_PYTHON.exists(), reason="repo venv python not available"
)


def _summary_csv(path, rows):
    header = "year,round,n_players,mae,rmse,pct_within_5,pct_within_10,bias\n"
    body = "".join(
        f"2026,{r},{n},{mae},{rmse},{w5},{w10},{bias}\n"
        for r, n, mae, rmse, w5, w10, bias in rows
    )
    path.write_text(header + body)


def _by_team_csv(path, rows):
    header = "year,round,team,n,bias\n"
    body = "".join(f"2026,{r},{team},{n},{bias}\n" for r, team, n, bias in rows)
    path.write_text(header + body)


def _make_repo(tmp_path, aria_window):
    """Copy the real script/README/banner into a tmp repo, with a stale aria-label.

    aria_window is the round-range text to plant (varies the dash form per test).
    """
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "docs").mkdir(parents=True)
    (repo / "data" / "prediction" / "backtest").mkdir(parents=True)
    (repo / "data" / "player_data").mkdir(parents=True)
    # The script counts player files with `ls ... | wc -l`; under `set -o pipefail`
    # an empty glob makes ls exit non-zero and kills the run, so seed one file.
    # Also needs a pre-target-year row: the script derives the training-corpus
    # season span from the loaded files and fails closed if it finds none.
    (repo / "data" / "player_data" / "smith_john_01011990_performance_details.csv").write_text(
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
    shutil.copy(REPO / "README.md", repo / "README.md")

    svg = (REPO / "docs" / "banner.svg").read_text(encoding="utf-8")
    # Plant a stale aria-label in the dash form under test.
    svg = re.sub(
        r"2026 season R\d+(?:&#8211;|–|-)R\d+: MAE \d+\.\d+, [\d.]+% within 5 disposals",
        f"2026 season {aria_window}: MAE {STALE_MAE}, {STALE_W5}% within 5 disposals",
        svg,
        count=1,
    )
    (repo / "docs" / "banner.svg").write_text(svg, encoding="utf-8")

    bt = repo / "data" / "prediction" / "backtest"
    _summary_csv(
        bt / "backtest_summary_20260720_120000.csv",
        [
            (1, 300, 4.100, 5.30, 70.0, 94.0, 0.10),
            (2, 310, 3.900, 5.10, 75.0, 95.0, -0.05),
            (3, 320, 3.700, 4.90, 78.0, 96.0, 0.02),
        ],
    )
    # Per-team n must reconcile with the summary's n_players (300/310/320): the
    # script now refuses to write when they disagree, because a mismatch means two
    # backtest vintages have been crossed.
    _by_team_csv(
        bt / "backtest_by_team_20260720_120000.csv",
        [
            (1, "Carlton", 150, -0.40),
            (1, "Sydney", 150, 0.35),
            (2, "Carlton", 155, -0.20),
            (2, "Sydney", 155, 0.50),
            (3, "Carlton", 160, -0.10),
            (3, "Sydney", 160, 0.30),
        ],
    )
    return repo


def _run(repo):
    return subprocess.run(
        ["bash", str(repo / "scripts" / "update_eval_surface.sh")],
        cwd=repo,
        capture_output=True,
        text=True,
    )


def _aria(svg):
    m = re.search(
        r"2026 season (R\d+(?:&#8211;|–|-)R\d+): MAE (\d+\.\d+), ([\d.]+)% within 5 disposals",
        svg,
    )
    assert m, "aria-label summary line not found in banner"
    return m.group(1), m.group(2), m.group(3)


def _pills(svg):
    window = re.search(r'<text x="181"[^>]*>(R\d+&#8211;R\d+) &#183; 2026</text>', svg)
    mae = re.search(r'<text x="365"[^>]*>MAE (\d+\.\d+)</text>', svg)
    w5 = re.search(r'<text x="563"[^>]*>([\d.]+)% within 5</text>', svg)
    assert window and mae and w5, "banner pills not found"
    return window.group(1), mae.group(1), w5.group(1)


def _player_counts(svg):
    """(aria-label count, visible Band 1 count) — the label and the band word the
    same fact differently, so each needs its own pattern and each can freeze."""
    aria = re.search(r"130 years of AFL data, ([\d,]+) player files", svg)
    band = re.search(r"130 seasons &#183; ([\d,]+) player files", svg)
    assert aria and band, "player-file counts not found in banner"
    return aria.group(1), band.group(1)


def test_aria_label_player_count_matches_visible_band(tmp_path):
    """The aria-label's player-file count froze at 13,329 while the visible band
    tracked reality — the same defect as the MAE freeze, one line over."""
    repo = _make_repo(tmp_path, "R1–R5")
    assert _run(repo).returncode == 0
    svg = (repo / "docs" / "banner.svg").read_text(encoding="utf-8")
    aria_count, band_count = _player_counts(svg)
    assert aria_count == band_count, (
        f"aria-label claims {aria_count} player files, band shows {band_count}"
    )


def _normalise(window):
    """Compare round windows independent of which dash encoding is used."""
    return re.sub(r"(?:&#8211;|–|-)", "-", window)


@pytest.mark.parametrize(
    "aria_window,dash_form",
    [
        ("R1–R5", "literal en-dash"),   # the shipped-file form — the real bug
        ("R1&#8211;R5", "entity en-dash"),   # regression guard: must keep working
        ("R1-R5", "plain hyphen"),           # tolerated third form
    ],
)
def test_aria_label_matches_pills(tmp_path, aria_window, dash_form):
    repo = _make_repo(tmp_path, aria_window)
    res = _run(repo)
    assert res.returncode == 0, f"script failed ({dash_form}): {res.stderr}"

    svg = (repo / "docs" / "banner.svg").read_text(encoding="utf-8")
    aria_win, aria_mae, aria_w5 = _aria(svg)
    pill_win, pill_mae, pill_w5 = _pills(svg)

    assert aria_mae != STALE_MAE, f"aria-label MAE never updated ({dash_form})"
    assert aria_w5 != STALE_W5, f"aria-label within-5 never updated ({dash_form})"
    assert _normalise(aria_win) == _normalise(pill_win), (
        f"aria-label window {aria_win} disagrees with pill {pill_win} ({dash_form})"
    )
    assert aria_mae == pill_mae, (
        f"aria-label MAE {aria_mae} disagrees with pill {pill_mae} ({dash_form})"
    )
    assert aria_w5 == pill_w5, (
        f"aria-label within-5 {aria_w5} disagrees with pill {pill_w5} ({dash_form})"
    )


def test_aria_label_updates_are_idempotent(tmp_path):
    """Running twice must leave the aria-label matching the pills, not drift."""
    repo = _make_repo(tmp_path, "R1–R5")
    assert _run(repo).returncode == 0
    first = (repo / "docs" / "banner.svg").read_text(encoding="utf-8")
    assert _run(repo).returncode == 0
    second = (repo / "docs" / "banner.svg").read_text(encoding="utf-8")
    assert _aria(first) == _aria(second)
    assert _aria(second)[1] == _pills(second)[1]
