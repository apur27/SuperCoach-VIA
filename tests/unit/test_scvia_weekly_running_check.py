"""scvia_weekly.sh refuses to start while a weekly cycle runs. Its check matched the harness script NAMES anywhere
in any command line, so a shell that merely mentioned them (`git add scripts/weekly_refresh.sh && git commit`, an
editor, a grep) counted as a running cycle: the pre-commit hook then failed 19 test_weekly_candidate tests and
blocked the commit (2026-10-06). The check now matches only bash EXECUTING one of those scripts, in every form the
harness is actually launched."""

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "scvia_weekly.sh"


def _pattern() -> str:
    m = re.search(r"^HARNESS_PROC_RE='([^']+)'", SCRIPT.read_text(), re.MULTILINE)
    assert m, "HARNESS_PROC_RE not defined in scvia_weekly.sh"
    return m.group(1)


def _matches(cmdline: str) -> bool:
    r = subprocess.run(["grep", "-Eq", _pattern()], input=cmdline + "\n", text=True)
    return r.returncode == 0


def test_every_real_harness_launch_is_detected():
    for cmd in (
        "bash scripts/weekly_refresh.sh",
        "/usr/bin/bash /home/x/SuperCoach-VIA/scripts/weekly_refresh.sh",
        "bash /home/x/SuperCoach-VIA/refresh_and_rank.sh",
        "bash ./refresh_and_rank.sh",
        "bash refresh_and_rank.sh",
        "bash -x scripts/weekly_refresh.sh",
    ):
        assert _matches(cmd), cmd


def test_a_command_line_that_only_mentions_the_scripts_is_not_a_cycle():
    for cmd in (
        "/bin/bash -c git add scripts/weekly_refresh.sh && git commit -m x",
        "git add scripts/weekly_refresh.sh refresh_and_rank.sh",
        "vim scripts/weekly_refresh.sh",
        "grep -n x refresh_and_rank.sh",
        "bash scripts/weekly_refresh.sh.bak",
        "pgrep -af scripts/weekly_refresh.sh|refresh_and_rank.sh",
    ):
        assert not _matches(cmd), cmd


def test_the_running_check_uses_the_pattern():
    src = SCRIPT.read_text()
    assert 'pgrep -f "$HARNESS_PROC_RE"' in src
    assert "pgrep -af 'scripts/weekly_refresh.sh|refresh_and_rank.sh'" not in src


def test_the_candidate_smoke_uses_the_same_narrow_check():
    src = (REPO / "scripts" / "smoke_scvia_candidate.sh").read_text()
    assert "pgrep -af 'scripts/weekly_refresh.sh|refresh_and_rank.sh'" not in src
    assert f"HARNESS_PROC_RE='{_pattern()}'" in src and 'pgrep -f "$HARNESS_PROC_RE"' in src
