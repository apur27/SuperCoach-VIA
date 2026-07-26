"""Harness wiring contracts.

These assert that fixes actually reached the production execution path. Every
incident below was a correct piece of logic that simply was never called — the
defect class this file exists to catch:

  * P25-F1: update_team_analysis.py's [13/14] step holds the HOF profile
    regeneration AND the hard check_top100_consistency() gate. CLAUDE.md's
    tag-exemption for docs/hall-of-fame-top100.md is justified by that gate — but
    the step was invoked by neither harness script, so the gate never fired and
    stale profile numbers shipped.
  * The FATALed-run backtest orphan: artifact-exists was used as a proxy for
    round-complete, so an aborted cycle's output was published.
  * The lineup allowlist: real pipeline output drifted uncommitted for months.

A unit test that passes while the code is unreachable in production is worthless,
so these check reachability, not behaviour.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
REFRESH = REPO / "refresh_and_rank.sh"
WEEKLY = REPO / "scripts" / "weekly_refresh.sh"


@pytest.fixture(scope="module")
def refresh_src():
    return REFRESH.read_text()


@pytest.fixture(scope="module")
def weekly_src():
    return WEEKLY.read_text()


def _uncommented(src):
    """Drop comment-only lines so a mention in prose can't satisfy a contract."""
    return "\n".join(
        l for l in src.splitlines() if not l.lstrip().startswith("#")
    )


def _pos(src, subcommand):
    """Character offset of a backtest_completeness.py invocation, or -1.

    Tolerant of shell quoting: the two harness scripts spell the path differently
    ("$REPO_ROOT/scripts/..." vs scripts/...), so match the call, not the spelling.
    """
    m = re.search(
        r'backtest_completeness\.py"?\s+--dir\s+\S+\s+' + subcommand, src
    )
    return m.start() if m else -1


# ------------------------------------------- backtest completeness (quarantine)


def test_start_round_uses_completeness_module(refresh_src):
    """The start-round must come from the manifest, not from newest-file-on-disk."""
    live = _uncommented(refresh_src)
    assert "backtest_completeness.py" in live
    assert "last-round" in live


def test_orphaned_artifacts_are_swept_before_backtesting(refresh_src):
    """Sweep must precede the backtest loop, or orphans still steer the start round."""
    live = _uncommented(refresh_src)
    sweep = _pos(live, "sweep")
    last_round = _pos(live, "last-round")
    assert sweep != -1, "no quarantine sweep in the harness"
    assert last_round != -1
    assert sweep < last_round, "sweep must run before the start-round is computed"


def test_old_filename_based_detection_is_gone(refresh_src):
    """The exact line that caused the incident must not survive."""
    live = _uncommented(refresh_src)
    assert not re.search(
        r"ls .*backtest_summary_\*\.csv.*\|\s*sort\s*\|\s*tail", live
    ), "newest-summary-by-filename detection is back"


def test_completion_is_marked_only_after_a_successful_push(refresh_src, weekly_src):
    """Marking before the push would re-create the bug under a later failure."""
    for src, name in ((refresh_src, "refresh_and_rank.sh"), (weekly_src, "weekly_refresh.sh")):
        live = _uncommented(src)
        mark = _pos(live, "mark")
        push = live.find("git push origin main")
        assert mark != -1, f"{name} never marks the cycle complete"
        assert push != -1, f"{name} has no push to gate the mark on"
        assert mark > push, f"{name} marks completion before pushing"


# ---------------------------------------------------------- lineup allowlist


def test_lineups_are_staged(refresh_src):
    """Real pipeline output; the S3 scraper corruption that justified excluding
    them is confined to 700 legacy rows — current output is clean name-form."""
    live = _uncommented(refresh_src)
    assert "data/lineups/" in live


# ------------------------------------------- F3 stays deferred (end-of-season)


def test_top100_yearly_is_not_staged_weekly(refresh_src, weekly_src):
    """Per user decision: top100/yearly regenerates end-of-season, not weekly.

    It must stay OUT of both allowlists so the weekly cycle neither ships nor
    blocks on its per-cycle churn.
    """
    for src in (refresh_src, weekly_src):
        live = _uncommented(src)
        assert "data/top100/yearly" not in live


# -------------------------------------------------- Skeptic gate on insights


def test_skeptic_gates_the_insights_doc(weekly_src):
    live = _uncommented(weekly_src)
    assert "--agent Skeptic" in live, "Skeptic never runs in the weekly cycle"
    assert "skeptic_verdict.py" in live, "Skeptic's verdict is never enforced"


def test_skeptic_runs_after_datasentinel_and_before_commit(weekly_src):
    """Council order: DataSentinel clears the numbers, then Skeptic reads the prose,
    then Phase 4 stages the file."""
    live = _uncommented(weekly_src)
    ds = live.find("gate_insights")
    sk = live.find("--agent Skeptic")
    stage = live.find("git add")
    assert ds != -1 and sk != -1 and stage != -1
    assert ds < sk < stage, "Skeptic must sit between DataSentinel and staging"


# --------------------------------------------------- HOF profile gate (P25-F1)


def test_hof_profile_gate_is_reachable_from_the_harness(refresh_src, tmp_path,
                                                        monkeypatch):
    """check_top100_consistency() backs CLAUDE.md's tag exemption for
    docs/hall-of-fame-top100.md, so it must actually execute in production.

    The reachable path is the harness -> refresh_readme.py -> the gate.
    update_team_analysis.py's main() (which also holds the gate, at step [13/14])
    has never run in production; refresh_readme.py mirrored it and had dropped
    the profile pass, which is exactly why the table refreshed weekly while the
    profile prose froze. Assert the behaviour, not a string in a shell file.

    Both halves now delegate to update_team_analysis.update_top100_hof_doc(), so
    this asserts the call actually happens rather than grepping either file for
    the gate's name — a grep would pass on a copy that is never invoked, and
    would fail on a correct delegation.
    """
    import sys as _sys
    _sys.path.insert(0, str(REPO))
    import update_team_analysis as uta
    import refresh_readme

    assert "refresh_readme.py" in _uncommented(refresh_src), (
        "refresh_readme.py is not invoked by the harness"
    )

    # Every real path this step writes is redirected into tmp_path — the probe
    # must not regenerate assets/charts/* or rewrite docs/hall-of-fame-top100.md.
    bio = tmp_path / "bio.csv"
    scores = tmp_path / "scores.csv"
    hof = tmp_path / "hof.md"
    bio.write_text("Serial Number,Player Name,Footy Teams,Comment\n", encoding="utf-8")
    scores.write_text("player,all_time_score\n", encoding="utf-8")
    # Must carry the marker pair: replace_top100_section now raises rather than
    # silently returning the text unchanged when it matches nothing, so a
    # marker-less probe doc would abort the step before the profile pass and
    # make this wiring probe assert the wrong thing.
    hof.write_text(
        "# x\n\n<!-- ALL-TIME-TOP100-START -->\nold\n<!-- ALL-TIME-TOP100-END -->\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(uta, "TOP100_CSV", str(bio))
    monkeypatch.setattr(uta, "TOP100_SCORES_CSV", str(scores))
    monkeypatch.setattr(uta, "HALL_OF_FAME_PATH", str(hof))
    monkeypatch.setattr(uta, "generate_top100_chart", lambda: None)
    monkeypatch.setattr(uta, "generate_top100_section", lambda: "new table")

    called = []
    monkeypatch.setattr(
        uta, "check_top100_consistency",
        lambda *a, **k: (called.append("gate"), ([], []))[1],
    )
    monkeypatch.setattr(
        uta, "regenerate_top100_profiles",
        lambda text, *a, **k: (called.append("regen"), (text, []))[1],
    )

    refresh_readme._step_top100_markdown()

    assert "regen" in called, (
        "the harness path refreshes the top-100 table without regenerating the "
        "profile prose — the asymmetry that caused the stale-profile drift"
    )
    assert "gate" in called, (
        "the harness path does not run the HOF consistency gate — the gate that "
        "justifies CLAUDE.md's tag exemption would never fire in production"
    )
