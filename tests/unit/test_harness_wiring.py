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
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
REFRESH = REPO / "refresh_and_rank.sh"
WEEKLY = REPO / "scripts" / "weekly_refresh.sh"


@pytest.mark.parametrize(
    ("script", "count"),
    [(REFRESH, 40), (REPO / "scripts" / "smoke_harness.sh", 15)],
)
def test_harness_log_tail_uses_portable_count_option(script, count):
    """uutils tail rejects obsolete -40/-15 syntax used on failure paths."""
    option = re.search(r"\btail\s+(-\S+)", _uncommented(script.read_text()))
    assert option is not None
    assert option[1] == f"-n{count}"
    lines = [f"line {n}\n" for n in range(50)]
    result = subprocess.run(
        ["tail", option[1]], input="".join(lines), text=True,
        capture_output=True, check=True,
    )
    assert result.stdout == "".join(lines[-count:])


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


def test_gated_backtest_doc_is_reverified_before_staging(refresh_src):
    """A gated doc that the harness regenerates must have a re-verify hop.

    docs/afl-backtest-2026.md is regenerated every cycle and carries a stamp the
    pre-commit gate cross-checks against a content-hash-keyed DataSentinel record.
    Regeneration orphans that record, so without a re-verify step the commit is
    refused and the harness dies with a round of scraped data uncommitted — which
    is exactly what happened on 2026-07-27. The HOF hub and afl-insights.md have
    this hop; this page was gated without one.
    """
    live = _uncommented(refresh_src)
    gate = live.find("afl-backtest-2026.md")
    stage = live.find("git add")
    assert gate != -1, "no re-verification of the gated backtest doc"
    assert "--agent DataSentinel" in live, "re-verification does not invoke DataSentinel"
    assert gate < stage, "the doc is staged before it is re-verified"


def test_weekly_writes_a_terminal_status_on_failure(weekly_src):
    """`last_refresh_complete.json` is written only on success, so a run that dies
    mid-phase leaves the previous cycle's sentinel and looks identical to a run
    still in progress. That is why the 2026-07-27 failure sat undetected for hours."""
    live = _uncommented(weekly_src)
    assert "last_refresh_status.json" in live, "no terminal status marker"
    assert "trap _write_terminal_status EXIT" in live, (
        "the status marker is not on an EXIT trap, so it will not fire on failure"
    )


def test_backtest_doc_is_fully_regenerated_before_it_is_gated(refresh_src):
    """docs/afl-backtest-2026.md has TWO generators; both must run before the gate.

    refresh_readme.py writes the per-round and top-30 tables; update_eval_surface.sh
    writes the CUMULATIVE/TEAMBIAS/MISSES blocks. On 2026-07-27 only the first ran
    before the Phase-1 commit, so the page claimed 21 rounds in one table while the
    pooled figures below it were still at 20 — and because the doc is not in the
    Phase-4 allowlist, the later regeneration would never have been committed.
    """
    live = _uncommented(refresh_src)
    readme_gen = live.find("refresh_readme.py")
    eval_gen = live.find("update_eval_surface.sh")
    gate = live.find("afl-backtest-2026.md")
    stage = live.find("git add")
    assert eval_gen != -1, "update_eval_surface.sh never runs before the Phase-1 commit"
    assert readme_gen < eval_gen < gate < stage, (
        "generator/gate/stage order is wrong: both generators must precede the gate, "
        "and the gate must precede staging"
    )


def test_backtest_doc_is_staged_before_it_is_gated(refresh_src):
    """check-council-stamp.sh verifies the STAGED blob, so the doc must be staged
    before the gate runs.

    It was not, so the gate read the index copy — still the previous commit's
    content — while DataSentinel hashed the regenerated worktree copy. The hop
    reported success against bytes that were not the ones being shipped, and the
    real commit gate then blocked on the ones that were.
    """
    live = _uncommented(refresh_src)
    stage_doc = live.find("git add docs/afl-backtest-2026.md")
    gate = live.find("--agent DataSentinel")
    assert stage_doc != -1, "the gated doc is never staged before verification"
    assert stage_doc < gate, "the doc is gated before it is staged (wrong bytes verified)"


def test_harness_scripts_do_not_hardcode_the_repo_root(refresh_src, weekly_src):
    """A hardcoded absolute repo path makes worktree isolation an illusion.

    refresh_and_rank.sh pinned REPO_ROOT=/home/abhi/git/SuperCoach-VIA and cd'd to
    it, so running the harness from a git worktree still read and WROTE the real
    repository. The smoke runner built for CLAUDE.md 6.2 was mutating live assets/
    while reporting itself sandboxed — caught on its first real run.
    """
    import re as _re
    for src, name in ((refresh_src, "refresh_and_rank.sh"), (weekly_src, "weekly_refresh.sh")):
        for line in _uncommented(src).splitlines():
            m = _re.match(r'\s*REPO_ROOT=(.*)', line)
            if m and "/home/" in m.group(1) and "dirname" not in m.group(1):
                raise AssertionError(
                    f"{name} hardcodes REPO_ROOT ({line.strip()}); derive it from "
                    f"BASH_SOURCE so a worktree run cannot touch the real repo"
                )


# ------------------------------------------- AFL Tables reconciliation gate
def test_reconciliation_gate_runs_on_the_scrape_before_the_phase1_push(weekly_src):
    """The changed seasons are re-audited against AFL Tables after the scrape is
    committed locally and BEFORE anything reaches origin, so a bad row never ships."""
    src = _uncommented(weekly_src)
    gate = src.find("scripts/reconciliation_gate.py")
    assert gate != -1, "reconciliation_gate.py is never invoked by the weekly harness"
    assert src.find("match_completeness_gate.py") < gate < src.find("push origin main")
    call = src[gate:src.find("\n", gate)]
    assert "--fix" in call and "--data-root" in call
    # fail closed: a blocking verdict exits before the push
    after = src[gate:src.find("push origin main")]
    assert "exit 1" in after
    # source-backed corrections are committed through the serialising wrapper
    assert "git_commit_safe.sh commit" in after


def test_reconciliation_corrections_commit_only_what_the_gate_changed(weekly_src):
    """Smoke run 2026-10-06: the commit step fired on ANY dirty data file, so with nothing corrected it still
    logged "corrections committed" and would have swept Phase 1 leftovers into that commit. The state is taken
    before and after the gate, and the commit runs only when the gate itself changed files."""
    src = _uncommented(weekly_src)
    gate = src.find("scripts/reconciliation_gate.py")
    before = src.rfind("RECON_BEFORE=", 0, gate)
    assert before != -1, "no pre-gate snapshot of the data dirs"
    after = src[gate:src.find("push origin main")]
    assert '"$RECON_BEFORE"' in after, "the commit is not conditioned on the gate's own changes"


def test_reconciliation_gate_log_reflects_its_recorded_decision(weekly_src):
    """Gaffer H1: a skipped or warned gate must not log "passed". The harness reads the decision the gate
    persisted and logs skip / warn / pass distinctly."""
    src = _uncommented(weekly_src)
    gate = src.find("scripts/reconciliation_gate.py")
    block = src[gate:src.find("push origin main")]
    assert "reconciliation_gate_status.json" in block
    for word in ("SKIPPED", "WARN"):
        assert word in block, f"no distinct log line for a {word.lower()} gate"


def test_data_changed_by_fix_is_re_gated_before_it_is_committed(weekly_src):
    """Gaffer M2: --fix can add, replace or delete rows after the phantom-row and match-completeness gates ran.
    Both gates run again on the corrected tree before the corrections commit."""
    src = _uncommented(weekly_src)
    gate = src.find("scripts/reconciliation_gate.py")
    block = src[gate:src.find("push origin main")]
    commit = block.find("git_commit_safe.sh commit")
    assert commit != -1
    for check in ("phantom_row_validator.py", "match_completeness_gate.py"):
        at = block.find(check)
        assert at != -1 and at < commit, f"{check} is not re-run before the corrections commit"


def test_python_ci_installs_node_and_browser_contract_dependencies():
    """Python-produced releases must be parsed by the real TypeScript readers in CI."""
    workflow = (REPO / ".github/workflows/scvia-ci.yml").read_text()
    python_job = workflow.split("  python:", 1)[1].split("  web:", 1)[0]
    assert "actions/setup-node@820762786026740c76f36085b0efc47a31fe5020" in python_job
    assert 'node-version-file: ".node-version"' in python_job
    assert "npm ci --prefix web" in python_job
    assert python_job.index("npm ci --prefix web") < python_job.index("Hermetic pytest tier")
    contract = (REPO / "tests/scvia/unit/test_builder.py").read_text()
    assert 'shutil.which("node")' in contract
    assert '"/usr/bin/node"' not in contract
    assert "tests/unit/python-release.test.ts" in contract


def test_recap_prompt_requires_evidence_for_a_mechanical_relationship(weekly_src):
    assert "say that they overlap by definition" not in weekly_src
    assert "clearances and contested possessions do" not in weekly_src
    assert "cite a source that defines the overlap" in weekly_src


@pytest.mark.parametrize("output_format", ["compact", "multiline", "malformed"])
def test_recap_retry_delivers_complete_failure_details_as_data(weekly_src, tmp_path, output_format):
    marker = tmp_path / "must-not-execute"
    details = [{
        "claim": "Daicos '23 [data] games' (rounds 1–25)",
        "reason": f'Use "22", not 23.\nKeep $HOME, $(touch {marker}) and `touch {marker}` literal.',
        "nested": {"rows": ["WF", "[data]"]},
    }]
    verdict = "CLI warning: retry context follows\n" + json.dumps(
        {"verdict": "FAIL", "failed_tags": details}, ensure_ascii=False,
        indent=2 if output_format == "multiline" else None,
    )
    if output_format == "malformed":
        verdict = "Malformed output with [data], 'quotes', \"double quotes\"\n" + details[0]["reason"]
    output = tmp_path / "sentinel.out"
    output.write_text(verdict)
    capture = tmp_path / "claude.json"
    claude = tmp_path / "claude-stub"
    claude.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "with open(os.environ['CLAUDE_CAPTURE'], 'w') as file:\n"
        "    json.dump({'argv': sys.argv[1:], 'stdin': sys.stdin.read()}, file)\n"
    )
    claude.chmod(0o755)
    start = weekly_src.index('    log "[3b/5] DataSentinel FAIL on pass 1')
    end = weekly_src.index('    log "[3b/5] Re-gating', start)
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", "log() { :; }\n" + weekly_src[start:end]],
        env={**os.environ, "ROUND": "25", "DS_OUT": str(output), "CLAUDE": str(claude),
             "PYTHON": sys.executable, "CLAUDE_CAPTURE": str(capture), "LOG_FILE": str(tmp_path / "log")},
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    received = json.loads(capture.read_text())
    assert received["argv"][-4:] == ["--agent", "FootyStrategy", "--permission-mode", "bypassPermissions"]
    if output_format != "malformed":
        assert json.loads(received["stdin"])["failed_tags"] == details
    else:
        assert received["stdin"] == verdict
    assert not marker.exists(), "verdict text was executed as shell code"


def test_agent_contracts_match_the_harness_weekly_approval_chain():
    qa = (REPO / ".claude/agents/QA.md").read_text()
    gaffer = (REPO / ".claude/agents/Gaffer.md").read_text()
    weekly = (REPO / ".claude/skills/weekly-cycle.md").read_text()
    skeptic = (REPO / ".claude/agents/Skeptic.md").read_text()
    for src in (qa, gaffer, weekly):
        assert "Phase 3d" in src
        assert "does not invoke the QA agent" in src
    assert "Pre-existing test failures also block ship" in qa
    assert "Pre-existing failures do not block ship" not in qa
    assert "/home/abhi/sourceCode/python" not in qa
    assert '"$PYTHON" -m pytest tests/ -q -m "not integration" -n 4' in qa
    assert '"$PYTHON" -m pytest tests/integration -q -m integration' in qa
    assert "weekly recap in `docs/afl-insights.md`" in skeptic
    assert "Do not invent recommendations or require lens sections" in skeptic


def test_legacy_ci_uses_the_locked_supported_environment():
    workflow = (REPO / ".github/workflows/scvia-ci.yml").read_text()
    legacy = workflow.split("  legacy:", 1)[1].split("  web:", 1)[0]
    assert "uv sync --locked --group dev --group legacy --extra ml" in legacy
    assert 'pytest tests/unit -m "not integration" -q -n 4' in legacy
    for obsolete in ("tests.yml", "pylint.yml", "python-package-conda.yml"):
        assert not (REPO / ".github/workflows" / obsolete).exists()
    assert "uv run --locked ruff check" in workflow
    assert "uv run --locked mypy" in workflow
