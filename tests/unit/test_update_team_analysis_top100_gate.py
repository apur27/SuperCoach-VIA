"""Unit tests: the CANONICAL top-100 profile regen + consistency gate must fail
CLOSED.

Twin of `tests/unit/test_refresh_readme_top100_gate.py`. That test pinned the
*mirror* (`refresh_readme._step_top100_markdown`, Step 2b). This one pins the
*canonical* definition the mirror was copied from — `update_team_analysis.main()`
step [13/14] — which carried the identical fail-open guard:

    if os.path.exists(TOP100_CSV) and os.path.exists(TOP100_SCORES_CSV):
        ... regenerate profiles ...
        ... run check_top100_consistency ...
    if new_hof != hof_text:
        write()                       # <-- writes even when the gate never ran

A missing gate input skipped the regen AND the gate but still wrote the doc, so
docs/hall-of-fame-top100.md — which CLAUDE.md exempts from inline `[data]` tags
*because* this gate exists — could ship unverified profile numbers with no error
on stdout or stderr.

The guard being duplicated in two files at once is itself the bug class: it was
copied from here into refresh_readme.py without being challenged. So the fix
single-sources the step into `update_team_analysis.update_top100_hof_doc()` and
both call sites delegate to it; `test_no_divergent_copy_of_the_gate_survives`
pins that there is no second copy to drift.
"""
import inspect
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import update_team_analysis as uta  # noqa: E402
import refresh_readme  # noqa: E402


STALE_DOC = """# Top 100

<!-- ALL-TIME-TOP100-START -->
old table
<!-- ALL-TIME-TOP100-END -->

## Player profiles - FootyStrategy tactical reads

### #1 Alpha Player — Club A
*200 games · 60 goals · 900 disposals · Score: 1.800*

Alpha prose stays intact.

### #2 Bravo Player — Club B
*100 games · 50 goals · 400 disposals · Score: 1.700*

Bravo prose stays intact.

## Related

- [back](x.md)
"""


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """Bio/scores CSVs + a stale HOF doc, with uta pointed at them."""
    bio = pd.DataFrame(
        [
            (1, "Alpha Player", "Club A",
             "Alpha Player played and 210 games. He recorded 999 total disposals and 65 goals."),
            (2, "Bravo Player", "Club B",
             "Bravo Player played and 105 games. He recorded 500 total disposals and 52 goals."),
        ],
        columns=["Serial Number", "Player Name", "Footy Teams", "Comment"],
    )
    scores = pd.DataFrame({"player": ["a", "b"], "all_time_score": [2.500, 2.400]})

    bio_p = tmp_path / "all_time_top_100.csv"
    sc_p = tmp_path / "scores.csv"
    hof_p = tmp_path / "hall-of-fame-top100.md"
    bio.to_csv(bio_p, index=False)
    scores.to_csv(sc_p, index=False)
    hof_p.write_text(STALE_DOC, encoding="utf-8")

    monkeypatch.setattr(uta, "TOP100_CSV", str(bio_p))
    monkeypatch.setattr(uta, "TOP100_SCORES_CSV", str(sc_p))
    monkeypatch.setattr(uta, "HALL_OF_FAME_PATH", str(hof_p))
    # Keep the table + chart halves off the real filesystem.
    monkeypatch.setattr(uta, "generate_top100_section", lambda: "new table")
    monkeypatch.setattr(uta, "generate_top100_chart", lambda: None)
    return str(hof_p)


def test_happy_path_regenerates_profiles_and_passes_gate(wired):
    written, errors = uta.update_top100_hof_doc()
    assert errors == [], f"canonical step should not error, got: {errors}"

    text = open(wired, encoding="utf-8").read()
    bio = pd.read_csv(uta.TOP100_CSV)
    scores = pd.read_csv(uta.TOP100_SCORES_CSV)
    hard, _warn = uta.check_top100_consistency(text, bio, scores)
    assert hard == [], f"doc written by the canonical step must pass the gate: {hard}"

    assert "210 games · 65 goals · 999 disposals · Score: 2.500" in text
    assert "*200 games · 60 goals · 900 disposals · Score: 1.800*" not in text
    assert "Alpha prose stays intact." in text
    assert wired in written


@pytest.mark.parametrize("missing_attr", ["TOP100_CSV", "TOP100_SCORES_CSV"])
def test_fails_closed_when_gate_inputs_missing(wired, monkeypatch, tmp_path, capsys,
                                               missing_attr):
    """Missing consistency inputs is an ERROR state, not a silent skip.

    This is the defect being closed: the old `if exists(a) and exists(b):` guard
    let the regen + gate be skipped while the doc was still written.
    """
    gone = str(tmp_path / "does_not_exist.csv")
    monkeypatch.setattr(uta, missing_attr, gone)
    before = open(wired, encoding="utf-8").read()

    written, errors = uta.update_top100_hof_doc()

    assert errors, "missing gate inputs must be reported as an error"
    assert any("missing" in e.lower() for e in errors), errors
    assert written == [], "nothing should be reported written when inputs are missing"
    assert open(wired, encoding="utf-8").read() == before, (
        "doc was written despite the consistency gate not running"
    )
    # Same error-reporting shape as the mirror: a [profile][FAIL] stderr line
    # that names the missing path.
    err = capsys.readouterr().err
    assert "[profile][FAIL]" in err, err
    assert gone in err, err


def test_fails_closed_when_hof_doc_missing(wired, monkeypatch, tmp_path):
    """The outer `and os.path.exists(HALL_OF_FAME_PATH)` guard was fail-open too.

    An absent destination doc skipped the entire step — table, profiles and gate —
    without an error, which is how a broken path would look identical to success.
    """
    monkeypatch.setattr(uta, "HALL_OF_FAME_PATH", str(tmp_path / "nope.md"))
    written, errors = uta.update_top100_hof_doc()
    assert errors, "a missing HOF doc must be reported, not silently skipped"
    assert written == []


def test_fails_closed_when_section_body_empty(wired, monkeypatch):
    """generate_top100_section() returns "" exactly when the ranking CSVs are
    absent. The canonical step's `if top100_body and ...` swallowed that."""
    monkeypatch.setattr(uta, "generate_top100_section", lambda: "")
    before = open(wired, encoding="utf-8").read()

    written, errors = uta.update_top100_hof_doc()

    assert errors, "an empty top-100 section body must be reported as an error"
    assert written == []
    assert open(wired, encoding="utf-8").read() == before


def test_reports_gate_failure_and_does_not_write(wired, monkeypatch, capsys):
    """A hard gate mismatch blocks the write instead of shipping bad numbers."""
    monkeypatch.setattr(uta, "regenerate_top100_profiles", lambda text, b, s: (text, []))

    written, errors = uta.update_top100_hof_doc()

    assert errors, "a hard gate mismatch must be reported as an error"
    assert any("consistency" in e.lower() or "mismatch" in e.lower() for e in errors), errors
    assert written == []
    text = open(wired, encoding="utf-8").read()
    assert "*200 games · 60 goals · 900 disposals · Score: 1.800*" in text


def test_no_divergent_copy_of_the_gate_survives():
    """Structural: both call sites must route through the one implementation.

    Deliberately a source-level assertion. The incident was not that either copy
    was wrong in isolation — it was that a fail-open guard existed in two places
    and only one got fixed. Pinning "there is exactly one copy" is the invariant
    that prevents the next divergence; pinning behaviour alone would not.
    """
    main_src = inspect.getsource(uta.main)
    mirror_src = inspect.getsource(refresh_readme._step_top100_markdown)

    for name, src in (("update_team_analysis.main", main_src),
                      ("refresh_readme._step_top100_markdown", mirror_src)):
        assert "update_top100_hof_doc(" in src, (
            f"{name} must delegate to update_top100_hof_doc()"
        )
        assert "check_top100_consistency" not in src, (
            f"{name} carries its own copy of the consistency gate — "
            "that duplication is the bug class"
        )
        assert "regenerate_top100_profiles" not in src, (
            f"{name} carries its own copy of the profile regeneration"
        )


def test_main_hard_aborts_when_the_step_errors(monkeypatch):
    """The only permitted divergence between the two envelopes.

    main() is a script entrypoint, so an error is a non-zero exit; refresh_readme
    is a multi-step orchestrator, so it collects the error and keeps going. Both
    refuse to write — the fail-closed semantics are identical, only the reporting
    differs.
    """
    src = inspect.getsource(uta.main)
    step = src.split("[13/14]", 1)[1]
    assert "SystemExit" in step, (
        "main() must still hard-abort when the top-100 gate step reports an error"
    )
