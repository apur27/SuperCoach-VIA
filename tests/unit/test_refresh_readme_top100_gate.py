"""Unit tests: the top-100 profile regen + consistency gate must actually RUN
in production.

`update_team_analysis.main()` step [13/14] regenerates the HOF profile prose and
hard-fails on a mismatch, but no harness script invokes `main()`. What the
harness DOES run is `refresh_readme.py` (refresh_and_rank.sh), whose Step 2b
(`_step_top100_markdown`) rewrote only the ALL-TIME-TOP100 *table* and skipped
the profile pass entirely. Result: the table advanced every week while the
profile stat-lines stayed frozen, and the gate that CLAUDE.md leans on to
justify the doc's inline-`[data]`-tag exemption never fired.

These tests pin the contract that Step 2b regenerates profiles AND gates.
"""
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


def _fixtures(tmp_path):
    """Write bio/scores CSVs + a stale HOF doc, and point uta at them."""
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
    return str(bio_p), str(sc_p), str(hof_p)


@pytest.fixture
def wired(tmp_path, monkeypatch):
    bio_p, sc_p, hof_p = _fixtures(tmp_path)
    monkeypatch.setattr(uta, "TOP100_CSV", bio_p)
    monkeypatch.setattr(uta, "TOP100_SCORES_CSV", sc_p)
    monkeypatch.setattr(uta, "HALL_OF_FAME_PATH", hof_p)
    # Keep the step off the network/filesystem for the table + chart halves.
    monkeypatch.setattr(uta, "generate_top100_section", lambda: "new table")
    monkeypatch.setattr(uta, "generate_top100_chart", lambda: None)
    return hof_p


def test_step_2b_regenerates_profiles_and_passes_gate(wired):
    """Step 2b must leave the doc consistent with the ranking CSV."""
    written, errors = refresh_readme._step_top100_markdown()
    assert errors == [], f"step should not error, got: {errors}"

    text = open(wired, encoding="utf-8").read()
    bio = pd.read_csv(uta.TOP100_CSV)
    scores = pd.read_csv(uta.TOP100_SCORES_CSV)

    hard, _warn = uta.check_top100_consistency(text, bio, scores)
    assert hard == [], f"doc written by Step 2b must pass the gate, got: {hard}"

    # The stale stat-line is gone, the current one is present.
    assert "210 games · 65 goals · 999 disposals · Score: 2.500" in text
    assert "*200 games · 60 goals · 900 disposals · Score: 1.800*" not in text
    # Narrative prose is preserved verbatim by the regeneration.
    assert "Alpha prose stays intact." in text
    assert "Bravo prose stays intact." in text
    assert written, "step should report the HOF doc as written"


@pytest.mark.parametrize("missing_attr", ["TOP100_CSV", "TOP100_SCORES_CSV"])
def test_step_2b_fails_closed_when_gate_inputs_missing(wired, monkeypatch, tmp_path,
                                                       missing_attr):
    """Missing consistency inputs is an ERROR state, not a silent skip.

    The original guard was `if os.path.exists(bio) and os.path.exists(scores):`
    — so an absent CSV skipped the regen + gate but still WROTE the doc, shipping
    unregenerated profiles with no error. That fail-open is the exact failure mode
    that hid this incident for seven weekly cycles: a gate that quietly does not
    run. The step must refuse to write instead.
    """
    monkeypatch.setattr(uta, missing_attr, str(tmp_path / "does_not_exist.csv"))
    before = open(wired, encoding="utf-8").read()

    written, errors = refresh_readme._step_top100_markdown()

    assert errors, "missing gate inputs must be reported as an error"
    assert any("missing" in e.lower() for e in errors), errors
    assert written == [], "nothing should be reported written when inputs are missing"
    # The doc must be untouched — stale profiles must not ship unverified.
    assert open(wired, encoding="utf-8").read() == before, (
        "doc was written despite the consistency gate not running"
    )


def test_step_2b_reports_gate_failure_and_does_not_write(wired, monkeypatch):
    """If the gate hard-fails, Step 2b surfaces an error and leaves the doc alone.

    Simulated by neutering the regeneration pass so the stale profiles survive
    into the gate — the gate must then block the write rather than shipping
    numbers that disagree with the ranking.
    """
    monkeypatch.setattr(uta, "regenerate_top100_profiles", lambda text, b, s: (text, []))

    written, errors = refresh_readme._step_top100_markdown()

    assert errors, "a hard gate mismatch must be reported as an error"
    assert any("consistency" in e.lower() or "mismatch" in e.lower() for e in errors), errors
    assert written == [], "nothing should be reported written when the gate fails"
    # Doc must still carry the original stale profile stat-line (no partial write).
    text = open(wired, encoding="utf-8").read()
    assert "*200 games · 60 goals · 900 disposals · Score: 1.800*" in text
