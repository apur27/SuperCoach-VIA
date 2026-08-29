"""Finals-mode contracts.

Once the home-and-away season ends there is no "next round" to predict. The
predictor cannot express a finals round at all — `extract_round_number` returns
NaN for every finals label, `round_number` is a model feature, and NaN-round rows
are dropped before the forward CSV is written — so `get_next_round()` returns
max(integer round) + 1, i.e. a round that will never be played.

Shipping that phantom is not a cosmetic problem. `generate_weekly_cheat_sheet.py`
selects its input by `(round_number, timestamp)` parsed from the FILENAME, and
prediction filenames carry no year, so a `next_round_26` file outranks every
round 1-25 of the following season indefinitely. One accidental full-harness run
during finals poisons cheat-sheet selection for a year.

FINALS_MODE=1 is the clean skip: no prediction is generated, and the round label
and backtest upper bound are derived from the MATCH DATA instead of from the
prediction artifact. That derivation is the load-bearing half — the harness
previously read the round number out of the prediction filename, so merely
skipping the prediction step made `seq START END` go empty and the backtest was
skipped SILENTLY, with no error.

Shell contracts here are reachability checks in the style of test_harness_wiring:
a correct branch that the production path never reaches is worthless. Real
end-to-end behaviour is covered by the CLAUDE.md 6.2 smoke run.
"""
import re
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
REFRESH = REPO / "refresh_and_rank.sh"
WEEKLY = REPO / "scripts" / "weekly_refresh.sh"


def _uncommented(src: str) -> str:
    """Drop comment-only lines so a mention in prose cannot satisfy a contract."""
    return "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))


@pytest.fixture(scope="module")
def refresh_src():
    return _uncommented(REFRESH.read_text())


@pytest.fixture(scope="module")
def weekly_src():
    return _uncommented(WEEKLY.read_text())


# --------------------------------------------------------------- refresh_and_rank

def test_prediction_step_is_gated_behind_finals_mode(refresh_src):
    """`supercoach.prediction` must not run unconditionally."""
    assert "supercoach.prediction" in refresh_src, "prediction step vanished entirely"
    # The invocation has to sit inside a FINALS_MODE conditional, not at top level.
    m = re.search(r"if\s+\[\s+\"\$FINALS_MODE\"\s*(?:!=|=)\s*\"1\"\s+\].*?supercoach\.prediction",
                  refresh_src, re.DOTALL)
    assert m, "supercoach.prediction is not guarded by a FINALS_MODE conditional"


def test_finals_mode_derives_backtest_bound_from_match_data(refresh_src):
    """The backtest upper bound must come from the data, not the prediction file.

    This is the regression that matters: with the bound derived from
    UPCOMING_ROUND-1, skipping the prediction silently emptied the scoring loop.
    """
    assert "--print-last-ha-round" in refresh_src, (
        "finals mode must derive the last settled H&A round from matches data "
        "via check_round_settled.py --print-last-ha-round"
    )
    # And it must actually feed END_SCORE_ROUND.
    assert re.search(r"END_SCORE_ROUND=.*--print-last-ha-round", refresh_src, re.DOTALL), \
        "--print-last-ha-round is present but does not feed END_SCORE_ROUND"


def test_default_path_still_uses_upcoming_round_minus_one(refresh_src):
    """Flag-off behaviour must be unchanged."""
    assert "UPCOMING_ROUND - 1" in refresh_src or "UPCOMING_ROUND-1" in refresh_src, \
        "the normal-season backtest bound was removed"


def test_finals_mode_exported_for_child_scripts(refresh_src):
    """refresh_readme.py reads FINALS_MODE from the environment."""
    assert re.search(r"export\s+FINALS_MODE", refresh_src), \
        "FINALS_MODE must be exported so child python steps observe it"


# ----------------------------------------------------------------- weekly_refresh

def test_weekly_derives_round_from_data_in_finals_mode(weekly_src):
    assert "--print-last-ha-round" in weekly_src, (
        "weekly_refresh must derive ROUND from matches data in finals mode, "
        "not from the newest prediction CSV"
    )


def test_weekly_skips_prediction_mtime_assert_in_finals_mode(weekly_src):
    """The freshness assert aborts the cycle when no new prediction was written.

    It must not fire in finals mode, where writing one is the whole point of not
    doing.
    """
    m = re.search(r"predates this run", weekly_src)
    assert m, "the prediction-freshness assert disappeared"
    guard = weekly_src[:m.start()]
    assert "FINALS_MODE" in guard, \
        "the prediction-freshness assert is not guarded by FINALS_MODE"


def test_weekly_skips_cheat_sheet_in_finals_mode(weekly_src):
    m = re.search(r"generate_weekly_cheat_sheet\.py", weekly_src)
    assert m, "cheat sheet step vanished"
    guard = weekly_src[:m.start()]
    assert "FINALS_MODE" in guard, \
        "the cheat sheet step is not guarded by FINALS_MODE (no forward CSV exists)"


def test_weekly_recap_prompt_states_round_is_completed_in_finals_mode(weekly_src):
    """ROUND flips meaning in finals mode: upcoming -> completed.

    The FootyStrategy prompt must say so, or the recap repeats the BL-19 mislabel
    in the other direction.
    """
    assert re.search(r"FINALS_MODE", weekly_src), "FINALS_MODE absent from weekly_refresh"
    assert re.search(r"complet(ed|e)", weekly_src, re.IGNORECASE), \
        "no wording distinguishes the completed round from the upcoming one"


# ------------------------------------------------------------------ refresh_readme

def _fake_uta(calls, tmp_path):
    """Stub update_team_analysis that records which sections were generated."""
    mod = types.ModuleType("update_team_analysis")
    pred_path = tmp_path / "afl-predictions.md"
    back_path = tmp_path / "afl-backtest.md"
    pred_path.write_text("PRED\n")
    back_path.write_text("BACK\n")

    def _pred(year):
        calls.append("predictions")
        return "pred-body"

    def _back(year):
        calls.append("backtest")
        return "back-body"

    mod.generate_predictions_section = _pred
    mod.generate_backtest_section = _back
    mod.PREDICTIONS_PATH = str(pred_path)
    mod.BACKTEST_PATH = str(back_path)
    mod.replace_predictions_section = lambda text, year, body: text + body
    mod.replace_backtest_section = lambda text, year, body: text + body
    mod.load_all_player_games = lambda: []
    mod.detect_current_year = lambda games: 2026
    return mod


@pytest.fixture
def refresh_readme_mod():
    sys.path.insert(0, str(REPO))
    import refresh_readme
    return refresh_readme


def test_finals_mode_skips_predictions_section_keeps_backtest(
    monkeypatch, tmp_path, refresh_readme_mod
):
    """A played round must not be re-stamped 'generated today'.

    generate_predictions_section always sets gen_date = now(), so regenerating it
    during finals republishes round-25 projections as if freshly made for an
    upcoming round. The backtest section must still run — it is this cycle's
    actual deliverable.
    """
    calls = []
    monkeypatch.setitem(sys.modules, "update_team_analysis", _fake_uta(calls, tmp_path))
    monkeypatch.setenv("FINALS_MODE", "1")

    written, errors = refresh_readme_mod._step_predictions_and_backtest()

    assert errors == [], errors
    assert "backtest" in calls, "backtest section must still regenerate in finals mode"
    assert "predictions" not in calls, (
        "predictions section must NOT regenerate in finals mode — it would stamp "
        "today's date on a round that has already been played"
    )


def test_default_mode_still_regenerates_both_sections(
    monkeypatch, tmp_path, refresh_readme_mod
):
    """Flag-off behaviour unchanged."""
    calls = []
    monkeypatch.setitem(sys.modules, "update_team_analysis", _fake_uta(calls, tmp_path))
    monkeypatch.delenv("FINALS_MODE", raising=False)

    written, errors = refresh_readme_mod._step_predictions_and_backtest()

    assert errors == [], errors
    assert "predictions" in calls and "backtest" in calls, \
        "normal-season path must regenerate both sections"
