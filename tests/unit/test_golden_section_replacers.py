"""Golden-file regression tier for the marker-driven section replacers.

`update_team_analysis.py` publishes docs/afl-*-2026.md by replacing the content
between HTML-comment markers. Every one of these helpers has the same failure
mode: if the marker pair (or the fallback anchor) is not found, the helper
returns the input text verbatim. The callers in `refresh_readme.py` and
`main()` then evaluate `if new_text != text:` and simply do not write — with no
error, no non-zero exit, and a green run log. The published section silently
freezes.

That is not hypothetical. It has shipped twice: once as a frozen profile
section, once as a pattern that matched nothing because the file used a
different dash encoding than the pattern expected.

Two things are pinned here:

  1. A golden snapshot of the composed output — all six replacers applied to
     one fixture doc. A replacer that stops substituting shows up as an
     explicit diff instead of a stale published file.
  2. A parametrised ZERO-MATCH case per replacer. Matching nothing must be
     loud (SystemExit or ValueError). "Returned the input unchanged" is a FAIL.

Hermetic: the only input is a committed fixture, and no test writes outside
tmp_path.

Re-record with `UPDATE_GOLDEN=1 pytest tests/unit/test_golden_section_replacers.py`
and review `git diff tests/fixtures/golden/` before committing.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import update_team_analysis as uta  # noqa: E402

FIXTURES = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "fixtures", "golden", "sections")
)
SECTIONS_INPUT = os.path.join(FIXTURES, "sections_input.md")

YEAR = 2026
EM = "—"
EN = "–"


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _assert_golden(name, actual):
    path = os.path.join(FIXTURES, name)
    if os.environ.get("UPDATE_GOLDEN"):
        with open(path, "w", encoding="utf-8") as f:
            f.write(actual)
        pytest.skip(f"re-recorded golden {name} — review the diff before committing")
    assert os.path.exists(path), (
        f"golden {name} missing; re-record with UPDATE_GOLDEN=1 and review the diff"
    )
    expected = _read(path)
    if actual != expected:
        import difflib
        diff = "\n".join(
            difflib.unified_diff(
                expected.splitlines(), actual.splitlines(),
                fromfile=f"golden/{name}", tofile="generated", lineterm="",
            )
        )
        pytest.fail(
            f"section replacer output drifted from golden {name}. If intended, "
            f"re-record with UPDATE_GOLDEN=1 and review.\n{diff}"
        )


# The six marker-driven replacers, with the call shape each one takes.
REPLACERS = [
    ("replace_section", lambda t, b: uta.replace_section(t, YEAR, b)),
    ("replace_finals_pathway_section",
     lambda t, b: uta.replace_finals_pathway_section(t, YEAR, b)),
    ("replace_brownlow_predictor_section",
     lambda t, b: uta.replace_brownlow_predictor_section(t, YEAR, b)),
    ("replace_stat_leaders_section",
     lambda t, b: uta.replace_stat_leaders_section(t, YEAR, b)),
    ("replace_predictions_section",
     lambda t, b: uta.replace_predictions_section(t, YEAR, b)),
    ("replace_backtest_section",
     lambda t, b: uta.replace_backtest_section(t, YEAR, b)),
    ("replace_5year_section",
     lambda t, b: uta.replace_5year_section(t, YEAR, [2021, 2022, 2023, 2024, 2025], b)),
]


def test_all_replacers_composed_match_golden():
    """Apply every replacer in pipeline order to one doc and snapshot it."""
    text = _read(SECTIONS_INPUT)
    for name, fn in REPLACERS:
        before = text
        text = fn(text, f"FRESH {name} body.")
        assert text != before, f"{name} left the document unchanged"
    _assert_golden("expected_sections_doc.md", text)


@pytest.mark.parametrize("name,fn", REPLACERS, ids=[r[0] for r in REPLACERS])
def test_replacer_is_loud_when_it_matches_nothing(name, fn):
    """No markers, no fallback anchor: the helper must raise, not no-op.

    Three of these (predictions, backtest, stat-leaders) used to print a line
    to stderr and return the text verbatim. stderr in a 2h harness run is not
    a gate — `refresh_readme._step_predictions_and_backtest` recorded no error
    and the cycle reported success with a frozen section.
    """
    doc = "# A doc with none of the markers or anchors this replacer needs\n\nbody\n"
    with pytest.raises((SystemExit, ValueError)):
        fn(doc, "NEW BODY")


def test_replace_section_is_loud_when_the_header_anchor_dash_encoding_differs():
    """The exact incident shape, on the one replacer with a prose anchor.

    `replace_section`'s fallback anchor is the literal string
    `## 2026 season — live team analysis`. Re-encode that em dash as an en dash
    and the anchor matches nothing. The doc looks correct to a human reader and
    the substitution finds no home.
    """
    header_em = f"## {YEAR} season {EM} live team analysis"
    doc_en = f"# Doc\n\n{header_em.replace(EM, EN)}\n\nold body\n"
    with pytest.raises(SystemExit):
        uta.replace_section(doc_en, YEAR, "NEW BODY")

    # Control: the em-dash spelling still resolves, so the guard is specific.
    ok = uta.replace_section(f"# Doc\n\n{header_em}\n\nold body\n", YEAR, "NEW BODY")
    assert "NEW BODY" in ok


@pytest.mark.parametrize("name,fn", REPLACERS, ids=[r[0] for r in REPLACERS])
def test_replacer_is_idempotent(name, fn):
    """Running the weekly generator twice must not duplicate a section."""
    text = _read(SECTIONS_INPUT)
    once = fn(text, f"FRESH {name} body.")
    twice = fn(once, f"FRESH {name} body.")
    assert twice == once, f"{name} is not idempotent — repeated runs drift"


def test_replacers_do_not_touch_content_outside_their_markers():
    """Prose above and below every marker pair survives byte-identical."""
    text = _read(SECTIONS_INPUT)
    for name, fn in REPLACERS:
        text = fn(text, f"FRESH {name} body.")
    assert text.startswith("# AFL insights (golden fixture)")
    assert "Text after every marker pair must survive untouched." in text
    assert "STALE" not in text, "a stale body survived a replacement"
