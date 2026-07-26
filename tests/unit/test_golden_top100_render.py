"""Golden-file regression tier for the top-100 doc generators.

Why this file exists
--------------------
`update_team_analysis.py` and `refresh_readme.py` rewrite published docs. Twice
we shipped a generator defect that no test caught:

  1. The profile prose froze while the table refreshed — the regeneration pass
     existed but the production entry point never called it, so the doc looked
     freshly generated and was half stale.
  2. A pattern silently matched NOTHING because the file used a different dash
     encoding than the pattern expected. The generator reported success and
     changed nothing.

Both are invisible to assertion-style tests that only spot-check a substring:
if the generator emits nothing, a substring assertion on the *input* still
passes. A golden file catches it, because "changed nothing" is a diff.

These tests are HERMETIC — every input is a committed fixture under
`tests/fixtures/golden/top100/` and every module path constant is
monkeypatched, so nothing here reads or writes the real `data/` tree or the
real `docs/`. They belong in the fast tier, not the integration tier.

Updating a golden
-----------------
When a generator change is intended, re-record and review the diff:

    UPDATE_GOLDEN=1 pytest tests/unit/test_golden_top100_render.py

then `git diff tests/fixtures/golden/` and confirm every changed line is a
change you meant to publish. Never re-record to make a red test green without
reading the diff — that is precisely the review step this tier exists to force.
"""
import os
import re
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import update_team_analysis as uta  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures", "golden", "top100")
BIO_CSV = os.path.abspath(os.path.join(FIXTURES, "bio.csv"))
SCORES_CSV = os.path.abspath(os.path.join(FIXTURES, "scores.csv"))
HOF_INPUT = os.path.abspath(os.path.join(FIXTURES, "hof_input.md"))

EM = "—"  # em dash — the dash the profile/heading patterns expect
EN = "–"  # en dash — the encoding variant that silently broke a pattern


# --------------------------------------------------------------------- helpers


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _normalise_date(text):
    """`generate_top100_section` stamps today's date. Freeze it in the golden."""
    return re.sub(r"\*Last updated: \d{4}-\d{2}-\d{2} ", "*Last updated: <DATE> ", text)


def _assert_golden(name, actual):
    """Compare against the committed snapshot, or re-record under UPDATE_GOLDEN=1."""
    path = os.path.abspath(os.path.join(FIXTURES, name))
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
            f"generator output drifted from golden {name}. If the change is "
            f"intended, re-record with UPDATE_GOLDEN=1 and review.\n{diff}"
        )


@pytest.fixture
def ranking():
    return pd.read_csv(BIO_CSV), pd.read_csv(SCORES_CSV)


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Point every module path constant at fixtures / tmp_path.

    Nothing under this fixture may touch the real repo artifacts — an earlier
    test in this repo rewrote real charts and docs because it patched only some
    of the constants.
    """
    monkeypatch.setattr(uta, "TOP100_CSV", BIO_CSV)
    monkeypatch.setattr(uta, "TOP100_SCORES_CSV", SCORES_CSV)
    hof = tmp_path / "hall-of-fame-top100.md"
    hof.write_text(_read(HOF_INPUT), encoding="utf-8")
    monkeypatch.setattr(uta, "HALL_OF_FAME_PATH", str(hof))
    # The chart is matplotlib I/O against a real assets/ path — out of scope for
    # a text-rendering golden, and a real-artifact write hazard.
    monkeypatch.setattr(uta, "generate_top100_chart", lambda: "")
    return hof


def _render_doc(ranking):
    """The production render order: table block first, then profile resync."""
    bio_df, scores_df = ranking
    original = _read(HOF_INPUT)
    text = uta.replace_top100_section(original, uta.generate_top100_section())
    text, warnings = uta.regenerate_top100_profiles(text, bio_df, scores_df)
    assert text != original, "generator produced no change at all"
    return text, warnings


# ------------------------------------------------------------ table rendering


def test_top100_table_section_matches_golden(wired, ranking):
    """The markdown table built from the ranking CSVs, byte for byte."""
    body = uta.generate_top100_section()
    assert body, "generate_top100_section returned empty for a valid ranking"
    _assert_golden("expected_top100_section.md", _normalise_date(body))


def test_table_renders_thousands_separators_like_the_stat_line(wired, ranking):
    """The same parsed value must not render two ways in the same document.

    The table built goals with `str(v)` while the profile stat-line built it
    with `f"{v:,}"`, so a 1,360-goal career appeared as `1360` in the table and
    `1,360` three screens below it — from one parse of one Comment field. A
    reader reconciling the two has no way to tell that apart from two different
    numbers. Assert the agreement generically rather than pinning a literal, so
    a future column added to one renderer and not the other is caught too.
    """
    bio_df, scores_df = ranking
    body = uta.generate_top100_section()
    score_map = uta._top100_score_map(scores_df)
    checked = 0
    for _, r in bio_df.iterrows():
        rank = int(r["Serial Number"])
        stats = uta._parse_top100_comment(str(r["Comment"]))
        row = [ln for ln in body.splitlines() if ln.startswith(f"| {rank} |")][0]
        cells = [c.strip() for c in row.strip("|").split("|")]
        stat_line = uta._format_top100_stat_line(stats, score_map.get(rank, 0.0))
        for key in ("games", "goals", "disposals", "brownlow"):
            if stats.get(key, 0) < 1000:
                continue  # `:,` is a no-op below 1000 — nothing to disagree about
            grouped = f"{stats[key]:,}"
            assert grouped in stat_line, (
                f"fixture drift: stat-line for #{rank} lost the {key} value"
            )
            assert grouped in cells, (
                f"{key} for #{rank} renders as {stats[key]!r} in the table but "
                f"{grouped!r} in the stat-line ({stat_line}) — same parsed value, "
                f"two renderings. Table row: {row}"
            )
            assert str(stats[key]) not in cells, (
                f"table row for #{rank} still carries the unseparated {key} "
                f"form {str(stats[key])!r}: {row}"
            )
            checked += 1
    assert checked, "no fixture value >= 1000 — this test verified nothing"


def test_table_omits_stats_the_era_did_not_record(wired, ranking):
    """Bravo has no disposals/Brownlow in his Comment — the row must show em
    dashes, not zeros. A zero here would be a fabricated [data] number."""
    body = uta.generate_top100_section()
    row = [ln for ln in body.splitlines() if ln.startswith("| 2 |")][0]
    assert row.count("|" + " " + "—" + " |") >= 1 or "—" in row, row
    assert " 0 " not in row, f"absent stat rendered as zero: {row}"


# ---------------------------------------------------------- profile rendering


def test_rendered_hof_doc_matches_golden(wired, ranking):
    """Full doc: table replaced, profiles reordered/restat-lined, prose intact.

    This is the snapshot that would have caught the frozen-profile defect: a
    generator that rewrites the table and skips the profiles produces a
    different file, and the diff names every profile it failed to touch.
    """
    text, _warnings = _render_doc(ranking)
    _assert_golden("expected_hof_doc.md", _normalise_date(text))


def test_render_is_idempotent(wired, ranking):
    """Re-running the generator on its own output must be a no-op.

    Non-idempotence is how a weekly generator accumulates duplicated sections
    in a published doc over successive cycles.
    """
    bio_df, scores_df = ranking
    once, _ = _render_doc(ranking)
    twice, _ = uta.regenerate_top100_profiles(once, bio_df, scores_df)
    twice = uta.replace_top100_section(twice, uta.generate_top100_section())
    assert _normalise_date(twice) == _normalise_date(once)


def test_regenerated_doc_passes_the_consistency_gate(wired, ranking):
    bio_df, scores_df = ranking
    text, _ = _render_doc(ranking)
    hard, _warn = uta.check_top100_consistency(text, bio_df, scores_df)
    assert hard == [], f"freshly generated doc fails its own gate: {hard}"


def test_membership_swap_is_reported_not_silent(wired, ranking):
    """A dropped player and a new entrant must both surface as warnings."""
    _text, warnings = _render_doc(ranking)
    assert any("Zulu" in w for w in warnings), warnings
    assert any("Charlie" in w and "new entrant" in w for w in warnings), warnings


def test_update_top100_hof_doc_writes_only_the_patched_path(wired, ranking, monkeypatch):
    """End-to-end through the shared entry point, writing to tmp_path only."""
    written, errors = uta.update_top100_hof_doc()
    assert errors == [], errors
    assert written == [str(wired)], written
    _assert_golden("expected_hof_doc.md", _normalise_date(wired.read_text(encoding="utf-8")))


# ------------------------------------------------- ZERO-MATCH regression cases
#
# The defect class: a pattern/anchor matches nothing, the generator returns the
# text unchanged, and the caller — which only checks `new_text != text` before
# writing — reports success. Every case below must be LOUD.


def test_replace_top100_section_raises_when_nothing_matches():
    """No markers and no heading anchor: must not return the text unchanged."""
    doc = "# A doc with neither the markers nor the heading\n\nbody\n"
    with pytest.raises(ValueError) as exc:
        uta.replace_top100_section(doc, "NEW BODY")
    assert "ALL-TIME-TOP100" in str(exc.value)


def test_replace_top100_section_raises_on_heading_dash_variant():
    """The exact incident: markers absent, heading present but en-dashed.

    The fallback anchor is matched as a literal string containing an em dash.
    An en dash (or an HTML entity) means the anchor matches nothing. Before the
    guard this returned the input verbatim and the pipeline logged success.
    """
    heading_em = "### Top 100 AFL players of all time " + EM + " ranked by the data"
    heading_en = heading_em.replace(EM, EN)
    doc = f"# Doc\n\n{heading_en}\n\nold body\n"
    with pytest.raises(ValueError) as exc:
        uta.replace_top100_section(doc, "NEW BODY")
    assert "ALL-TIME-TOP100" in str(exc.value)

    # Control: with the em dash the fallback DOES insert, so the guard is not
    # simply refusing everything.
    ok = uta.replace_top100_section(f"# Doc\n\n{heading_em}\n\nold body\n", "NEW BODY")
    assert "NEW BODY" in ok


def test_profile_regen_raises_when_a_heading_uses_a_different_dash(wired, ranking):
    """A profile heading the block regex cannot parse must fail, not vanish.

    `_parse_profile_blocks` requires ` <em dash> ` between name and club. With an
    en dash the heading is not a block boundary, so the whole profile — heading,
    stat-line and prose — is silently swallowed into the previous block's prose
    and then re-emitted as a placeholder, destroying authored narrative.
    """
    bio_df, scores_df = ranking
    doc = _read(HOF_INPUT).replace(
        "### #3 Alpha Player " + EM + " Club A",
        "### #3 Alpha Player " + EN + " Club A",
    )
    with pytest.raises(ValueError) as exc:
        uta.regenerate_top100_profiles(doc, bio_df, scores_df)
    assert "Alpha Player" in str(exc.value)


def test_consistency_gate_fails_when_it_matches_zero_profiles(ranking):
    """A gate that checked nothing must not report PASS.

    If every heading carries an encoding variant the gate's regex does not
    match, `check_top100_consistency` returns ([], []) — indistinguishable from
    a clean doc. That is how an unverified doc ships under the CLAUDE.md
    inline-tag exemption, which is granted *because* this gate runs.
    """
    bio_df, scores_df = ranking
    doc = _read(HOF_INPUT).replace(EM, EN)
    hard, _warn = uta.check_top100_consistency(doc, bio_df, scores_df)
    assert hard, "gate matched zero profiles in a doc with a profiles section and passed"
    assert any("0 " in h or "zero" in h.lower() for h in hard), hard


def test_consistency_gate_still_passes_on_a_doc_with_no_profiles_section(ranking):
    """The zero-match guard must be scoped: a doc that legitimately has no
    profiles section (e.g. the table-only render path) is not a gate failure."""
    bio_df, scores_df = ranking
    hard, _warn = uta.check_top100_consistency("# Just a table\n\n| a |\n", bio_df, scores_df)
    assert hard == []
