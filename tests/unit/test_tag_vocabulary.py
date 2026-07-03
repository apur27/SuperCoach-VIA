"""Tests for scripts/tag_vocabulary.py — the single source of truth for the
repo's [data] / [historical record] tag vocabulary.

Written test-first (TDD). Covers the F2 regression: spec-form `**[data: ...]**`
tags must be counted (the broken Skeptic matcher returned zero for these).
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import tag_vocabulary as tv  # noqa: E402


# --- reference matcher (the validated one from inject_trust_badge.py) ---------
_REF_STAMP_RE = re.compile(r"^.*<!-- council-pipeline:.*$", re.MULTILINE)
_REF_DATA_RE = re.compile(r"\*\*\[data(?:\]|\s*:)|(?<!\*)\[data\]:")


def _reference_data_count(text: str) -> int:
    return len(_REF_DATA_RE.findall(_REF_STAMP_RE.sub("", text)))


# --- fixtures ----------------------------------------------------------------

SPEC_ONLY = """\
Dusty averaged 27.4 disposals **[data: martin_dustin_..._performance.csv ; disposals ; season-mean]**.
He kicked 0.8 goals a game **[data: martin_dustin_..._performance.csv ; goals ; season-mean]**.
"""

ALL_THREE_DATA = """\
Career games: 302 **[data]**.
Season average 27.4 **[data: file.csv ; disposals ; mean]**.
[data]: forgotten-heroes derivation lives here.
"""

ALL_THREE_HR = """\
Best-on-ground count **[historical record]**.
Kicked the sealer **[historical record: 1978 Grand Final ; result]**.
[historical record]: pre-1965 tally, unverified in data.
"""

META_AND_STAMP = """\
This doc has verified [data] tags and every [data] figure is checked.
A real one here **[data]**.
<!-- council-pipeline: gated ; every [data] figure ; **[data]** token in stamp -->
Another real one **[data: f.csv ; goals ; sum]**.
"""

MIXED = """\
Intro prose mentioning [data] tags in passing.
Line with a bare tag **[data]**.
Line with a spec tag **[data: f.csv ; disposals ; mean]**.
Line with colon form [data]: something.
A historical bare **[historical record]**.
A historical spec **[historical record: gf ; result]**.
Historical colon [historical record]: old.
<!-- council-pipeline: stamp with **[data]** and **[historical record]** noise -->
"""

EMPTY = ""
NO_TAGS = "Just some prose. Nothing tagged here. [data] mentioned meta-style.\n"


# --- tests -------------------------------------------------------------------

def test_all_three_data_forms_counted():
    assert tv.count_tags(ALL_THREE_DATA, kinds=("data",)) == 3


def test_spec_form_only_not_zero_F2_regression():
    # The broken matcher returned 0 here. Must be 2.
    assert tv.count_tags(SPEC_ONLY, kinds=("data",)) == 2


def test_all_three_historical_record_forms_counted():
    assert tv.count_tags(ALL_THREE_HR, kinds=("historical record",)) == 3


def test_meta_references_and_stamp_excluded():
    # Two genuine tags; the "verified [data] tags"/"every [data] figure" meta
    # refs and the two stamp-line tokens must NOT count.
    assert tv.count_tags(META_AND_STAMP, kinds=("data",)) == 2


def test_count_matches_reference_regex_on_mixed():
    assert tv.count_tags(MIXED, kinds=("data",)) == _reference_data_count(MIXED)


def test_default_count_is_data_only():
    # MIXED has 3 data + 3 hr genuine tags. Default count_tags = data only.
    assert tv.count_tags(MIXED) == 3


def test_extract_returns_line_numbers_and_kinds_in_order():
    tags = tv.extract_tags(MIXED)  # default: both kinds
    # document order, genuine tags only (stamp + meta excluded)
    assert [(t["line"], t["tag"]) for t in tags] == [
        (2, "data"),
        (3, "data"),
        (4, "data"),
        (5, "historical record"),
        (6, "historical record"),
        (7, "historical record"),
    ]
    # source line is stripped
    assert tags[0]["text"] == "Line with a bare tag **[data]**."


def test_extract_respects_kinds_filter():
    tags = tv.extract_tags(MIXED, kinds=("data",))
    assert all(t["tag"] == "data" for t in tags)
    assert len(tags) == 3


def test_extract_line_number_alignment_after_stamp_strip():
    # Stamp line is blanked but newline preserved -> line numbers stay aligned.
    tags = tv.extract_tags(META_AND_STAMP, kinds=("data",))
    assert [t["line"] for t in tags] == [2, 4]


def test_empty_and_no_tags():
    assert tv.count_tags(EMPTY) == 0
    assert tv.extract_tags(EMPTY) == []
    assert tv.count_tags(NO_TAGS) == 0
    assert tv.extract_tags(NO_TAGS) == []


def test_extract_span_offsets_index_original_text():
    # span is (start, end) into the ORIGINAL text and slices back to the tag
    # token opening — usable for masking before an untagged-number scan.
    tags = tv.extract_tags(MIXED, kinds=("data",))
    for t in tags:
        s, e = t["span"]
        assert MIXED[s:e].startswith("**[data") or MIXED[s:e].startswith("[data]:")
    # first genuine data tag is the bare form on line 2
    s, e = tags[0]["span"]
    assert MIXED[s:e] == "**[data]"


def test_module_level_tag_re_exists():
    assert hasattr(tv, "TAG_RE")
    assert tv.TAG_RE.search("**[data]**") is not None


_CLI = Path(__file__).resolve().parents[2] / "scripts" / "tag_vocabulary.py"


def test_cli_prints_tab_separated_spans(tmp_path):
    import subprocess

    doc = tmp_path / "d.md"
    doc.write_text(MIXED)
    r = subprocess.run(
        [sys.executable, str(_CLI), str(doc)], capture_output=True, text=True
    )
    assert r.returncode == 0
    lines = r.stdout.splitlines()
    # default kinds: 3 data + 3 historical record genuine tags
    assert len(lines) == 6
    cols = lines[0].split("\t")
    assert len(cols) == 5  # line, tag, start, end, text
    line, tag, start, end, text = cols
    assert line == "2" and tag == "data"
    assert MIXED[int(start):int(end)] == "**[data]"
    assert text == "Line with a bare tag **[data]**."


def test_cli_no_tags_exits_zero_no_stdout(tmp_path):
    import subprocess

    doc = tmp_path / "e.md"
    doc.write_text(NO_TAGS)
    r = subprocess.run(
        [sys.executable, str(_CLI), str(doc)], capture_output=True, text=True
    )
    assert r.returncode == 0
    assert r.stdout == ""
