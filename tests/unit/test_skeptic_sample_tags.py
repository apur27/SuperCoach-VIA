"""Tests for scripts/skeptic_sample_tags.py — deterministic Skeptic tag sampling.

Skeptic spot-probes a fixed-size subset of a draft's verifiable tags. The subset
MUST be a deterministic function of the document path (hash(doc_path) -> indices),
so the same doc always probes the same tags and a prompt/run is reproducible.
"""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Put scripts/ on the path BEFORE exec'ing the target module. skeptic_sample_tags
# is loaded by file path (spec_from_file_location), which does NOT add scripts/ to
# sys.path — so once skeptic_sample_tags.py gains `import tag_vocabulary`, the
# nested import would raise ImportError at exec time and fail collection. Same
# pattern as test_tag_vocabulary.py.
sys.path.insert(0, str(REPO / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "skeptic_sample_tags", REPO / "scripts" / "skeptic_sample_tags.py"
)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


DOC = """# Test brief

Macrae averaged **[data]** 28.4 disposals last five.
The Dogs won by **[data]** 14 points.
Gawn is a **[historical record]** three-time All-Australian.
Bont had **[data]** 9 clearances.
An **[unverified]** claim about crowd mood.
Neale tallied **[data]** 31 touches.
"""


def test_extract_tags_finds_verifiable_tags():
    tags = mod.extract_tags(DOC)
    # 4 [data] + 1 [historical record] = 5 verifiable; [unverified] is excluded.
    assert len(tags) == 5
    assert all("line" in t and "tag" in t for t in tags)
    assert tags[0]["tag"] == "data"


def test_selection_is_deterministic_for_a_path():
    a = mod.select_tag_indices("docs/news/foo.md", 5, n=3)
    b = mod.select_tag_indices("docs/news/foo.md", 5, n=3)
    assert a == b


def test_selection_is_distinct_in_range_and_sorted():
    idx = mod.select_tag_indices("docs/news/foo.md", 5, n=3)
    assert len(idx) == 3
    assert len(set(idx)) == 3
    assert all(1 <= i <= 5 for i in idx)
    assert idx == sorted(idx)


def test_path_changes_selection():
    a = mod.select_tag_indices("docs/news/foo.md", 12, n=3)
    b = mod.select_tag_indices("docs/news/bar.md", 12, n=3)
    assert a != b  # different paths should (almost always) probe different tags


def test_fewer_tags_than_sample_returns_all():
    assert mod.select_tag_indices("docs/news/foo.md", 2, n=3) == [1, 2]
    assert mod.select_tag_indices("docs/news/foo.md", 0, n=3) == []


def test_end_to_end_sample_from_doc(tmp_path):
    doc = tmp_path / "brief.md"
    doc.write_text(DOC)
    sample = mod.sample(str(doc), n=3)
    assert len(sample) == 3
    for s in sample:
        assert set(s) >= {"index", "line", "tag", "text"}
    # deterministic given the path
    assert mod.sample(str(doc), n=3) == sample
