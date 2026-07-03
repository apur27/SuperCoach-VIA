"""Tests for scripts/inject_trust_badge.py — the product trust badge.

Every published council doc carries a visible verification line:
  ✓ All N stats verified against source data · council-pipeline-gated · <date>
N = number of file-backed [data] claims. The badge line MUST contain the token
`council-pipeline-gated` so scripts/council-content-hash.sh strips it — injecting
the badge must NOT change a doc's canonical content hash (else it would invalidate
the DataSentinel audit record). Injection is idempotent.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# inject_trust_badge imports tag_vocabulary; make scripts/ importable before loading it.
sys.path.insert(0, str(REPO / "scripts"))
HASH = REPO / "scripts" / "council-content-hash.sh"
SPEC = importlib.util.spec_from_file_location(
    "inject_trust_badge", REPO / "scripts" / "inject_trust_badge.py"
)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)

STAMP = "<!-- council-pipeline: DataSentinel:PASS(pass2)@t, Skeptic:PASS@t, Gaffer:SHIP@t -->"
DOC = (
    "# Dusty: The Storm\n\n"
    "Martin averaged **[data]** 28.4 disposals.\n"
    "He kicked **[data]** 3 goals in the 2017 Grand Final.\n"
    "A **[historical record]** three-time Norm Smith medallist.\n\n"
    f"{STAMP}\n"
)


def test_count_verified_stats_counts_data_tags():
    # 2 [data] tags; [historical record] is not a source-data-verified stat.
    assert mod.count_verified_stats(DOC) == 2


def test_count_covers_all_three_genuine_tag_forms():
    text = (
        "Bare **[data]** 5 marks.\n"
        "Spec **[data: file.csv ; filter=all ; column=goals ; aggregation=sum]** 338 goals.\n"
        "Inline [data]: 24.2 disposals.\n"
    )
    assert mod.count_verified_stats(text) == 3


def test_count_excludes_meta_references_and_stamp():
    # "[data] tags" prose and the provenance stamp are NOT real tagged stats.
    text = (
        "# T\n\nAdd verified [data] tags here later.\n\n"
        "Real **[data]** 5 marks.\n"
        "<!-- council-pipeline: every [data] figure re-read ; Gaffer:SHIP -->\n"
    )
    assert mod.count_verified_stats(text) == 1


def test_zero_tag_doc_gets_no_badge():
    text = "# Guide\n\nAll prose, an aside about [data] tags, no real stats.\n"
    assert mod.count_verified_stats(text) == 0
    out = mod.inject_badge_or_strip(text, date="2026-07-03")
    assert "council-pipeline-gated" not in out  # no false "All 0 stats" badge


def test_badge_singular_grammar():
    out = mod.inject_badge(DOC, n=1, date="2026-07-03")
    badge = next(l for l in out.splitlines() if "council-pipeline-gated" in l)
    assert "1 stat verified" in badge and "1 stats" not in badge


def test_inject_adds_visible_badge_after_h1():
    out = mod.inject_badge(DOC, n=2, date="2026-07-03")
    lines = out.splitlines()
    h1 = next(i for i, l in enumerate(lines) if l.startswith("# "))
    badge = lines[h1 + 1]
    assert "council-pipeline-gated" in badge
    assert "All 2 stats verified" in badge
    assert "2026-07-03" in badge
    assert "✓" in badge


def test_injection_is_idempotent():
    once = mod.inject_badge(DOC, n=2, date="2026-07-03")
    twice = mod.inject_badge(once, n=2, date="2026-07-03")
    assert once == twice
    assert once.count("council-pipeline-gated") == 1


def test_badge_does_not_change_canonical_hash(tmp_path):
    doc = tmp_path / "d.md"
    doc.write_text(DOC)
    before = subprocess.run([str(HASH), str(doc)], capture_output=True, text=True, check=True).stdout
    doc.write_text(mod.inject_badge(DOC, n=2, date="2026-07-03"))
    after = subprocess.run([str(HASH), str(doc)], capture_output=True, text=True, check=True).stdout
    assert before == after and len(before.strip()) == 64


def test_no_h1_puts_badge_at_top():
    out = mod.inject_badge("Just prose with **[data]** 5 marks.\n", n=1, date="2026-07-03")
    assert out.splitlines()[0].startswith(">") and "council-pipeline-gated" in out.splitlines()[0]
