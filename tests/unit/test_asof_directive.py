"""F02a — the verify-asof directive: badge rendering + content-hash inclusion.

A doc frozen at a snapshot round declares `<!-- verify-asof: round=N -->`. Two guards:
  1. The trust badge renders the as-of round visibly (a snapshot cannot masquerade as
     current-data-verified).
  2. The directive is part of the canonical content hash — removing it changes the hash
     (so a stripped directive fails the audit cross-check), while the volatile badge/stamp
     lines still do NOT affect the hash.
(The DataSentinel round-cap *compute* — actually verifying tags against round<=N data — is
the Scientist-owned half, tracked separately.)
"""
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HASH = REPO / "scripts" / "council-content-hash.sh"

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "inject_trust_badge", REPO / "scripts" / "inject_trust_badge.py"
)
itb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(itb)


def test_badge_renders_asof_round():
    doc = "# Doc\n\n<!-- verify-asof: round=9 -->\n\nPendlebury **[data]** 436 games.\n"
    out = itb.inject_badge_or_strip(doc, "2026-07-07")
    badge = [l for l in out.splitlines() if itb.MARK in l][0]
    assert "verified as of Round 9" in badge, badge


def test_badge_current_when_no_directive():
    doc = "# Doc\n\nPendlebury **[data]** 436 games.\n"
    out = itb.inject_badge_or_strip(doc, "2026-07-07")
    badge = [l for l in out.splitlines() if itb.MARK in l][0]
    assert "verified against source data" in badge and "as of Round" not in badge


def test_parse_asof_round():
    assert itb.parse_asof_round("x <!-- verify-asof: round=17 --> y") == 17
    assert itb.parse_asof_round("no directive here") is None


def _hash(tmp_path: Path, text: str) -> str:
    f = tmp_path / "d.md"
    f.write_text(text)
    return subprocess.run([str(HASH), str(f)], capture_output=True, text=True, check=True).stdout.strip()


def test_directive_is_part_of_content_hash(tmp_path):
    base = "# Doc\n\nPendlebury 436 games.\n"
    with_directive = "# Doc\n\n<!-- verify-asof: round=9 -->\n\nPendlebury 436 games.\n"
    assert _hash(tmp_path, base) != _hash(tmp_path, with_directive), (
        "verify-asof directive was stripped from the hash — a stripped directive would go undetected"
    )


def test_badge_and_stamp_do_not_affect_hash(tmp_path):
    base = "# Doc\n\nPendlebury 436 games.\n"
    badged = (
        "# Doc\n> ✓ All 1 stat verified as of Round 9 · council-pipeline-gated · 2026-07-07\n\n"
        "Pendlebury 436 games.\n\n<!-- council-pipeline: Gaffer:SHIP@t -->\n"
    )
    assert _hash(tmp_path, base) == _hash(tmp_path, badged), "badge/stamp must be stripped from the hash"
