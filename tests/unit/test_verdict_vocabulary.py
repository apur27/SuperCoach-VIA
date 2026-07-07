"""F07 — record-sentinel-verdict.sh accepts the canonical verdict vocabulary.

Skeptic verdicts (PASS_WITH_CONCERNS / BLOCK) must be recordable with the same
content-hash discipline as DataSentinel's PASS/FAIL, and unknown tokens must be
rejected (exact-token match — no silent acceptance of a mistyped verdict).
"""
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RECORD = REPO / "scripts" / "record-sentinel-verdict.sh"


def _record(tmp_path: Path, verdict: str, agent: str = "Skeptic"):
    doc = tmp_path / "docs" / "d.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("# d\n\nprose\n")
    audit = tmp_path / "audit"
    audit.mkdir()
    env = {"COUNCIL_AUDIT_DIR": str(audit), "PATH": "/usr/bin:/bin"}
    r = subprocess.run(
        [str(RECORD), "--doc", "docs/d.md", "--verdict", verdict, "--agent", agent],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    return r, audit


@pytest.mark.parametrize("verdict", ["PASS", "FAIL", "BLOCK", "PASS_WITH_CONCERNS", "PASS_WITH_WARNINGS"])
def test_canonical_verdicts_accepted(tmp_path, verdict):
    r, audit = _record(tmp_path, verdict)
    assert r.returncode == 0, r.stderr
    records = list(audit.glob("*.json"))
    assert len(records) == 1
    assert f'"verdict":"{verdict}"' in records[0].read_text()
    assert '"agent_id":"Skeptic"' in records[0].read_text()


@pytest.mark.parametrize("bad", ["pass", "CONCERNS", "PASS_WITH_CONCERN", "OK", ""])
def test_unknown_verdict_rejected(tmp_path, bad):
    r, _ = _record(tmp_path, bad)
    assert r.returncode != 0, f"mistyped verdict {bad!r} was silently accepted"
