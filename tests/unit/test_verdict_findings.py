"""BL-05 — gate records must persist the REASONING, not just the verdict.

`record-sentinel-verdict.sh` wrote exactly {doc_path, doc_hash, verdict, ts,
agent_id}. The findings behind a FAIL or BLOCK lived only in the invoking agent's
transient stdout, so nothing durable recorded WHY a gate blocked or how many
distinct findings it carried.

That is not hypothetical. On 2026-07-26 a Skeptic BLOCK carrying FOUR findings was
read as one; three were never routed, and a full gate cycle was spent rediscovering
them. No audit trail could have caught the mistake, because a one-finding and a
four-finding BLOCK record were byte-identical. Separately, a BLOCK's findings were
lost entirely to a truncated pipe, forcing the whole review to be re-run.

So: persist the findings, and surface the COUNT where a human or agent reading the
log will see it.
"""
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RECORD = REPO / "scripts" / "record-sentinel-verdict.sh"

FINDINGS = [
    {"id": "SK-1", "severity": "BLOCK", "quote": "the same skill expressed twice",
     "issue": "identity asserted from a correlation"},
    {"id": "SK-2", "severity": "BLOCK", "quote": "the clearest example",
     "issue": "unsourced superlative"},
    {"id": "SK-3", "severity": "BLOCK", "quote": "this round",
     "issue": "season aggregate attributed to one round"},
    {"id": "SK-4", "severity": "BLOCK", "quote": "extended their lead",
     "issue": "comparative from a single snapshot"},
]


def _run(tmp_path, verdict, agent="Skeptic", findings=None, extra_env=None):
    doc = tmp_path / "docs" / "d.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("# d\n\nprose\n")
    audit = tmp_path / "audit"
    audit.mkdir(exist_ok=True)
    args = [str(RECORD), "--doc", "docs/d.md", "--verdict", verdict, "--agent", agent]
    if findings is not None:
        fp = tmp_path / "findings.json"
        fp.write_text(json.dumps(findings))
        args += ["--findings-file", str(fp)]
    env = {"COUNCIL_AUDIT_DIR": str(audit), "PATH": "/usr/bin:/bin"}
    env.update(extra_env or {})
    r = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True)
    recs = list(audit.glob("*.json"))
    return r, (json.loads(recs[0].read_text()) if recs else None)


def test_findings_are_persisted_with_the_verdict(tmp_path):
    r, rec = _run(tmp_path, "BLOCK", findings=FINDINGS)
    assert r.returncode == 0, r.stderr
    assert rec["verdict"] == "BLOCK"
    assert len(rec["findings"]) == 4, "findings not persisted"
    assert [f["id"] for f in rec["findings"]] == ["SK-1", "SK-2", "SK-3", "SK-4"]


def test_finding_count_is_recorded_and_echoed(tmp_path):
    """The count is the thing that would have caught the 4-read-as-1 mistake."""
    r, rec = _run(tmp_path, "BLOCK", findings=FINDINGS)
    assert rec["finding_count"] == 4
    assert "4" in r.stdout, f"count not surfaced to the operator: {r.stdout!r}"


def test_blocking_verdict_without_findings_is_flagged(tmp_path):
    """A BLOCK with no findings is unactionable — say so loudly."""
    r, rec = _run(tmp_path, "BLOCK")
    assert r.returncode == 0, "must still record; the verdict itself is not in doubt"
    assert rec["finding_count"] == 0
    combined = (r.stdout + r.stderr).lower()
    assert "no findings" in combined or "unactionable" in combined


def test_findings_can_be_required(tmp_path):
    """Opt-in strict mode, so the harness can enforce it without breaking callers."""
    r, _ = _run(tmp_path, "FAIL", agent="DataSentinel",
                extra_env={"COUNCIL_REQUIRE_FINDINGS": "1"})
    assert r.returncode != 0
    assert "findings" in (r.stdout + r.stderr).lower()


def test_clearing_verdict_needs_no_findings(tmp_path):
    r, rec = _run(tmp_path, "PASS", agent="DataSentinel",
                  extra_env={"COUNCIL_REQUIRE_FINDINGS": "1"})
    assert r.returncode == 0, r.stderr
    assert rec["finding_count"] == 0


def test_record_stays_valid_json_with_quotes_in_findings(tmp_path):
    """Findings quote the document, so they carry quotes, newlines and unicode."""
    tricky = [{"id": "X", "severity": "BLOCK",
               "quote": 'he said "the same skill" — twice\nand again',
               "issue": "quoting/encoding must not corrupt the record"}]
    r, rec = _run(tmp_path, "BLOCK", findings=tricky)
    assert r.returncode == 0, r.stderr
    assert rec["findings"][0]["quote"].startswith('he said "the same skill"')


def test_malformed_findings_file_fails_closed(tmp_path):
    doc = tmp_path / "docs" / "d.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("# d\n")
    audit = tmp_path / "audit"; audit.mkdir()
    bad = tmp_path / "bad.json"; bad.write_text("{not json")
    r = subprocess.run(
        [str(RECORD), "--doc", "docs/d.md", "--verdict", "BLOCK",
         "--agent", "Skeptic", "--findings-file", str(bad)],
        cwd=tmp_path, env={"COUNCIL_AUDIT_DIR": str(audit), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True)
    assert r.returncode != 0
    assert not list(audit.glob("*.json")), "wrote a record from unparseable findings"
