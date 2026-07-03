"""F6 — requires_stamp() routing in check-council-stamp.sh.

- Legacy pre-gate news docs are exempt by EXACT filename.
- A brand-new unstamped docs/news/*.md must still hard-FAIL (no pattern relaxation).
- coaches-strategy-corner briefs are opt-in-sticky: gated only once they carry a stamp.

Run outside a git index (F4 falls back to the working file), so this isolates routing.
"""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHECK = REPO / "scripts" / "check-council-stamp.sh"
STAMP = "<!-- council-pipeline: DataSentinel:PASS@t, Skeptic:PASS@t, Gaffer:SHIP@t -->"
ENV = {"PATH": "/usr/bin:/bin"}


def _run(tmp_path, relpath, body):
    doc = tmp_path / relpath
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(body)
    env = {**ENV, "COUNCIL_AUDIT_DIR": str(tmp_path / "noaudit")}
    return subprocess.run([str(CHECK), relpath], cwd=tmp_path, env=env,
                          capture_output=True, text=True)


def test_legacy_news_exact_filename_is_exempt(tmp_path):
    r = _run(tmp_path, "docs/news/2026-05-13-voss-carlton.md", "# Voss\n\nNo stamp, legacy.\n")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "skipped" in r.stdout  # routed to SKIP, not gated


def test_new_unstamped_news_still_fails(tmp_path):
    r = _run(tmp_path, "docs/news/2026-08-01-fresh-take.md", "# Fresh\n\nNo stamp.\n")
    assert r.returncode == 1
    assert "missing the" in r.stderr and "provenance stamp" in r.stderr


def test_coaches_brief_unstamped_is_opt_in_skipped(tmp_path):
    r = _run(tmp_path, "docs/coaches-strategy-corner/some-brief.md", "# Brief\n\nLegacy, no stamp.\n")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "skipped" in r.stdout


def test_coaches_brief_becomes_gated_once_stamped(tmp_path):
    # Opt-in-sticky: a stamped brief IS gated. With enforce defaulting to 1 and no
    # audit record, it must FAIL (proving the stamp armed the gate).
    r = _run(tmp_path, "docs/coaches-strategy-corner/gated.md", f"# Brief\n\nProse.\n\n{STAMP}\n")
    assert r.returncode == 1
    assert "cannot be verified" in r.stderr  # gated, then failed on missing record
