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


def test_news_index_readme_is_exempt(tmp_path):
    """Locks the existing exemption at check-council-stamp.sh:172.

    The news INDEX is navigation, not a council-authored article, and has no
    pipeline to stamp it with — but it matches the `docs/news/*.md` rule that
    gates real articles. Without the exemption, routine index edits (adding a
    row, fixing a stray "125+ years" to the "130 years" used everywhere else)
    would be unshippable. The exemption was already there and untested; this
    pins it, and the test above still proves real articles stay gated.
    """
    r = _run(tmp_path, "docs/news/README.md", "# News desk\n\nIndex of articles.\n")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "skipped" in r.stdout


def test_new_unstamped_news_still_fails(tmp_path):
    r = _run(tmp_path, "docs/news/2026-08-01-fresh-take.md", "# Fresh\n\nNo stamp.\n")
    assert r.returncode == 1
    assert "missing the" in r.stderr and "provenance stamp" in r.stderr


def test_afl_doc_without_a_stamp_is_skipped(tmp_path):
    """docs/afl-*.md are opt-in-sticky, like the coaches briefs.

    Most are pipeline-generated surfaces with no council pipeline behind them, so
    gating them all would block every routine refresh.
    """
    r = _run(tmp_path, "docs/afl-season-2026.md", "# Season\n\nGenerated, no stamp.\n")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "skipped" in r.stdout


def test_afl_doc_becomes_gated_once_stamped(tmp_path):
    """The root cause of the afl-backtest-2026.md drift.

    That page carried five frozen figure blocks — a cumulative summary and team
    table stuck at R1-R13, a misses table at R1-R11 — while README's equivalents
    stayed current. The difference was not the content but the routing: nothing
    ever asked DataSentinel to look at it. Adding the marker must actually pull it
    into the gate, otherwise the marker is decoration.
    """
    r = _run(tmp_path, "docs/afl-backtest-2026.md",
             f"# Backtest\n\nStamped, so it must now be gated.\n\n{STAMP}\n")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "no DataSentinel PASS" in r.stderr or "audit" in r.stderr.lower()


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
