"""Integrity checker on a real snapshot and its sealed release (integration tier).

Inputs are explicit: ``SCVIA_INTEGRITY_DATA_ROOT`` and ``SCVIA_INTEGRITY_RELEASE_DIR`` (absolute),
optionally ``SCVIA_INTEGRITY_EVIDENCE`` (extra evidence directory). Without them the test is
skipped with that reason; it is never reported as a pass.

It asserts what must hold for any correct retained artifact: the pinned bytes verify, the
release agrees with the facts it was built from, captured source pages agree cell for cell,
every published resource had an executed semantic comparison, and the audit is complete.
The curated article sources are this checkout (``config/public_content.toml``). Point it at a
correct artifact (the follow-up candidate); the first retained artifacts FAIL
``source.match_pages`` because their blanks contradict the captured pages. Real data anomalies (for example a current-season value that
contradicts its pinned source) are reported by the checker, not hidden by this test.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from supercoach_via.integrity.report_schema import validate
from supercoach_via.integrity.runner import AuditOptions, run_audit

REPO = Path(__file__).resolve().parents[3]
DATA = os.environ.get("SCVIA_INTEGRITY_DATA_ROOT")
RELEASE = os.environ.get("SCVIA_INTEGRITY_RELEASE_DIR")

pytestmark = pytest.mark.skipif(
    not (DATA and RELEASE), reason="set SCVIA_INTEGRITY_DATA_ROOT and SCVIA_INTEGRITY_RELEASE_DIR to the retained artifacts"
)


@pytest.fixture(scope="module")
def report() -> dict:
    assert DATA and RELEASE
    evidence = [REPO / "docs/rewrite/evidence/b1/raw"]
    if os.environ.get("SCVIA_INTEGRITY_EVIDENCE"):
        evidence.append(Path(os.environ["SCVIA_INTEGRITY_EVIDENCE"]))
    res = run_audit(AuditOptions(data_root=Path(DATA), release_dir=Path(RELEASE), scope="full",
                                 as_of=os.environ.get("SCVIA_INTEGRITY_AS_OF", "2026-09-28T12:00:00Z"),
                                 evidence_dirs=tuple(evidence), workers=int(os.environ.get("SCVIA_INTEGRITY_WORKERS", "2")),
                                 content_root=REPO, content_manifest=REPO / "config/public_content.toml"))
    return res.report


def test_report_is_valid_and_complete(report: dict) -> None:
    validate(report)
    assert report["scope"]["complete"] is True
    assert report["outcome"] in ("PASS", "FAIL")


@pytest.mark.parametrize(
    "check_id",
    ["storage.identity", "storage.fragments", "contract.keys", "contract.values", "release.artifact",
     "release.validate", "release.public", "release.derived", "release.forecast", "release.content",
     "release.coverage", "source.match_pages", "source.player_pages", "inputs.stable"],
)
def test_integrity_of_bytes_release_and_captured_sources(report: dict, check_id: str) -> None:
    check = next(c for c in report["checks"] if c["check_id"] == check_id)
    assert check["status"] == "PASS", check


def test_the_grand_final_capture_was_compared(report: dict) -> None:
    cov = report["coverage"]["source_capture"]
    assert cov["pages_compared"] >= 1 and cov["player_games_compared"] >= 46
    pub = report["coverage"]["public_compare"]
    assert pub["match_details"] > 17000 and pub["player_details"] > 13000


def test_every_published_resource_was_semantically_compared(report: dict) -> None:
    sem = report["coverage"]["semantic"]
    assert report["scope"]["semantic_complete"] is True
    assert sem["uncompared"] == {}
    for kind in ("team_season", "history_table", "lists_season", "overview", "quality", "downloads", "download"):
        assert sem["by_type"][kind]["compared"] == sem["by_type"][kind]["resources"] > 0, kind


def test_the_b1_player_pages_were_compared(report: dict) -> None:
    pp = report["coverage"]["player_pages"]
    assert pp["pages_compared"] == 3 and pp["rows_compared"] == pp["rows_on_pages"] > 0
