"""TDD tests for the F6 blocking match-completeness gate inside refresh_matches().

Context: audit_match_rounds runs immediately after the match scrape (before the
~2h player scrape) but only WARNED, so a truncated match file let every
downstream step run on incomplete data and wasted the whole cycle. F6 promotes
that in-place audit to BLOCKING: an incomplete round raises SystemExit before
player scraping begins. Fail-open on a fixture outage (audit returns []).

No network: the scraper and the audit are both mocked.
"""
import os
import sys

import pytest
from unittest.mock import patch, MagicMock

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import refresh_data


def _patch_scraper():
    return patch.object(refresh_data, "MatchScraper", return_value=MagicMock())


def test_refresh_matches_aborts_on_incomplete_round():
    audit_result = [
        {"year": 2026, "round_num": 10, "n_matches": 3, "expected": 9,
         "severity": "WARNING", "missing": ["Carlton v Brisbane Lions"]},
    ]
    with _patch_scraper(), \
         patch.object(refresh_data.os.path, "exists", return_value=True), \
         patch.object(refresh_data, "audit_match_rounds", return_value=audit_result), \
         patch.dict(os.environ, {}, clear=False):
        os.environ.pop("ALLOW_INCOMPLETE_MATCHES", None)
        with pytest.raises(SystemExit) as exc:
            refresh_data.refresh_matches()
        assert exc.value.code != 0


def test_refresh_matches_continues_when_clean():
    audit_result = [
        {"year": 2026, "round_num": 9, "n_matches": 9, "expected": 9,
         "severity": "INFO", "missing": []},
    ]
    with _patch_scraper(), \
         patch.object(refresh_data.os.path, "exists", return_value=True), \
         patch.object(refresh_data, "audit_match_rounds", return_value=audit_result):
        refresh_data.refresh_matches()  # must not raise


def test_refresh_matches_fails_open_on_fixture_outage():
    """audit_match_rounds returns [] when the fixture can't be fetched -> no abort."""
    with _patch_scraper(), \
         patch.object(refresh_data.os.path, "exists", return_value=True), \
         patch.object(refresh_data, "audit_match_rounds", return_value=[]):
        refresh_data.refresh_matches()  # must not raise


def test_override_env_var_allows_incomplete():
    audit_result = [
        {"year": 2026, "round_num": 10, "n_matches": 3, "expected": 9,
         "severity": "WARNING", "missing": ["Carlton v Brisbane Lions"]},
    ]
    with _patch_scraper(), \
         patch.object(refresh_data.os.path, "exists", return_value=True), \
         patch.object(refresh_data, "audit_match_rounds", return_value=audit_result), \
         patch.dict(os.environ, {"ALLOW_INCOMPLETE_MATCHES": "1"}):
        refresh_data.refresh_matches()  # override -> must not raise
