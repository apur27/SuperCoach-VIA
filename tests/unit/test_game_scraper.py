"""
Unit tests for the career-year-overlap name-collision guard in
``scrapers/game_scraper.py``.

Bug: ``_player_url_from_csv_path`` derives the afltables profile URL from the
player's NAME only, discarding the DOB in the filename. When two players share a
name (Maurice Rioli Sr/Jr, the two Matthew Kennedys), the audit fetches the
WRONG player's page and reports spurious WARNING deltas.

Fix: before reconciling, compare the afltables page's career year range against
our CSV's year range. If they do not overlap (within a 2-year slack), it is a
name collision -- log an info line and return zero issues.

All HTTP is mocked at ``_get_player_totals`` (the network boundary); no test
makes a real request.
"""

import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scrapers import game_scraper  # noqa: E402
from scrapers.game_scraper import audit_player_career_totals  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_player_csv(tmp_path, *, career_start, career_end,
                      filename="kennedy_matthew_01011996_performance_details.csv",
                      games_played_max=100, disposals_total=1500, goals_total=120):
    """
    Write a minimal performance-details CSV spanning career_start..career_end.

    games_played is a cumulative counter ending at games_played_max; disposals
    and goals are spread so their column sums equal the requested totals.
    """
    years = list(range(career_start, career_end + 1))
    n_rows = len(years)

    def _spread(total):
        base = total // n_rows
        out = [base] * n_rows
        out[-1] += total - base * n_rows
        return out

    disposals = _spread(disposals_total)
    goals = _spread(goals_total)
    games_step = _spread(games_played_max)
    cumulative, running = [], 0
    for g in games_step:
        running += g
        cumulative.append(running)
    cumulative[-1] = games_played_max

    rows = [{
        "year": years[i],
        "round": i + 1,
        "disposals": disposals[i],
        "goals": goals[i],
        "games_played": cumulative[i],
    } for i in range(n_rows)]

    path = tmp_path / filename
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _totals_payload(*, year_min, year_max, **stat_cols):
    """
    Build the dict that the patched ``_get_player_totals`` returns:
    ``{'totals': Series, 'year_min': int, 'year_max': int}``.
    """
    return {
        "totals": pd.Series(stat_cols),
        "year_min": year_min,
        "year_max": year_max,
    }


# ---------------------------------------------------------------------------
# Name-collision guard
# ---------------------------------------------------------------------------

class TestCareerYearOverlapGuard:
    def test_non_overlapping_years_skip_as_collision(self, tmp_path, capsys):
        """
        Our CSV is the modern Matthew Kennedy (2016-2026) but afltables returned
        the 1980s namesake (1982-1993). The year ranges do not overlap -> treat
        as a name collision, emit zero issues, and log the collision line.

        Without the guard, the mismatched totals would raise spurious WARNINGs.
        """
        csv = _write_player_csv(tmp_path, career_start=2016, career_end=2026,
                                games_played_max=100)
        # Wrong player's totals: large career, totally different numbers.
        payload = _totals_payload(year_min=1982, year_max=1993,
                                  GM=300, DI=8000, GL=400)
        with patch.object(game_scraper, "_get_player_totals", return_value=payload):
            issues = audit_player_career_totals(csv)

        assert issues == [], "collision should yield zero issues, not WARNINGs"
        out = capsys.readouterr().out
        assert "name collision" in out
        assert "career years don't overlap" in out

    def test_matching_years_reconcile_normally(self, tmp_path):
        """
        afltables and the CSV both span 2020-2026 and the totals agree ->
        the guard must not fire and the (clean) comparison runs: zero issues.
        """
        csv = _write_player_csv(tmp_path, career_start=2020, career_end=2026,
                                games_played_max=100, disposals_total=1500,
                                goals_total=120)
        payload = _totals_payload(year_min=2020, year_max=2026,
                                  GM=100, DI=1500, GL=120)
        with patch.object(game_scraper, "_get_player_totals", return_value=payload):
            issues = audit_player_career_totals(csv)

        assert issues == [], "matching years + matching totals should reconcile clean"

    def test_matching_years_real_delta_still_flags(self, tmp_path):
        """
        Years overlap (guard passes) but the CSV genuinely under-counts games.
        The guard must NOT mask a real reconciliation WARNING.
        """
        csv = _write_player_csv(tmp_path, career_start=2020, career_end=2026,
                                games_played_max=97)
        payload = _totals_payload(year_min=2020, year_max=2026,
                                  GM=100, DI=1500, GL=120)
        with patch.object(game_scraper, "_get_player_totals", return_value=payload):
            issues = audit_player_career_totals(csv)

        games_issues = [i for i in issues if i["stat"] == "games_played"]
        assert len(games_issues) == 1
        assert games_issues[0]["delta"] == 3

    def test_overlapping_years_reconcile_normally(self, tmp_path):
        """
        afltables 2018-2026 vs CSV 2020-2026 -> ranges overlap (2020-2026), so
        the guard passes and the comparison runs. Totals agree -> zero issues.
        """
        csv = _write_player_csv(tmp_path, career_start=2020, career_end=2026,
                                games_played_max=100, disposals_total=1500,
                                goals_total=120)
        payload = _totals_payload(year_min=2018, year_max=2026,
                                  GM=100, DI=1500, GL=120)
        with patch.object(game_scraper, "_get_player_totals", return_value=payload):
            issues = audit_player_career_totals(csv)

        assert issues == [], "overlapping ranges should reconcile normally"

    def test_adjacent_years_within_slack_reconcile_normally(self, tmp_path):
        """
        afltables ends 2019, CSV starts 2020 -> ranges touch but do not strictly
        overlap. The 2-year slack must keep this as a normal reconcile, not a
        collision (a one-season scraping gap should not be misread as a namesake).
        """
        csv = _write_player_csv(tmp_path, career_start=2020, career_end=2026,
                                games_played_max=100, disposals_total=1500,
                                goals_total=120)
        payload = _totals_payload(year_min=2012, year_max=2019,
                                  GM=100, DI=1500, GL=120)
        with patch.object(game_scraper, "_get_player_totals", return_value=payload):
            issues = audit_player_career_totals(csv)

        assert issues == [], "ranges within 2-year slack should not be flagged as collision"

    def test_year_range_unavailable_falls_back_to_reconcile(self, tmp_path):
        """
        If the year range can't be parsed (year_min/year_max are None), the guard
        must NOT skip -- it falls back to running the comparison so real deltas
        are still caught.
        """
        csv = _write_player_csv(tmp_path, career_start=2020, career_end=2026,
                                games_played_max=97)
        payload = _totals_payload(year_min=None, year_max=None,
                                  GM=100, DI=1500, GL=120)
        with patch.object(game_scraper, "_get_player_totals", return_value=payload):
            issues = audit_player_career_totals(csv)

        games_issues = [i for i in issues if i["stat"] == "games_played"]
        assert len(games_issues) == 1, "missing year range must not suppress real deltas"
