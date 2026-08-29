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


# ---------------------------------------------------------------------------
# Finals coverage in audit_match_rounds (Surveyor F2)
#
# Before F2 the audit coerced round_num with pd.to_numeric and kept only the
# rows that survived, so the ENTIRE finals series was invisible to the only
# exact completeness gate: a dropped Semi Final (the R10-2026 class of bug,
# where 6 of 9 rows were silently lost) could not be detected at all.
#
# The contract the fix must not break: fail OPEN when the fixture is
# unavailable, and NEVER flag a stage that simply has not been played yet --
# every finals stage is absent for the whole season, so flagging absence would
# block every cycle. "Absent entirely" = fine. "Present but short" = WARNING.
#
# No network: the H&A fixture fetch is patched at its boundary.
# ---------------------------------------------------------------------------

_MATCH_COLS = ["year", "round_num", "team_1_team_name", "team_2_team_name"]


def _write_matches(tmp_path, rows, year=2026, name="matches_2026.csv"):
    df = pd.DataFrame([[year] + list(r) for r in rows], columns=_MATCH_COLS)
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


_FULL_WEEK_ONE = [
    ("Qualifying Final", "Adelaide", "Brisbane Lions"),
    ("Qualifying Final", "Carlton", "Collingwood"),
    ("Elimination Final", "Essendon", "Fremantle"),
    ("Elimination Final", "Geelong", "Hawthorn"),
]


class TestFinalsAudit:
    def _audit(self, path):
        # No integer rounds in these fixtures, but patch the H&A fetch anyway so
        # a regression that reaches the network fails loudly instead of hanging.
        with patch.object(game_scraper, "fetch_round_fixture", return_value=None):
            return game_scraper.audit_match_rounds(path)

    def test_finals_rows_are_audited_at_all(self, tmp_path):
        """Regression: a file of only finals rows used to return zero issues."""
        path = _write_matches(tmp_path, _FULL_WEEK_ONE)
        issues = self._audit(path)
        assert issues, "finals rows must be audited, not dropped by to_numeric"
        assert {str(i["round_num"]) for i in issues} == {
            "Qualifying Final", "Elimination Final"}

    def test_complete_finals_stage_is_info_not_warning(self, tmp_path):
        path = _write_matches(tmp_path, _FULL_WEEK_ONE + [
            ("Semi Final", "Collingwood", "Essendon"),
            ("Semi Final", "Brisbane Lions", "Geelong"),
            ("Preliminary Final", "Adelaide", "Collingwood"),
            ("Preliminary Final", "Carlton", "Brisbane Lions"),
            ("Grand Final", "Adelaide", "Carlton"),
        ])
        issues = self._audit(path)
        assert [i for i in issues if i["severity"] == "WARNING"] == []
        assert len(issues) == 5

    def test_present_but_short_stage_flags_warning(self, tmp_path):
        """One Semi Final scraped where two are played -> WARNING."""
        path = _write_matches(tmp_path, _FULL_WEEK_ONE + [
            ("Semi Final", "Collingwood", "Essendon"),
        ])
        issues = self._audit(path)
        warns = [i for i in issues if i["severity"] == "WARNING"]
        assert len(warns) == 1
        assert str(warns[0]["round_num"]) == "Semi Final"
        assert warns[0]["n_matches"] == 1 and warns[0]["expected"] == 2

    def test_absent_stage_is_never_flagged(self, tmp_path):
        """Finals not yet played is the normal state -- must not block a cycle."""
        path = _write_matches(tmp_path, _FULL_WEEK_ONE)
        issues = self._audit(path)
        assert [i for i in issues if i["severity"] == "WARNING"] == []

    def test_short_code_finals_labels_are_audited(self, tmp_path):
        """The player corpus spells finals QF/EF/SF/PF/GF."""
        path = _write_matches(tmp_path, [
            ("QF", "Adelaide", "Brisbane Lions"),
            ("QF", "Carlton", "Collingwood"),
            ("EF", "Essendon", "Fremantle"),
            ("EF", "Geelong", "Hawthorn"),
            ("SF", "Collingwood", "Essendon"),
        ])
        issues = self._audit(path)
        warns = [i for i in issues if i["severity"] == "WARNING"]
        assert len(warns) == 1
        assert warns[0]["n_matches"] == 1 and warns[0]["expected"] == 2

    def test_dropped_earlier_stage_flagged_when_later_stage_present(self, tmp_path):
        """A stage that vanished entirely IS detectable once a later stage
        exists -- the series cannot have skipped it. This is the finals form of
        the R10 silently-dropped-round bug."""
        path = _write_matches(tmp_path, [
            ("Qualifying Final", "Adelaide", "Brisbane Lions"),
            ("Qualifying Final", "Carlton", "Collingwood"),
            # both Elimination Finals dropped
            ("Semi Final", "Collingwood", "Essendon"),
            ("Semi Final", "Brisbane Lions", "Geelong"),
        ])
        issues = self._audit(path)
        warns = [i for i in issues if i["severity"] == "WARNING"]
        assert len(warns) == 1
        assert str(warns[0]["round_num"]) == "Elimination Final"
        assert warns[0]["n_matches"] == 0 and warns[0]["expected"] == 2

    def test_unknown_round_label_is_ignored(self, tmp_path):
        path = _write_matches(tmp_path, [
            ("Wildcard Round", "Adelaide", "Brisbane Lions"),
        ])
        assert self._audit(path) == []


class TestHomeAndAwayAuditUnchanged:
    """Guard: the integer-round path keeps its exact fixture-comparison
    behaviour and its fail-open-on-outage contract."""

    def test_missing_ha_matchup_still_flagged(self, tmp_path):
        path = _write_matches(tmp_path, [
            (25, "Adelaide", "Brisbane Lions"),
            (25, "Carlton", "Collingwood"),
        ])
        fixture = {
            frozenset(("Adelaide", "Brisbane Lions")),
            frozenset(("Carlton", "Collingwood")),
            frozenset(("Essendon", "Fremantle")),
        }
        with patch.object(game_scraper, "fetch_round_fixture", return_value=fixture):
            issues = game_scraper.audit_match_rounds(path)
        warns = [i for i in issues if i["severity"] == "WARNING"]
        assert len(warns) == 1
        assert warns[0]["round_num"] == 25
        assert warns[0]["missing"] == ["Essendon v Fremantle"]

    def test_ha_fixture_outage_fails_open(self, tmp_path):
        path = _write_matches(tmp_path, [(25, "Adelaide", "Brisbane Lions")])
        with patch.object(game_scraper, "fetch_round_fixture", return_value=None):
            issues = game_scraper.audit_match_rounds(path)
        assert issues == []

    def test_mixed_ha_and_finals_file_audits_both(self, tmp_path):
        path = _write_matches(tmp_path, [
            (25, "Adelaide", "Brisbane Lions"),
        ] + _FULL_WEEK_ONE)
        fixture = {frozenset(("Adelaide", "Brisbane Lions"))}
        with patch.object(game_scraper, "fetch_round_fixture", return_value=fixture):
            issues = game_scraper.audit_match_rounds(path)
        labels = [str(i["round_num"]) for i in issues]
        assert "25" in labels and "Qualifying Final" in labels
        assert [i for i in issues if i["severity"] == "WARNING"] == []


class TestFinalsEraFloor:
    """Pre-2000 seasons used the final four/five/six and the McIntyre eight,
    whose stage counts differ from the modern 2-2-2-2-1 shape. Auditing them
    against modern counts flagged 103 historical files (verified against the
    real data/matches tree), so the finals audit has an era floor."""

    def test_pre_2000_finals_not_audited(self, tmp_path):
        path = _write_matches(tmp_path, [
            ("Semi Final", "Adelaide", "Brisbane Lions"),
            ("Preliminary Final", "Carlton", "Collingwood"),
            ("Grand Final", "Adelaide", "Carlton"),
        ], year=1975, name="matches_1975.csv")
        with patch.object(game_scraper, "fetch_round_fixture", return_value=None):
            issues = game_scraper.audit_match_rounds(path)
        assert issues == []

    def test_2000_season_is_audited(self, tmp_path):
        path = _write_matches(tmp_path, _FULL_WEEK_ONE + [
            ("Semi Final", "Adelaide", "Brisbane Lions"),
        ], year=2000, name="matches_2000.csv")
        with patch.object(game_scraper, "fetch_round_fixture", return_value=None):
            issues = game_scraper.audit_match_rounds(path)
        warns = [i for i in issues if i["severity"] == "WARNING"]
        assert [str(w["round_num"]) for w in warns] == ["Semi Final"]

    def test_drawn_grand_final_replay_is_not_a_warning(self, tmp_path):
        """2010 (and 1948, 1977) carry two Grand Final rows for one fixture.
        More rows than expected is never a completeness gap."""
        path = _write_matches(tmp_path, _FULL_WEEK_ONE + [
            ("Semi Final", "Collingwood", "Essendon"),
            ("Semi Final", "Brisbane Lions", "Geelong"),
            ("Preliminary Final", "Collingwood", "Geelong"),
            ("Preliminary Final", "St Kilda", "Adelaide"),
            ("Grand Final", "Collingwood", "St Kilda"),
            ("Grand Final", "St Kilda", "Collingwood"),
        ], year=2010, name="matches_2010.csv")
        with patch.object(game_scraper, "fetch_round_fixture", return_value=None):
            issues = game_scraper.audit_match_rounds(path)
        assert [i for i in issues if i["severity"] == "WARNING"] == []
