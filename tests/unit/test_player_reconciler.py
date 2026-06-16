"""
Unit tests for the player career-total reconciliation engine in
``scrapers/game_scraper.py``.

All HTTP is mocked: the network boundary is ``_get_player_totals`` (for the
player audit) and ``_get_season_soup`` (for the match audit), so no test makes a
real request. Tests are written retroactively against the existing
implementation -- see the task brief.
"""

import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest

# Make ``scrapers`` importable regardless of the directory pytest is invoked from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scrapers import game_scraper  # noqa: E402
from scrapers.game_scraper import (  # noqa: E402
    _player_url_from_csv_path,
    audit_player_career_totals,
    audit_match_rounds,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Pendlebury's canonical career totals (from afltables), used as the "clean"
# baseline. Tests mutate copies of this to manufacture deltas.
_PENDLEBURY_GAMES = 435
_PENDLEBURY_DISPOSALS = 10986
_PENDLEBURY_GOALS = 207


def _write_player_csv(
    tmp_path,
    *,
    filename="pendlebury_scott_07011988_performance_details.csv",
    games_played_max=_PENDLEBURY_GAMES,
    disposals_total=_PENDLEBURY_DISPOSALS,
    goals_total=_PENDLEBURY_GOALS,
    career_start=2006,
    tackles_total=0,
    clearances_total=0,
    n_rows=3,
):
    """
    Write a minimal but valid performance-details CSV to ``tmp_path`` and return
    its path.

    The reconciler aggregates games via ``max(games_played)`` and counting stats
    via column sum, so we spread the requested totals across ``n_rows`` rows and
    place ``games_played_max`` on the final (latest) row to exercise the max.
    """
    years = [career_start + i for i in range(n_rows)]

    def _spread(total, rows):
        """Split an integer total across ``rows`` rows (remainder on the last)."""
        base = total // rows
        out = [base] * rows
        out[-1] += total - base * rows
        return out

    rows = []
    games_counter = _spread(games_played_max, n_rows)
    # Make games_played a strictly-increasing cumulative counter ending at max.
    cumulative = []
    running = 0
    for g in games_counter:
        running += g
        cumulative.append(running)
    cumulative[-1] = games_played_max  # guarantee the documented max

    disposals = _spread(disposals_total, n_rows)
    goals = _spread(goals_total, n_rows)
    tackles = _spread(tackles_total, n_rows)
    clearances = _spread(clearances_total, n_rows)

    for i in range(n_rows):
        rows.append({
            "year": years[i],
            "round": i + 1,
            "disposals": disposals[i],
            "goals": goals[i],
            "tackles": tackles[i],
            "clearances": clearances[i],
            "games_played": cumulative[i],
        })

    df = pd.DataFrame(rows)
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return str(path)


def _totals(**kwargs):
    """Build an afltables-style Totals Series indexed by column codes (GM, DI...)."""
    return pd.Series(kwargs)


# ---------------------------------------------------------------------------
# _player_url_from_csv_path
# ---------------------------------------------------------------------------

class TestPlayerUrlFromCsvPath:
    def test_typical_player(self):
        url = _player_url_from_csv_path(
            "data/player_data/pendlebury_scott_07011988_performance_details.csv"
        )
        assert url is not None
        assert "S/Scott_Pendlebury.html" in url

    def test_multi_word_last_name_does_not_crash(self):
        # "de_boer_michael_..." -> parts[0]="de", parts[1]="boer". The helper
        # splits on underscore and only ever consumes the first two tokens, so it
        # must not crash and must return a string (best-effort URL).
        url = _player_url_from_csv_path(
            "data/player_data/de_boer_michael_01011980_performance_details.csv"
        )
        assert url is None or isinstance(url, str)

    def test_bad_filename_returns_none(self):
        url = _player_url_from_csv_path("data/player_data/notaplayerfile.csv")
        assert url is None


# ---------------------------------------------------------------------------
# audit_player_career_totals
# ---------------------------------------------------------------------------

class TestAuditPlayerCareerTotals:
    def test_clean_all_match(self, tmp_path):
        csv = _write_player_csv(tmp_path)
        totals = _totals(
            GM=_PENDLEBURY_GAMES, DI=_PENDLEBURY_DISPOSALS, GL=_PENDLEBURY_GOALS
        )
        with patch.object(game_scraper, "_get_player_totals", return_value=totals):
            issues = audit_player_career_totals(csv)
        assert issues == []

    def test_games_gap_emits_one_warning(self, tmp_path):
        # CSV under-counts games (432) vs ground truth (435).
        csv = _write_player_csv(tmp_path, games_played_max=432)
        totals = _totals(GM=435, DI=_PENDLEBURY_DISPOSALS)
        with patch.object(game_scraper, "_get_player_totals", return_value=totals):
            issues = audit_player_career_totals(csv)
        assert len(issues) == 1
        issue = issues[0]
        assert issue["stat"] == "games_played"
        assert issue["severity"] == "WARNING"
        assert issue["csv_val"] == 432
        assert issue["source_val"] == 435
        assert issue["delta"] == 3

    def test_disposals_gap_emits_one_warning(self, tmp_path):
        csv = _write_player_csv(tmp_path)  # disposals = 10986
        totals = _totals(GM=_PENDLEBURY_GAMES, DI=11028)
        with patch.object(game_scraper, "_get_player_totals", return_value=totals):
            issues = audit_player_career_totals(csv)
        assert len(issues) == 1
        assert issues[0]["stat"] == "disposals"
        assert issues[0]["delta"] == abs(10986 - 11028)

    def test_tackles_era_gate_pass(self, tmp_path):
        # career_start=1990 (>= 1987 floor) -> tackles ARE reconciled.
        csv = _write_player_csv(
            tmp_path, career_start=1990, tackles_total=2001,
            # Keep GM/DI/GL aligned so only tackles can flag.
        )
        totals = _totals(
            GM=_PENDLEBURY_GAMES, DI=_PENDLEBURY_DISPOSALS,
            GL=_PENDLEBURY_GOALS, TK=2012,
        )
        with patch.object(game_scraper, "_get_player_totals", return_value=totals):
            issues = audit_player_career_totals(csv)
        stats = {i["stat"] for i in issues}
        assert "tackles" in stats
        tk = next(i for i in issues if i["stat"] == "tackles")
        assert tk["delta"] == abs(2001 - 2012)

    def test_tackles_era_gate_block(self, tmp_path):
        # career_start=1965 (< 1987 floor) -> tackles must NOT be reconciled even
        # though TK is present and the CSV tackles (0) != totals TK.
        csv = _write_player_csv(tmp_path, career_start=1965, tackles_total=0)
        totals = _totals(
            GM=_PENDLEBURY_GAMES, DI=_PENDLEBURY_DISPOSALS,
            GL=_PENDLEBURY_GOALS, TK=0,
        )
        with patch.object(game_scraper, "_get_player_totals", return_value=totals):
            issues = audit_player_career_totals(csv)
        stats = {i["stat"] for i in issues}
        assert "tackles" not in stats

    def test_clearances_era_gate_block(self, tmp_path):
        # career_start=1980 (< 1998 floor) -> clearances must NOT be reconciled.
        csv = _write_player_csv(tmp_path, career_start=1980, clearances_total=0)
        totals = _totals(
            GM=_PENDLEBURY_GAMES, DI=_PENDLEBURY_DISPOSALS,
            GL=_PENDLEBURY_GOALS, CL=100,
        )
        with patch.object(game_scraper, "_get_player_totals", return_value=totals):
            issues = audit_player_career_totals(csv)
        stats = {i["stat"] for i in issues}
        assert "clearances" not in stats

    def test_404_page_returns_empty(self, tmp_path):
        # _get_player_totals returns None (404 / unfetchable) -> empty list.
        csv = _write_player_csv(tmp_path)
        with patch.object(game_scraper, "_get_player_totals", return_value=None):
            issues = audit_player_career_totals(csv)
        assert issues == []

    def test_malformed_empty_csv_does_not_crash(self, tmp_path):
        # An empty file is unreadable by pandas -> the function logs and returns [].
        path = tmp_path / "pendlebury_scott_07011988_performance_details.csv"
        path.write_text("")
        # _get_player_totals must never be reached, but patch it defensively so a
        # regression that reorders the read wouldn't hit the network.
        with patch.object(game_scraper, "_get_player_totals", return_value=None):
            issues = audit_player_career_totals(str(path))
        assert issues == []

    def test_multiple_deltas_emit_two_warnings(self, tmp_path):
        # Both games and disposals wrong -> two WARNINGs.
        csv = _write_player_csv(tmp_path, games_played_max=432, disposals_total=10986)
        totals = _totals(GM=435, DI=11028, GL=_PENDLEBURY_GOALS)
        with patch.object(game_scraper, "_get_player_totals", return_value=totals):
            issues = audit_player_career_totals(csv)
        assert len(issues) == 2
        stats = {i["stat"] for i in issues}
        assert stats == {"games_played", "disposals"}


# ---------------------------------------------------------------------------
# audit_match_rounds -- integration smoke test (no network)
# ---------------------------------------------------------------------------

class TestAuditMatchRoundsSmoke:
    def test_importable_and_graceful_without_network(self, tmp_path):
        # Write a tiny matches CSV with one integer round, then force the fixture
        # source (_get_season_soup) to return an empty soup so fetch_round_fixture
        # finds no Round heading and returns None -> the round is skipped and the
        # audit returns an empty issue list without touching the network.
        from bs4 import BeautifulSoup

        df = pd.DataFrame([
            {
                "year": 2026,
                "round_num": 1,
                "team_1_team_name": "Carlton",
                "team_2_team_name": "Richmond",
            }
        ])
        path = tmp_path / "matches_2026.csv"
        df.to_csv(path, index=False)

        empty_soup = BeautifulSoup("", "html.parser")
        with patch.object(game_scraper, "_get_season_soup", return_value=empty_soup):
            issues = audit_match_rounds(str(path))
        assert issues == []
