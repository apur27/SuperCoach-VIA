"""Regression tests for the Brownlow proxy table's season-scaled column.

Two defects motivate this file:

1. The column was published under the heading "Proj. votes" while the
   generator's own comments state the proxy is dimensionless and NOT
   vote-interpretable. It routinely rendered values near +60 against a
   Brownlow all-time record of 36.
2. The season-length multiplier was a local literal (`HOME_AND_AWAY = 22`)
   duplicating `config.HOME_AND_AWAY_GAMES`, and the heading hard-coded
   its own copy of the number. Label and multiplier could drift apart
   silently — that pair drifting is the root-cause class here.
"""
from __future__ import annotations

import inspect
import os
import re
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import config  # noqa: E402
import update_team_analysis as uta  # noqa: E402


HEADING_RE = re.compile(r"Season proxy \(×(\d+)\)")


def _player_games() -> pd.DataFrame:
    """Small synthetic player-game frame in the shape
    `_load_player_games_with_names` returns."""
    rows = []
    specs = [
        ("daicos_nick_04011900", "Nick Daicos", "Collingwood", 32.0, 6.0, 14.0, 1.0, 3.0),
        ("oliver_clayton_04011900", "Clayton Oliver", "Melbourne", 30.0, 7.0, 15.0, 0.0, 4.0),
        ("neale_lachie_04011900", "Lachie Neale", "Brisbane Lions", 28.0, 6.0, 13.0, 0.0, 3.0),
        ("bench_barry_04011900", "Barry Bench", "Essendon", 12.0, 1.0, 4.0, 0.0, 2.0),
    ]
    for stem, display, team, disp, clr, cp, goals, clangers in specs:
        for rnd in range(1, 6):  # 5 games each — clears the min_games=3 filter
            rows.append({
                "player_stem": stem,
                "player_display": display,
                "team": team,
                "year": 2026,
                "round": rnd,
                "disposals": disp,
                "kicks": disp / 2,
                "handballs": disp / 2,
                "marks": 4.0,
                "goals": goals,
                "tackles": 3.0,
                "clearances": clr,
                "contested_possessions": cp,
                "clangers": clangers,
            })
    return pd.DataFrame(rows)


def test_heading_names_a_proxy_not_votes():
    """The published heading must not call a dimensionless proxy a vote count."""
    table = uta._build_brownlow_proxy_table(_player_games(), min_games=3)
    header = uta._build_brownlow_table_md(table.head(15)).splitlines()[0]

    assert "vote" not in header.lower(), (
        f"heading still claims votes: {header!r}"
    )
    assert HEADING_RE.search(header), (
        f"heading does not carry a 'Season proxy (×N)' label: {header!r}"
    )


def test_heading_multiplier_equals_the_multiplier_actually_applied():
    """The N in the heading must be the N used in the arithmetic.

    This is the anti-drift regression: a heading that names a scale factor
    other than the one applied is the defect class being fixed.
    """
    table = uta._build_brownlow_proxy_table(_player_games(), min_games=3)
    header = uta._build_brownlow_table_md(table.head(15)).splitlines()[0]

    match = HEADING_RE.search(header)
    assert match, f"no 'Season proxy (×N)' heading found: {header!r}"
    heading_n = int(match.group(1))

    scaled = table["season_proxy_scaled"]
    expected = table["brownlow_proxy_pg"] * heading_n
    pd.testing.assert_series_equal(
        scaled, expected, check_names=False,
        obj="scaled column vs heading multiplier",
    )


@pytest.mark.parametrize("n", [19, 23, 26])
def test_heading_and_multiplier_move_together(monkeypatch, n):
    """Change the one season-length constant and BOTH the heading and the
    arithmetic must follow it. Neither may carry a private copy."""
    monkeypatch.setattr(uta, "HOME_AND_AWAY_GAMES", n, raising=True)

    table = uta._build_brownlow_proxy_table(_player_games(), min_games=3)
    header = uta._build_brownlow_table_md(table.head(15)).splitlines()[0]

    assert f"(×{n})" in header, f"heading did not follow the constant: {header!r}"
    pd.testing.assert_series_equal(
        table["season_proxy_scaled"],
        table["brownlow_proxy_pg"] * n,
        check_names=False,
        obj=f"scaled column at HOME_AND_AWAY_GAMES={n}",
    )


def test_season_length_has_one_definition():
    """No private season-length literal inside the Brownlow generator —
    it must resolve to `config.HOME_AND_AWAY_GAMES`."""
    assert uta.HOME_AND_AWAY_GAMES == config.HOME_AND_AWAY_GAMES

    src = inspect.getsource(uta._build_brownlow_proxy_table)
    assert not re.search(r"^\s*HOME_AND_AWAY\s*=\s*\d+", src, re.M), (
        "the generator re-defines a local season-length literal; it must "
        "read the module-level constant sourced from config"
    )

    header_src = inspect.getsource(uta._build_brownlow_table_md)
    assert not re.search(r"×\s*\d", header_src), (
        "the heading hard-codes the multiplier instead of interpolating "
        "the constant"
    )


def test_scaled_column_is_a_pure_rescale_of_the_proxy():
    """Rank order must be identical to the per-game proxy — the column is a
    scale factor, not a second model."""
    table = uta._build_brownlow_proxy_table(_player_games(), min_games=3)
    assert list(table.sort_values("season_proxy_scaled", ascending=False)["player_stem"]) \
        == list(table.sort_values("brownlow_proxy_pg", ascending=False)["player_stem"])
