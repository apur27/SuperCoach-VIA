"""Integration: the authorized 2026 AFLTables source check vs the real refreshed corpus.

No network. Uses the season page captured by the single authorized fetch on 2026-09-25
(tests/scvia/fixtures/raw/afltables/, scripts stripped; original sha256 recorded in its
header comment) and compares it with data/matches/matches_2026.csv read-only.
"""

from __future__ import annotations

import csv
from pathlib import Path

from supercoach_via.domain.ids import ClubRegistry
from supercoach_via.domain.schemas import CheckOutcome
from supercoach_via.ingest import afltables as at

REPO = Path(__file__).resolve().parents[3]
CAPTURED = REPO / "tests/scvia/fixtures/raw/afltables/seas_2026_captured_20260925.html"


def test_source_fixture_agrees_with_refreshed_corpus() -> None:
    reg = ClubRegistry.from_csv(REPO / "config/team_aliases.csv")
    fx = at.parse_season_page(CAPTURED.read_bytes(), season=2026, club_resolver=reg.resolve)
    assert fx.outcome is CheckOutcome.PASS, fx.issues
    source = {
        (m.local_start, frozenset((m.home_name, m.away_name))): (m.stage.label, m.home_score, m.away_score)
        for m in fx.completed()
    }
    with (REPO / "data/matches/matches_2026.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    corpus = {}
    for r in rows:
        hs = int(r["team_1_final_goals"]) * 6 + int(r["team_1_final_behinds"])
        as_ = int(r["team_2_final_goals"]) * 6 + int(r["team_2_final_behinds"])
        corpus[(r["date"], frozenset((r["team_1_team_name"], r["team_2_team_name"])))] = (r["round_num"], hs, as_)
    assert len(source) == len(corpus) == 217  # DATA_REFRESH.md
    assert source == corpus
    assert max(k[0] for k in source).startswith("2026-09-19")
