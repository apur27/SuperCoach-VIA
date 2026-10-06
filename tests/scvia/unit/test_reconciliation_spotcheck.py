"""Independent spot-check of applied corrections with the PRODUCTION ingest parser (``ingest.afltables``), which
shares no code with the reconciliation reader, run against real committed page fixtures."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "reconciliation_spotcheck.py"
FX = Path(__file__).resolve().parents[1] / "fixtures" / "reconciliation"
URL = "https://afltables.com/afl/stats/players/S/Scott_Pendlebury.html"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("recon_spotcheck", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _game(m: ModuleType):  # type: ignore[no-untyped-def]
    page = m.parse_page((FX / "profile_pendlebury.html").read_bytes(), URL)
    return m.find_game(page, season=2006, team="Collingwood", opponent="Brisbane Lions", stage="10")


def test_the_ingest_parser_locates_the_corrected_game_on_the_player_page() -> None:
    m = _load()
    g = _game(m)
    assert g is not None and g.stats["kicks"] == 5 and g.stats["disposals"] == 11


def test_verdicts_for_a_value_a_proven_zero_and_a_nulled_cell() -> None:
    m = _load()
    g = _game(m)
    assert m.cell_verdict(g, "kicks", 5) == "agrees"
    assert m.cell_verdict(g, "kicks", 9) == "disagrees"
    assert m.cell_verdict(g, "behinds", 0) == "agrees"  # a proven zero: the page prints a blank
    assert m.cell_verdict(g, "behinds", None) == "agrees"  # a nulled cell: the page prints nothing
    assert m.cell_verdict(g, "tackles", None) == "disagrees"  # the page prints 3: null would be wrong


def test_a_game_that_is_not_on_the_page_is_reported_not_guessed() -> None:
    m = _load()
    page = m.parse_page((FX / "profile_pendlebury.html").read_bytes(), URL)
    assert m.find_game(page, season=2006, team="Collingwood", opponent="Nobody", stage="10") is None
