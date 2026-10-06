"""Real-page and real-input checks for the AFL Tables reconciliation (pilot traceability; DESIGN phase E).

The page tests read trimmed captures committed under ``tests/scvia/fixtures/reconciliation`` (see
``PROVENANCE.json`` for URL, capture time and original hash). Each asserted value was read by hand from
the cell named in its comment. The audit-input tests run only when the run directory named by
``SCVIA_RECON_RUN`` exists; they never touch the network.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from supercoach_via.reconciliation import cells as C
from supercoach_via.reconciliation import source as R
from supercoach_via.reconciliation.schema import STAT_FIELDS

pytestmark = pytest.mark.integration
FX = Path(__file__).resolve().parents[1] / "fixtures" / "reconciliation"
SITE = "https://afltables.com"
IDX = {f: i for i, f in enumerate(STAT_FIELDS)}


def test_committed_page_fixtures_match_their_recorded_hashes() -> None:
    for row in json.loads((FX / "PROVENANCE.json").read_text()):
        data = (FX / row["fixture"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["fixture_sha256"], row["fixture"]


def test_pilot_values_traceable_to_cells() -> None:
    p = R.read_profile(
        (FX / "profile_pendlebury.html").read_bytes(), f"{SITE}/afl/stats/players/S/Scott_Pendlebury.html"
    )
    g = p.games[0]  # Collingwood - 2006, Gm 1, Rd 10 vs Brisbane Lions
    assert (
        g.cells[IDX["kicks"]] == "5"
        and g.cells[IDX["disposals"]] == "11"
        and g.cells[IDX["time_on_ground_pct"]] == "66"
    )
    m = R.read_match((FX / "match_2021_r6.html").read_bytes(), f"{SITE}/afl/stats/games/2021/162020210424.html")
    ainsworth = next(x for x in m.players if x.name == "Ainsworth, Ben")
    assert ainsworth.cells[IDX["kicks"]] == "14" and ainsworth.cells[IDX["goals"]] == "3"  # Gold Coast row 1


def test_real_pre_1965_page_has_goals_only_so_everything_else_is_structurally_unrecorded() -> None:
    m = R.read_match((FX / "match_1975_r11.html").read_bytes(), f"{SITE}/afl/stats/games/1975/030719750616.html")
    notes = R.read_notes((FX / "notes.html").read_bytes())
    team = "Carlton"
    tc_totals = m.totals[team]
    ctx = C.AppearanceCtx(
        season=1975, is_final=False, match_usable=True, player_in_lineup=True,
        column_present=frozenset(STAT_FIELDS), team_totals=tc_totals, team_goals=19, team_behinds=None,
        available=frozenset(notes.availability[1975]), exception_fields=frozenset(f for f in STAT_FIELDS if f != "goals"),
        bad_fields=frozenset(), team_pct_recorded=False,
    )  # fmt: skip
    player = next(p for p in m.players if p.team == team)
    states = {
        f: c.state for f, c in zip(STAT_FIELDS, C.classify_appearance(player.cells, player.cells, ctx), strict=True)
    }
    assert states["kicks"] is C.St.NOT_RECORDED and states["tackles"] is C.St.NOT_RECORDED


def test_run_directory_inputs_when_present() -> None:
    run = os.environ.get("SCVIA_RECON_RUN")
    if not run or not (Path(run) / "plan.json").is_file():
        pytest.skip("SCVIA_RECON_RUN not set to a run directory")
    from supercoach_via.reconciliation.inventory import load_plan

    plan = load_plan(Path(run) / "plan.json")
    assert plan.scope.full_population and plan.source.reference_mode == "observed_current"
