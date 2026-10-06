"""Local adapters: pinned snapshot and RAW legacy CSV strings (DESIGN section 5; T22, T25, T26)."""

from __future__ import annotations

from pathlib import Path

import pytest

from supercoach_via.reconciliation import inventory as inv
from supercoach_via.reconciliation import local as L
from supercoach_via.reconciliation.schema import STAT_FIELDS
from tests.scvia.unit import integrity_fixtures as fx

HEADER = (
    "team,year,games_played,opponent,round,result,jersey_num,kicks,marks,handballs,disposals,goals,behinds,hit_outs,"
    "tackles,rebound_50s,inside_50s,clearances,clangers,free_kicks_for,free_kicks_against,brownlow_votes,"
    "contested_possessions,uncontested_possessions,contested_marks,marks_inside_50,one_percenters,bounces,"
    "goal_assist,percentage_of_game_played,date\n"
)


@pytest.fixture
def snap(tmp_path: Path) -> tuple[Path, L.LocalSnapshot]:
    root = tmp_path / "data"
    fx.build(root)
    pin = inv.pin_snapshot(root, "current")
    return root, L.LocalSnapshot.open(root, pin)


def test_snapshot_rows_are_typed_and_ordered(snap: tuple[Path, L.LocalSnapshot]) -> None:
    _root, s = snap
    assert s.seasons == [1970, 2026]
    players = s.players()
    assert [p.key for p in players] == [f"legacy:p{i}" for i in range(1, 5)] and players[0].layer == "snapshot"
    games = s.games(2026)
    assert len(games) == 8 and all(g.season == 2026 and len(g.cells) == 23 for g in games)
    g = games[0]
    assert g.club in ("Alpha", "Beta") and g.cells[STAT_FIELDS.index("kicks")] == 1
    assert g.cells[STAT_FIELDS.index("brownlow_votes")] is None  # null stays null
    assert [x.origin for x in games] == sorted(
        {x.origin for x in games}, key=lambda o: [y.origin for y in games].index(o)
    )


def test_pinned_fragments_that_differ_from_the_plan_are_refused(
    snap: tuple[Path, L.LocalSnapshot], tmp_path: Path
) -> None:
    root, _s = snap
    pin = inv.pin_snapshot(root, "current")
    bad = pin.model_copy(update={"fragments": {**pin.fragments, "players": {"": "0" * 64}}})
    with pytest.raises(L.LocalDataError, match="differ"):
        L.LocalSnapshot.open(root, bad)


def test_snapshot_drift_is_reported_after_the_fact(snap: tuple[Path, L.LocalSnapshot]) -> None:
    root, s = snap
    assert s.drift() == []
    frag = next((root / "fragments" / "player_games").rglob("*.parquet"))
    frag.chmod(0o644)
    frag.write_bytes(frag.read_bytes() + b"x")
    assert any("changed during the audit" in d for d in s.drift())


def _legacy(
    tmp_path: Path,
    rows: str,
    personal: str
    | None = "first_name,last_name,born_date,debut_date,height,weight\nAnn,Able,01-07-1990,01-01-2010,180,80\n",
) -> L.LocalLegacy:
    d = tmp_path / "data" / "player_data"
    d.mkdir(parents=True, exist_ok=True)
    (d / "able_ann_01071990_performance_details.csv").write_text(HEADER + rows)
    if personal is not None:
        (d / "able_ann_01071990_personal_details.csv").write_text(personal)
    return L.LocalLegacy(tmp_path)


def test_legacy_cells_stay_raw_blank_is_not_rewritten_to_zero(tmp_path: Path) -> None:
    leg = _legacy(tmp_path, "Alpha,2020,1,Beta,3,W,5,7.0,,5,12,1,,,3.0,,,,2,,,,,,,,,,,66,2020-05-03\n")
    p = leg.read("able_ann_01071990")
    g = p.games[0]
    idx = {f: i for i, f in enumerate(STAT_FIELDS)}
    assert g.cells[idx["kicks"]] == 7 and g.cells[idx["marks"]] is None and g.raw_cells[idx["marks"]] == ""
    assert g.cells[idx["tackles"]] == 3 and g.raw_cells[idx["tackles"]] == "3.0"  # float text kept raw
    assert g.cells[idx["time_on_ground_pct"]] == 66 and g.layer == "legacy_csv"
    assert (p.first_name, p.last_name, p.birth_date) == ("Ann", "Able", "1990-07-01")
    assert g.origin.endswith("able_ann_01071990_performance_details.csv#1") and g.counter == 1 and g.stage == "3"


@pytest.mark.parametrize("raw", ["abc", "7.5", "-3", "1e3", "NaN", "1,000"])
def test_unreadable_legacy_numbers_are_kept_as_text_not_nulled(tmp_path: Path, raw: str) -> None:
    quoted = f'"{raw}"' if "," in raw else raw
    leg = _legacy(tmp_path, f"Alpha,2020,1,Beta,3,W,5,{quoted},,,,,,,,,,,,,,,,,,,,,,,,2020-05-03\n")
    g = leg.read("able_ann_01071990").games[0]
    assert g.cells[STAT_FIELDS.index("kicks")] == raw


def test_legacy_schema_drift_and_missing_personal_details_are_reported(tmp_path: Path) -> None:
    d = tmp_path / "data" / "player_data"
    d.mkdir(parents=True)
    (d / "x_y_01011990_performance_details.csv").write_text("team,year,mystery\nA,2020,5\n")
    p = L.LocalLegacy(tmp_path).read("x_y_01011990")
    assert any("unexpected columns" in x for x in p.problems) and any("missing columns" in x for x in p.problems)
    assert any("personal details file missing" in x for x in p.problems) and p.birth_date is None


def test_legacy_player_listing_is_sorted_and_streams(tmp_path: Path) -> None:
    _legacy(tmp_path, "Alpha,2020,1,Beta,3,W,5,7,,,,,,,,,,,,,,,,,,,,,,,,2020-05-03\n")
    d = tmp_path / "data" / "player_data"
    (d / "aaa_bob_02021991_performance_details.csv").write_text(HEADER)
    leg = L.LocalLegacy(tmp_path)
    assert leg.slugs() == ["aaa_bob_02021991", "able_ann_01071990"]
    assert [p.slug for p in leg.players()] == leg.slugs()


def test_bad_born_dates_are_problems_not_guesses(tmp_path: Path) -> None:
    leg = _legacy(tmp_path, "", "first_name,last_name,born_date,debut_date,height,weight\nAnn,Able,31-02-1990,x,1,1\n")
    p = leg.read("able_ann_01071990")
    assert p.birth_date is None and any("calendar date" in x for x in p.problems)
