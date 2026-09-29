"""Source-blank semantics (domain.blanks) against hand-annotated evidence."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from supercoach_via.domain import blanks
from supercoach_via.domain.schemas import PLAYER_STAT_COLUMNS
from supercoach_via.ingest import afltables
from supercoach_via.ingest.reconcile import load_policy

REPO = Path(__file__).resolve().parents[3]
ANN = json.loads((REPO / "tests/scvia/fixtures/zero_semantics/annotations.json").read_text())
RECORDED = load_policy(REPO / "config").coverage.recorded_from
ALIAS = {"tog": "time_on_ground_pct"}


def _table(case: dict[str, Any]) -> pa.Table:
    rows = []
    for r in case["match"]:
        row: dict[str, Any] = {"match_id": "m1", "player_id": r["id"], "club_id": "c", "season": case["season"]}
        row |= {s: None for s in PLAYER_STAT_COLUMNS}
        row |= {ALIAS.get(k, k): v for k, v in r.items() if k != "id"}
        rows.append(row)
    return pa.Table.from_pylist(rows)


@pytest.mark.parametrize("case", ANN["row_cases"], ids=[c["name"] for c in ANN["row_cases"]])
def test_annotated_row_cases(case: dict[str, Any]) -> None:
    out, _counts = blanks.resolve_blanks(_table(case), {"m1": case["stage_type"]}, RECORDED)
    got = {r["player_id"]: r for r in out.to_pylist()}
    for pid, want in case["expect"].items():
        for k, v in want.items():
            assert got[pid][ALIAS.get(k, k)] == v, (case["name"], pid, k)


def test_resolution_is_idempotent_and_counts_conversions() -> None:
    case = ANN["row_cases"][0]
    once, counts = blanks.resolve_blanks(_table(case), {"m1": "regular"}, RECORDED)
    twice, again = blanks.resolve_blanks(once, {"m1": "regular"}, RECORDED)
    assert once.equals(twice)
    assert counts["goals"] == 1 and sum(again.values()) == 0


def test_captured_page_through_the_production_adapter() -> None:
    """The adapter keeps blanks as None; the shared rule turns reported-column blanks into zeros."""
    sha = "08a2bf5968e7ebce40e1cc4f5d2440d41498e1aba2e965a10ac703866e1b83a8"
    raw = gzip.decompress((REPO / f"docs/rewrite/evidence/b1/raw/{sha}.html.gz").read_bytes())
    detail = afltables.parse_match_detail(raw, season=2026, game_id="131820260329")
    rows = [{"match_id": "m", "player_id": f"{p.team}|{p.source_name}", "club_id": p.team, "season": 2026, **p.stats}
            for p in detail.players]  # fmt: skip
    out, _ = blanks.resolve_blanks(pa.Table.from_pylist(rows), {"m": "regular"}, RECORDED)
    got = {r["player_id"]: r for r in out.to_pylist()}
    for case in ANN["page_cases"]:
        assert got[f"{case['team']}|{case['player']}"][case["stat"]] == case["meaning"], case


def test_rows_of_other_matches_do_not_count_as_evidence() -> None:
    rows = [
        {"match_id": "m1", "player_id": "a", "club_id": "c", "season": 2021, "kicks": 5, "bounces": 2},
        {"match_id": "m2", "player_id": "b", "club_id": "c", "season": 2021, "kicks": 5, "bounces": None},
    ]
    rows = [{**{s: None for s in PLAYER_STAT_COLUMNS}, **r} for r in rows]
    out, _ = blanks.resolve_blanks(pa.Table.from_pylist(rows), {"m1": "regular", "m2": "regular"}, RECORDED)
    assert out.to_pylist()[1]["bounces"] is None  # m2 never reported bounces
