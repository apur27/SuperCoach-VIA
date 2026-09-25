"""Brownlow proxy: legacy formula parity, proxy labelling, votes kept distinct (A04)."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import pytest

from supercoach_via.analytics import awards
from supercoach_via.storage.queries import SnapshotQuery
from tests.scvia.fixtures.analytics import synthetic as syn


def _rows(seed: int = 3) -> list[dict]:  # type: ignore[type-arg]
    rng = random.Random(seed)
    rows = []
    for p in range(25):
        club = "AB"[p % 2]
        for g in range(rng.randint(1, 8)):
            def v() -> int | None:
                return None if rng.random() < 0.1 else rng.randint(0, 35)

            rows.append(
                syn.pg(f"m{g}_{p % 2}", f"legacy:p{p:02d}_x_01012000", club, 2026, g + 1,
                       disposals=v(), clearances=v(), contested_possessions=v(), clangers=v(),
                       goals=v(), tackles=v(), kicks=v(), handballs=v(),
                       brownlow_votes=None if g % 3 else rng.choice([None, 0, 1, 3]))
            )
    return rows


@pytest.fixture
def snap(tmp_path: Path):  # type: ignore[no-untyped-def]
    rows = _rows()
    tables = {
        "players": [syn.player(f"legacy:p{p:02d}_x_01012000", f"P{p} X") for p in range(25)],
        "clubs": [syn.club("A", "Adelaide"), syn.club("B", "Brisbane Lions")],
        "player_games": rows,
        "matches": [syn.match(f"f{i}", 2026, i + 1, "A", "B", None, None, status="scheduled") for i in range(23)],
    }
    return tmp_path, syn.build(tmp_path, tables), rows


class TestProxy:
    def test_per_game_proxy_matches_legacy_formula(self, snap) -> None:  # type: ignore[no-untyped-def]
        import update_team_analysis as uta

        root, manifest, rows = snap
        legacy_in = pd.DataFrame(
            [
                {
                    "player_stem": r["player_id"].removeprefix("legacy:"),
                    "player_display": r["player_id"],
                    "team": {"A": "Adelaide", "B": "Brisbane Lions"}[r["club_id"]],
                    **{k: r.get(k) for k in ("disposals", "clearances", "contested_possessions",
                                             "clangers", "goals", "tackles", "kicks", "handballs")},
                }
                for r in rows
            ]
        )
        want = uta._build_brownlow_proxy_table(legacy_in, min_games=3)
        with SnapshotQuery(root, manifest) as q:
            table = awards.brownlow_proxy(q, 2026, ineligible={})
        got = {r.player_id.removeprefix("legacy:"): r for r in table.rows}
        assert [r.player_id.removeprefix("legacy:") for r in table.rows] == want["player_stem"].tolist()
        for w in want.itertuples():
            g = got[w.player_stem]
            assert g.proxy_per_game == pytest.approx(w.brownlow_proxy_pg, abs=1e-12)
            assert g.games == w.games_played
            assert g.rank == w.rank
            assert g.club == w.team

    def test_labelled_as_proxy_never_prediction(self, snap) -> None:  # type: ignore[no-untyped-def]
        root, manifest, _ = snap
        with SnapshotQuery(root, manifest) as q:
            table = awards.brownlow_proxy(q, 2026, ineligible={})
        text = " ".join([table.label, table.scale_label]).lower()
        assert "not a vote count and not a probability" in table.method
        assert "proxy" in table.label.lower()
        for banned in ("predict", "probability", "projected votes", "chance"):
            assert banned not in text
        assert table.version == awards.BROWNLOW_PROXY_VERSION

    def test_observed_votes_are_kept_distinct(self, snap) -> None:  # type: ignore[no-untyped-def]
        root, manifest, rows = snap
        with SnapshotQuery(root, manifest) as q:
            table = awards.brownlow_proxy(q, 2026, ineligible={})
        for r in table.rows:
            recorded = [x["brownlow_votes"] for x in rows
                        if x["player_id"] == r.player_id and x.get("brownlow_votes") is not None]
            assert r.observed_votes == (sum(recorded) if recorded else None)
            assert r.votes_observed_games == len(recorded)

    def test_season_scale_comes_from_the_fixture(self, snap) -> None:  # type: ignore[no-untyped-def]
        root, manifest, _ = snap
        with SnapshotQuery(root, manifest) as q:
            table = awards.brownlow_proxy(q, 2026, ineligible={})
        assert table.season_games == 23
        assert "fixture" in table.scale_basis
        r = table.rows[0]
        assert r.season_proxy_scaled == pytest.approx(r.proxy_per_game * 23)

    def test_ineligibility_is_explicit_and_sourced(self, snap) -> None:  # type: ignore[no-untyped-def]
        root, manifest, _ = snap
        reg = {"legacy:p00_x_01012000": awards.Ineligibility("Suspended (2026 season)", "test registry")}
        with SnapshotQuery(root, manifest) as q:
            table = awards.brownlow_proxy(q, 2026, ineligible=reg)
        flagged = [r for r in table.rows if r.ineligible]
        assert [r.player_id for r in flagged] == ["legacy:p00_x_01012000"]
        assert flagged[0].ineligible_reason == "Suspended (2026 season)"
        assert flagged[0].ineligible_source == "test registry"

    def test_default_registry_is_the_legacy_one(self) -> None:
        import update_team_analysis as uta

        reg = awards.default_ineligibility(2026)
        assert {k.removeprefix("legacy:") for k in reg} == set(uta.BROWNLOW_INELIGIBLE_2026)
        assert awards.default_ineligibility(2025) == {}

    def test_min_games_filter(self, snap) -> None:  # type: ignore[no-untyped-def]
        root, manifest, _ = snap
        with SnapshotQuery(root, manifest) as q:
            table = awards.brownlow_proxy(q, 2026, ineligible={})
        assert all(r.games >= 3 for r in table.rows)
