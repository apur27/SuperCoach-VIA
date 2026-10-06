"""``player_season_awards``: season-level award values the source prints only per season (pre-1984 Brownlow votes).

AFL Tables prints a player's Brownlow votes before 1984 only in his season summary; the per-game cells are blank.
A per-game table cannot hold them without inventing which games earned the votes, so they live here, one row per
player, season, club and award, each naming the profile page and body hash it was read from.
"""

from __future__ import annotations

from pathlib import Path

from supercoach_via.domain.schemas import TABLES
from supercoach_via.storage import snapshots
from supercoach_via.storage.queries import SnapshotQuery


def test_the_table_contract() -> None:
    spec = TABLES["player_season_awards"]
    assert spec.key == ("player_id", "season", "club_id", "award") and spec.partition_by is None
    required = {c.name for c in spec.columns if not c.nullable}
    assert {"player_id", "season", "club_id", "club_source_name", "award", "value", "source_path",
            "source_sha256"} <= required  # fmt: skip


def test_award_rows_merge_into_a_child_snapshot_and_are_queryable(tmp_path: Path) -> None:
    from tests.scvia.unit import integrity_fixtures as fx

    root = tmp_path / "var"
    base = fx.build(root)
    row = {c: None for c in TABLES["player_season_awards"].column_names}
    row.update(player_id="legacy:x", season=1955, club_id="alpha", club_source_name="Alpha", award="brownlow_votes",
               value=13, provenance="source_fetch", source_path="https://afltables.com/afl/stats/players/X/X.html",
               source_sha256="ab" * 32)  # fmt: skip
    cand = snapshots.apply_upserts(root, base, {"player_season_awards": [row]}, clock=lambda: base.created_at,
                                   code_version="t", status=base.status)  # fmt: skip
    with SnapshotQuery(root, cand.manifest, tables={"player_season_awards"}) as q:
        assert q.rows("SELECT player_id, season, value FROM player_season_awards") == [("legacy:x", 1955, 13)]
