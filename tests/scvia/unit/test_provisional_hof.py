import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "provisional_hof", Path(__file__).parents[3] / "scripts/refresh_provisional_hof.py"
)


def module():
    m = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(m)
    return m


def test_leaders_preserve_names_ties_missing_and_observed_denominators():
    m = module()
    frame = pd.DataFrame(
        [
            dict(player_id="b", total=4.0, observed_games=2, mean=2.0),
            dict(player_id="a", total=4.0, observed_games=1, mean=4.0),
            dict(player_id="c", total=0.0, observed_games=1, mean=0.0),
            dict(player_id="d", total=float("nan"), observed_games=0, mean=float("nan")),
        ]
    )
    rows = m.leader_rows(frame, {"a": "Same", "b": "Same", "c": "Zero"})
    assert [(r["rank"], r["player_id"], r["name"]) for r in rows] == [
        (1, "a", "Same"),
        (1, "b", "Same"),
        (3, "c", "Zero"),
    ]
    assert rows[0]["mean"] == 4 and rows[1]["mean"] == 2
    assert m.encode_json(rows) == m.encode_json(
        m.leader_rows(frame.iloc[::-1], {"a": "Same", "b": "Same", "c": "Zero"})
    )


def test_audit_hash_and_snapshot_identity(tmp_path):
    m = module()
    p = tmp_path / "report.json"
    p.write_text(
        json.dumps(
            {
                "snapshot_id": "sha256:abc",
                "layers": {"snapshot": {"verdict": "FAIL"}, "legacy_csv": {"verdict": "FAIL"}},
            }
        )
    )
    digest = m.sha256_file(p)
    assert m.read_audit(p, digest, "sha256:abc")["layers"]["snapshot"]["verdict"] == "FAIL"
    with pytest.raises(ValueError, match="hash"):
        m.read_audit(p, "bad", "sha256:abc")
    with pytest.raises(ValueError, match="snapshot"):
        m.read_audit(p, digest, "sha256:def")


def test_output_containment(tmp_path):
    m = module()
    root = tmp_path / "provisional"
    root.mkdir()
    assert m.output_path(root, "career/goals.csv") == root / "career/goals.csv"
    with pytest.raises(ValueError):
        m.output_path(root, "../input.csv")
    (root / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError):
        m.output_path(root, "escape/input.csv")


def test_bundle_uses_observed_cells_and_separate_counter():
    import duckdb

    from supercoach_via.analytics.players import player_stats_bundle
    from supercoach_via.domain.metrics import CoverageEras

    class Query:
        def __init__(self):
            self.con = duckdb.connect(":memory:")
            self.con.execute("""CREATE TABLE player_games AS SELECT * FROM (VALUES
                ('a', 2000, 'club', DATE '2000-01-01', 10, 4),
                ('a', 2000, 'club', DATE '2000-01-02', 11, NULL),
                ('b', 2000, 'club', DATE '2000-01-01', 1, 0),
                ('c', 2000, 'club', DATE '2000-01-01', 1, NULL)
                ) v(player_id, season, club_id, match_date, career_game_counter, goals)""")

        def df(self, sql):
            return self.con.execute(sql).df()

    q = Query()
    try:
        bundle = player_stats_bundle(q, CoverageEras({}), ["goals"])
        a = bundle.careers.set_index("player_id").loc["a"]
        assert a.total == 4 and a.observed_games == 1 and a["mean"] == 4
        assert a.career_games == 11
        assert bundle.games.set_index("player_id").loc["a"].row_games == 2
        rows = module().leader_rows(bundle.careers, {})
        assert [(r["player_id"], r["total"]) for r in rows] == [("a", 4), ("b", 0)]
    finally:
        q.con.close()


def test_markdown_tags_all_numeric_cells_and_provenance():
    m = module()
    md = m.render_table("Goals", [{"rank": 1, "player_id": "abc", "total": 0, "mean": None}], "snapshot `sha256:abc`")
    assert "# PROVISIONAL" in md and "snapshot `sha256:abc`" in md
    assert "1 **[data]**" in md and "0 **[data]**" in md
    assert "not recorded" in md


def test_human_columns_keep_identity_and_explicit_denominators():
    md = module().render_table(
        "Goals", [dict(rank=1, player_id="a", total=4, observed_games=1, career_games=11, mean=4)], "PROVISIONAL"
    )
    assert "Observed games" in md and "Mean over observed games" in md
    assert "Player ID" in md and "Career games (max rows/counter)" in md
