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


def coverage_fixture(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS

    m = module()

    def table(digest, partition):
        return NS(fragments=(NS(sha256=digest, partition=partition),))

    parent = NS(
        snapshot_id="sha256:parent",
        parent=None,
        tables={
            "player_games": NS(
                fragments=(NS(sha256="a" * 64, partition="1932"), NS(sha256="b" * 64, partition="2026"))
            ),
            "quality_issues": table("c" * 64, None),
        },
    )
    child = NS(
        snapshot_id="sha256:child",
        parent=parent.snapshot_id,
        tables={
            "player_games": NS(
                fragments=(NS(sha256="a" * 64, partition="1932"), NS(sha256="d" * 64, partition="2026"))
            ),
            "quality_issues": table("e" * 64, None),
        },
    )
    loads = []

    def load(root, selector, *, verify):
        loads.append((selector, verify))
        assert selector == parent.snapshot_id
        return parent

    monkeypatch.setattr(m, "load_snapshot", load)
    receipt = {
        "schema": "scvia.source-coverage-composition/1",
        "kind": "coverage_argument_from_two_existing_audits",
        "snapshot_id": child.snapshot_id,
        "parent_snapshot_id": parent.snapshot_id,
        "source_verdict": "UNKNOWN",
        "confirmed_discrepancies": 0,
        "unresolved_cells_per_layer": {"legacy_csv": 1, "snapshot": 1},
        "unresolved_ledger": [
            {"id": "legacy-1", "layer": "legacy_csv", "season": 1932},
            {"id": "snapshot-1", "layer": "snapshot", "season": 1932},
        ],
        "unchanged_partitions": [{"table": "player_games", "partition": "1932", "sha256": ["a" * 64]}],
        "storage_verification": {
            "hash_mismatches": 0,
            "player_game_keys_unchanged": True,
            "changed_partitions": [
                {
                    "table": "player_games",
                    "partition": "2026",
                    "parent_sha256": ["b" * 64],
                    "candidate_sha256": ["d" * 64],
                },
                {
                    "table": "quality_issues",
                    "partition": None,
                    "parent_sha256": ["c" * 64],
                    "candidate_sha256": ["e" * 64],
                },
            ],
        },
        "evidence": [
            {
                "snapshot_id": parent.snapshot_id,
                "report_sha256": "1" * 64,
                "overall": "UNKNOWN",
                "layers": {"snapshot": "UNKNOWN", "legacy_csv": "UNKNOWN"},
                "scope": {
                    "population": "all",
                    "full_population": True,
                    "acquisition_window_utc": ["2026-10-01T11:46:00Z", "2026-10-02T04:44:55Z"],
                },
            },
            {
                "snapshot_id": child.snapshot_id,
                "report_sha256": "2" * 64,
                "overall": "PASS",
                "layers": {"snapshot": "PASS", "legacy_csv": "PASS"},
                "scope": {
                    "population": "seasons",
                    "seasons": [2026],
                    "acquisition_window_utc": ["2026-10-05T20:05:26Z", "2026-10-05T20:35:27Z"],
                },
            },
        ],
    }
    return m, child, receipt, loads


def test_composite_coverage_is_unknown_and_preserves_original_report_scopes(tmp_path, monkeypatch):
    m, child, receipt, loads = coverage_fixture(tmp_path, monkeypatch)
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(receipt))
    checked = m.read_coverage(path, m.sha256_file(path), child, tmp_path)
    assert checked == receipt
    assert loads == [("sha256:parent", True)]
    banner = m.coverage_banner(checked)
    for text in (
        "UNKNOWN",
        "1 unresolved source cells per layer",
        "sha256:parent",
        "sha256:child",
        "1" * 64,
        "2" * 64,
        "2026 only",
        "2026-10-01T11:46:00Z",
        "2026-10-05T20:05:26Z",
        "not a new full audit",
    ):
        assert text in banner
    assert "Known audit failures include" not in banner


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "kind",
        "snapshot",
        "parent",
        "verdict",
        "partition",
        "omit",
        "unresolved",
        "count",
        "scope",
        "report_id",
        "hash",
    ],
)
def test_composite_receipt_rejects_misbound_or_unsupported_claims(tmp_path, monkeypatch, mutation):
    m, child, receipt, _loads = coverage_fixture(tmp_path, monkeypatch)
    if mutation in ("schema", "kind"):
        receipt[mutation] = "unsupported"
    elif mutation == "snapshot":
        receipt["snapshot_id"] = "sha256:other"
    elif mutation == "parent":
        receipt["parent_snapshot_id"] = "sha256:other"
    elif mutation == "verdict":
        receipt["source_verdict"] = "PASS"
    elif mutation == "partition":
        receipt["unchanged_partitions"][0]["sha256"] = ["0" * 64]
    elif mutation == "omit":
        receipt["storage_verification"]["changed_partitions"].pop()
    elif mutation == "unresolved":
        receipt["unresolved_ledger"][0]["season"] = 2026
    elif mutation == "count":
        receipt["unresolved_cells_per_layer"]["snapshot"] = 0
    elif mutation == "scope":
        receipt["evidence"][1]["scope"]["seasons"] = [2025]
    elif mutation == "report_id":
        receipt["evidence"][0]["snapshot_id"] = child.snapshot_id
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(receipt))
    digest = "bad" if mutation == "hash" else m.sha256_file(path)
    with pytest.raises(ValueError):
        m.read_coverage(path, digest, child, tmp_path)


@pytest.mark.parametrize("composite", [False, True])
def test_generation_records_the_selected_evidence_without_upgrading_source_confidence(tmp_path, monkeypatch, composite):
    from contextlib import nullcontext
    from types import SimpleNamespace as NS

    m, child, receipt, _loads = coverage_fixture(tmp_path, monkeypatch)
    parent_loader = m.load_snapshot
    monkeypatch.setattr(
        m,
        "load_snapshot",
        lambda root, selector, verify: (
            child if selector == child.snapshot_id else parent_loader(root, selector, verify=verify)
        ),
    )
    monkeypatch.setattr(m, "REPO", tmp_path)
    monkeypatch.setattr(m, "STATS", ("goals",))
    monkeypatch.setattr(m, "SnapshotQuery", lambda *args: nullcontext(NS(rows=lambda sql: [("a", "Example Player")])))
    bundle = NS(
        games=pd.DataFrame([dict(player_id="a", row_games=1, counter_max=1)]),
        careers=pd.DataFrame([dict(player_id="a", stat="goals", total=2, observed_games=1, mean=2)]),
        seasons=pd.DataFrame([dict(player_id="a", season=2026, stat="goals", total=2, observed_games=1, mean=2)]),
    )
    monkeypatch.setattr(m, "player_stats_bundle", lambda *args: bundle)
    monkeypatch.setattr(m, "run_legacy_v1", lambda *args: NS(all_time=[]))
    data_root = tmp_path / "data"
    (data_root / "snapshots").mkdir(parents=True)
    (data_root / "snapshots/child.json").write_text("{}")
    evidence = (
        receipt
        if composite
        else {
            "snapshot_id": child.snapshot_id,
            "layers": {"snapshot": {"verdict": "FAIL"}, "legacy_csv": {"verdict": "FAIL"}},
        }
    )
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence))
    result = m.generate(data_root, child.snapshot_id, path, m.sha256_file(path), "2026-10-07", composite=composite)
    provenance = json.loads(result["provenance.json"])
    assert provenance["snapshot_id"] == child.snapshot_id
    assert provenance["audit_verdicts"]["snapshot"] == ("UNKNOWN" if composite else "FAIL")
    assert "2 **[data]**" in result["career/goals.md"]
    if composite:
        assert provenance["source_coverage"] == receipt
        assert "--coverage-receipt" in result["README.md"]
        assert "not a new full audit" in result["career/goals.md"]
        assert "Known audit failures include" not in result["career/goals.md"]
        assert "Historical Brownlow totals are incomplete, especially before 1984" not in result["README.md"]
    else:
        assert "source_coverage" not in provenance
        assert "--audit-report" in result["README.md"]
        assert "Known audit failures include" in result["career/goals.md"]
    import hashlib

    for rel, digest in json.loads(result["checksums.json"]).items():
        assert hashlib.sha256(result[rel].encode()).hexdigest() == digest
