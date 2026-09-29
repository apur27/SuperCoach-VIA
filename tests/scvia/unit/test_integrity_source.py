"""Integrity checker, source -> snapshot and freshness (families D and E)."""

from __future__ import annotations

import copy
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from supercoach_via.integrity.runner import AuditOptions, AuditResult, run_audit
from tests.scvia.unit import integrity_fixtures as fx

FAMILIES = ("source", "freshness")
M26 = "m:2026:r01:alpha:beta:0"
PG1 = f"player_game:{M26}|legacy:p1|alpha"
Rows = dict[str, list[dict[str, Any]]]


def audit(root: Path, **kw: Any) -> AuditResult:
    opts = dict(data_root=root, snapshot="current", scope="data", as_of=fx.AS_OF, families=FAMILIES, keep_all=True)
    opts.update(kw)
    return run_audit(AuditOptions(**opts))  # type: ignore[arg-type]


def found(res: AuditResult, rule_id: str) -> list[Any]:
    return [f for f in res.findings if f.rule_id == rule_id and f.status == "open"]


def keyed(res: AuditResult, rule_id: str) -> set[tuple[str, str | None]]:
    return {(f.entity, f.field) for f in found(res, rule_id)}


def pg(rows: Rows, player: str, match: str = M26) -> dict[str, Any]:
    return next(r for r in rows["player_games"] if r["match_id"] == match and r["player_id"] == player)


def test_clean_sourced_corpus_passes(tmp_path: Path) -> None:
    fx.with_sources(tmp_path)
    res = audit(tmp_path)
    assert res.findings == [], [f.as_dict() for f in res.findings]
    assert res.outcome.value == "PASS"
    cov = res.report["coverage"]["source_capture"]
    assert cov["player_games_compared"] == 4
    assert cov["cells_compared"] == 4 * 23


def test_swapped_player_identities_with_unchanged_totals(tmp_path: Path) -> None:
    def source_values(r: Rows) -> None:
        pg(r, "legacy:p1").update(kicks=5, disposals=6)
        pg(r, "legacy:p2").update(kicks=1, disposals=2)

    # the source says p1 kicked 5 and p2 kicked 1; the snapshot has them the other way round
    src = copy.deepcopy(fx.tables())
    source_values(src)
    fx.with_sources(tmp_path, page_rows=src, mutate=lambda r: (source_values(r), _swap_players(r)))
    res = audit(tmp_path)
    got = keyed(res, "source.stat_cell_mismatch")
    assert (PG1, "kicks") in got and (f"player_game:{M26}|legacy:p2|alpha", "kicks") in got
    assert res.outcome.value == "FAIL"


def _swap_players(r: Rows) -> None:
    a, b = pg(r, "legacy:p1"), pg(r, "legacy:p2")
    for s in ("kicks", "disposals"):
        a[s], b[s] = b[s], a[s]


def test_swapped_stat_columns_with_unchanged_totals(tmp_path: Path) -> None:
    def mutate(r: Rows) -> None:
        g = pg(r, "legacy:p1")
        g["kicks"], g["handballs"] = g["handballs"], g["kicks"]

    src = copy.deepcopy(fx.tables())
    pg(src, "legacy:p1").update(kicks=4, handballs=1, disposals=5)
    fx.with_sources(
        tmp_path,
        page_rows=src,
        mutate=lambda r: (pg(r, "legacy:p1").update(kicks=4, handballs=1, disposals=5), mutate(r)),
    )
    res = audit(tmp_path)
    assert {(PG1, "kicks"), (PG1, "handballs")} <= keyed(res, "source.stat_cell_mismatch")


def test_changed_number_with_all_hashes_recalculated(tmp_path: Path) -> None:
    """Every fragment hash and the snapshot id are valid; only the source comparison can see it."""
    fx.with_sources(tmp_path, mutate=lambda r: pg(r, "legacy:p1").update(kicks=2, disposals=3))
    res = audit(tmp_path)
    [f] = [f for f in found(res, "source.stat_cell_mismatch") if f.field == "kicks"]
    assert (f.entity, f.expected, f.actual) == (PG1, 1, 2)
    assert f.evidence["source_sha256"]
    assert res.outcome.value == "FAIL"


def test_null_converted_to_zero_is_a_source_contradiction(tmp_path: Path) -> None:
    src = copy.deepcopy(fx.tables())
    pg(src, "legacy:p1")["bounces"] = None
    fx.with_sources(tmp_path, page_rows=src, mutate=lambda r: pg(r, "legacy:p1").update(bounces=0))
    res = audit(tmp_path)
    [f] = [f for f in found(res, "source.stat_cell_mismatch") if f.field == "bounces"]
    assert (f.expected, f.actual) == (None, 0)


def test_player_missing_and_extra(tmp_path: Path) -> None:
    fx.with_sources(
        tmp_path,
        mutate=lambda r: r.update(
            player_games=[g for g in r["player_games"] if not (g["match_id"] == M26 and g["player_id"] == "legacy:p4")]
        ),
    )
    res = audit(tmp_path)
    assert any("legacy:p4" in f.entity or "Player 4" in f.entity for f in found(res, "source.player_missing"))


def test_wrong_date_and_club_on_the_match_row(tmp_path: Path) -> None:
    def mutate(r: Rows) -> None:
        m = next(m for m in r["matches"] if m["match_id"] == M26)
        m.update(venue_source_name="Elsewhere", attendance=999)

    fx.with_sources(tmp_path, mutate=mutate)
    res = audit(tmp_path)
    assert {(f"match:{M26}", "venue_source_name"), (f"match:{M26}", "attendance")} <= keyed(
        res, "source.match_value_mismatch"
    )


def test_permuted_source_columns_still_pass(tmp_path: Path) -> None:
    """The comparison maps by column label, so a reordered source table is not a difference."""
    fx.with_sources(tmp_path, columns=list(reversed(fx.LABELS)))
    res = audit(tmp_path)
    assert res.outcome.value == "PASS", [f.as_dict() for f in res.findings]


def test_absent_source_evidence_is_unknown_not_pass(tmp_path: Path) -> None:
    shas = fx.with_sources(tmp_path)
    (tmp_path / "raw" / "objects" / shas["match"][:2] / shas["match"]).unlink()
    res = audit(tmp_path)
    assert res.outcome.value == "UNKNOWN"
    assert found(res, "source.evidence_missing")
    check = next(c for c in res.report["checks"] if c["check_id"] == "source.match_pages")
    assert check["status"] == "UNKNOWN"


def test_evidence_dir_supplies_a_moved_payload(tmp_path: Path) -> None:
    import gzip

    shas = fx.with_sources(tmp_path)
    obj = tmp_path / "raw" / "objects" / shas["match"][:2] / shas["match"]
    ev = tmp_path / "evidence"
    ev.mkdir()
    (ev / f"{shas['match']}.html.gz").write_bytes(gzip.compress(obj.read_bytes()))
    obj.unlink()
    res = audit(tmp_path, evidence_dirs=(ev,))
    assert res.outcome.value == "PASS"


# -- freshness ------------------------------------------------------------------


def test_successful_unchanged_fixture_with_stale_metadata(tmp_path: Path) -> None:
    later = datetime(2026, 9, 28, tzinfo=UTC)

    def add_later(r: Rows) -> None:
        r["source_observations"].append({**r["source_observations"][0], "source_ref": "src:later", "fetched_at": later})

    fx.with_sources(tmp_path, mutate=add_later)
    res = audit(tmp_path)
    assert "season:2026" in {f.entity for f in found(res, "freshness.checked_at_behind_observation")}


def test_partial_fetch_marked_fresh(tmp_path: Path) -> None:
    def failed(r: Rows) -> None:
        r["source_observations"].append(
            {
                **r["source_observations"][0],
                "source_ref": "src:fail",
                "fetched_at": fx.CHECKED,
                "outcome": "FAIL",
                "content_sha256": None,
            }
        )
        r["source_observations"][0]["fetched_at"] = datetime(2026, 9, 26, tzinfo=UTC)

    fx.with_sources(tmp_path, mutate=failed)
    res = audit(tmp_path)
    assert "season:2026" in {f.entity for f in found(res, "freshness.checked_from_failed_observation")}


def test_request_timestamp_alone_does_not_prove_freshness(tmp_path: Path) -> None:
    fx.with_sources(tmp_path, mutate=lambda r: r["source_observations"].pop(0))
    res = audit(tmp_path)
    assert "season:2026" in {f.entity for f in found(res, "freshness.checked_without_observation")}


def test_evidence_after_as_of(tmp_path: Path) -> None:
    fx.with_sources(tmp_path)
    res = audit(tmp_path, as_of="2026-09-26T00:00:00Z")
    assert found(res, "freshness.after_as_of")
    assert res.outcome.value == "FAIL"


def test_stale_fixture_at_as_of_is_a_warning(tmp_path: Path) -> None:
    fx.with_sources(tmp_path)
    res = audit(tmp_path, as_of="2026-12-01T00:00:00Z")
    [f] = found(res, "freshness.fixture_stale")
    assert f.severity.value == "warning"
    assert res.outcome.value == "PASS"


def test_no_as_of_means_freshness_age_is_not_applicable(tmp_path: Path) -> None:
    fx.with_sources(tmp_path)
    res = audit(tmp_path, as_of=None)
    assert not found(res, "freshness.fixture_stale")
    check = next(c for c in res.report["checks"] if c["check_id"] == "freshness.as_of")
    assert check["status"] == "NOT_APPLICABLE"


def test_schedule_complete_needs_a_source_declaration(tmp_path: Path) -> None:
    fx.with_sources(tmp_path, schedule={"schedule_complete": True})
    res = audit(tmp_path)
    assert "season:2026" in {f.entity for f in found(res, "freshness.schedule_complete_unsupported")}


def test_missing_completed_fixture(tmp_path: Path) -> None:
    extra = [
        {
            **fx._match("m:x", 2026, "3", 3, date(2026, 3, 19), "alpha", "beta", (1, 1), (2, 2)),
            "game_id": "000120260319",
        }
    ]
    fx.with_sources(tmp_path, season_extra=extra)
    res = audit(tmp_path)
    [f] = found(res, "freshness.missing_result")
    assert f.entity == "fixture:2026-03-19|alpha|beta"
    assert res.outcome.value == "FAIL"


def test_future_scheduled_fixture_is_not_a_missing_result(tmp_path: Path) -> None:
    future = fx._match("m:y", 2026, "3", 3, date(2026, 3, 19), "alpha", "beta", (0, 0), (0, 0))
    for k in list(future):
        if k.endswith(("_goals", "_behinds", "_score")):
            future[k] = None
    fx.with_sources(tmp_path, season_extra=[future])
    res = audit(tmp_path)
    assert not found(res, "freshness.missing_result")
    [f] = found(res, "freshness.scheduled_fixture_absent")
    assert f.entity == "fixture:2026-03-19|alpha|beta"


def test_unexpected_match_not_in_the_source(tmp_path: Path) -> None:
    src = copy.deepcopy(fx.tables())
    src["matches"] = [m for m in src["matches"] if m["match_id"] != "m:2026:r02:alpha:beta:0"]
    fx.with_sources(tmp_path, page_rows=src)
    res = audit(tmp_path)
    assert "match:m:2026:r02:alpha:beta:0" in {f.entity for f in found(res, "freshness.unexpected_match")}


def test_fixture_score_mismatch(tmp_path: Path) -> None:
    fx.with_sources(
        tmp_path,
        mutate=lambda r: next(m for m in r["matches"] if m["match_id"] == "m:2026:r02:alpha:beta:0").update(
            home_q3_goals=0
        ),
    )
    res = audit(tmp_path)
    assert ("match:m:2026:r02:alpha:beta:0", "home_q3") in keyed(res, "freshness.fixture_value_mismatch")


def test_verified_status_over_legacy_rows(tmp_path: Path) -> None:
    fx.with_sources(tmp_path)
    fx.rewrite_manifest(tmp_path, lambda d: d.update(status="verified"))
    res = audit(tmp_path)
    assert found(res, "freshness.verified_claim")
