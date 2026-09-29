"""Integrity checker, relationships, aggregates and football rules (families B and C)."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from supercoach_via.integrity.runner import AuditOptions, AuditResult, run_audit
from supercoach_via.settings import default_config_dir
from tests.scvia.unit import integrity_fixtures as fx

FAMILIES = ("dataset", "relations", "aggregates", "football")
Rows = dict[str, list[dict[str, Any]]]


def audit(root: Path, **kw: Any) -> AuditResult:
    opts = dict(data_root=root, snapshot="current", scope="data", as_of=fx.AS_OF, families=FAMILIES, keep_all=True)
    opts.update(kw)
    return run_audit(AuditOptions(**opts))  # type: ignore[arg-type]


def found(res: AuditResult, rule_id: str) -> list[Any]:
    return [f for f in res.findings if f.rule_id == rule_id]


def entities(res: AuditResult, rule_id: str) -> set[str]:
    return {f.entity for f in found(res, rule_id) if f.status == "open"}


def g(rows: Rows, match: str, player: str) -> dict[str, Any]:
    return next(r for r in rows["player_games"] if r["match_id"] == match and r["player_id"] == player)


M26 = "m:2026:r01:alpha:beta:0"
M70 = "m:1970:r01:alpha:beta:0"


def test_clean_corpus_has_no_findings(tmp_path: Path) -> None:
    root = tmp_path / "var"
    fx.build(root)
    res = audit(root)
    assert res.findings == [], [f.as_dict() for f in res.findings]
    assert res.outcome.value == "PASS"


def test_validate_dataset_verdict_is_reused_with_severity(tmp_path: Path) -> None:
    root = tmp_path / "var"
    fx.rehash(root, lambda r: next(m for m in r["matches"] if m["match_id"] == M26).update(home_score=99))
    res = audit(root, checks=('dataset.validate',))
    assert M26 in entities(res, "dataset.score_arithmetic")
    assert found(res, "dataset.score_arithmetic")[0].severity.value == "blocking"
    assert res.outcome.value == "FAIL"


def test_orphan_reference(tmp_path: Path) -> None:
    root = tmp_path / "var"
    fx.rehash(root, lambda r: g(r, M26, "legacy:p1").update(player_id="legacy:ghost"))
    res = audit(root, checks=('dataset.validate',))
    assert found(res, "dataset.fk_orphan")
    assert res.outcome.value == "FAIL"


@pytest.mark.parametrize(
    ("mutate", "rule_id", "entity"),
    [
        (
            lambda r: g(r, M26, "legacy:p1").update(season=2025),
            "relations.season_mismatch",
            f"player_game:{M26}|legacy:p1|alpha",
        ),
        (
            lambda r: g(r, M26, "legacy:p1").update(match_date=date(2026, 3, 6)),
            "relations.verified_date_mismatch",
            f"player_game:{M26}|legacy:p1|alpha",
        ),
        (
            lambda r: g(r, M26, "legacy:p1").update(stage_id="r09"),
            "relations.stage_mismatch",
            f"player_game:{M26}|legacy:p1|alpha",
        ),
        (
            lambda r: g(r, M26, "legacy:p1").update(result="L"),
            "relations.result_mismatch",
            f"player_game:{M26}|legacy:p1|alpha",
        ),
        (
            lambda r: r["player_games"].append(
                {**g(r, M26, "legacy:p1"), "club_id": "beta", "opponent_club_id": "alpha"}
            ),
            "relations.player_in_both_sides",
            f"player_game:{M26}|legacy:p1",
        ),
    ],
)
def test_wrong_relationships(tmp_path: Path, mutate: Any, rule_id: str, entity: str) -> None:
    root = tmp_path / "var"
    fx.rehash(root, mutate)
    res = audit(root, checks=('relations.membership',))
    assert entity in entities(res, rule_id), [f.as_dict() for f in res.findings][:5]
    assert res.outcome.value == "FAIL"


def test_stage_label_vocabularies_may_differ_when_stage_id_agrees(tmp_path: Path) -> None:
    """Player rows keep the player-file token ("SF"); matches keep the fixture label ("Semi Final")."""
    root = tmp_path / "var"
    fx.rehash(root, lambda r: g(r, M26, "legacy:p1").update(stage_label="R1"))
    res = audit(root, checks=('relations.membership',))
    assert not found(res, "relations.stage_mismatch")


def test_wrong_club_uses_the_existing_rule(tmp_path: Path) -> None:
    root = tmp_path / "var"
    fx.rehash(root, lambda r: g(r, M26, "legacy:p1").update(opponent_club_id="alpha"))
    res = audit(root, checks=('dataset.validate',))
    assert found(res, "dataset.player_game_club_not_in_match")
    assert res.outcome.value == "FAIL"  # current season: escalated by the producer's own rule


def test_deleted_match_leaves_stale_season_aggregate(tmp_path: Path) -> None:
    def mutate(r: Rows) -> None:
        r["matches"] = [m for m in r["matches"] if m["match_id"] != "m:2026:r02:alpha:beta:0"]
        r["player_games"] = [x for x in r["player_games"] if x["match_id"] != "m:2026:r02:alpha:beta:0"]

    root = tmp_path / "var"
    fx.rehash(root, mutate)
    res = audit(root, checks=('aggregates.seasons',))
    fields = {(f.entity, f.field, f.expected, f.actual) for f in found(res, "aggregates.season_value")}
    assert ("season:2026", "matches_complete", 1, 2) in fields
    assert ("season:2026", "last_match_date", "2026-03-05", "2026-03-12") in fields
    assert res.outcome.value == "FAIL"


def test_moved_match_date_leaves_stale_season_aggregate(tmp_path: Path) -> None:
    def mutate(r: Rows) -> None:
        m = next(m for m in r["matches"] if m["match_id"] == M70)
        m.update(match_date=date(1970, 3, 28), local_start="1970-03-28 14:10")
        for x in r["player_games"]:
            if x["match_id"] == M70:
                x["match_date"] = date(1970, 3, 28)

    root = tmp_path / "var"
    fx.rehash(root, mutate)
    res = audit(root, checks=('aggregates.seasons',))
    assert ("season:1970", "first_match_date") in {(f.entity, f.field) for f in found(res, "aggregates.season_value")}


def test_season_row_missing(tmp_path: Path) -> None:
    root = tmp_path / "var"
    fx.rehash(root, lambda r: r.update(seasons=[s for s in r["seasons"] if s["season"] != 1970]))
    res = audit(root, checks=('aggregates.seasons',))
    assert "season:1970" in entities(res, "aggregates.season_row")


def test_alias_chain_and_cycle(tmp_path: Path) -> None:
    def mutate(r: Rows) -> None:
        extra = {**r["players"][0], "legacy_slug": None}
        r["players"].append(
            {**extra, "player_id": "legacy:a1", "identity_status": "alias", "canonical_player_id": "legacy:a2"}
        )
        r["players"].append(
            {**extra, "player_id": "legacy:a2", "identity_status": "alias", "canonical_player_id": "legacy:a1"}
        )
        r["players"].append(
            {**extra, "player_id": "legacy:a3", "identity_status": "alias", "canonical_player_id": "legacy:p1"}
        )

    root = tmp_path / "var"
    fx.rehash(root, mutate)
    res = audit(root, checks=('relations.identity',))
    assert entities(res, "relations.alias_target") == {"player:legacy:a1", "player:legacy:a2"}


def test_player_behinds_cannot_exceed_team_behinds(tmp_path: Path) -> None:
    def mutate(r: Rows) -> None:
        for x in r["player_games"]:
            if x["match_id"] == M26 and x["club_id"] == "beta":
                x["behinds"] = 3  # 6 > team behinds 4

    root = tmp_path / "var"
    fx.rehash(root, mutate)
    res = audit(root, checks=('football.arithmetic',))
    assert f"match:{M26}|beta" in entities(res, "football.player_behinds_exceed_team")
    assert found(res, "football.player_behinds_exceed_team")[0].severity.value == "blocking"


def test_brownlow_range(tmp_path: Path) -> None:
    root = tmp_path / "var"
    fx.rehash(root, lambda r: g(r, M26, "legacy:p1").update(brownlow_votes=5))
    res = audit(root, checks=('football.arithmetic',))
    assert f"player_game:{M26}|legacy:p1|alpha" in entities(res, "football.brownlow_range")


def test_unusual_but_valid_value_is_a_warning_only(tmp_path: Path) -> None:
    root = tmp_path / "var"
    fx.rehash(root, lambda r: g(r, M26, "legacy:p1").update(kicks=40, handballs=30, disposals=70))
    res = audit(root, checks=('football.values',))
    [w] = found(res, "football.unusual_value")
    assert (w.field, w.actual, w.expected, w.severity.value) == ("disposals", 70, 60, "warning")
    assert res.outcome.value == "PASS"


def test_null_converted_to_zero_before_recorded_era(tmp_path: Path) -> None:
    def mutate(r: Rows) -> None:
        for x in r["player_games"]:
            if x["season"] == 1970:
                x["tackles"] = 0

    root = tmp_path / "var"
    fx.rehash(root, mutate)
    res = audit(root, checks=('football.coverage',))
    assert "stat:tackles:1970" in entities(res, "football.zero_before_recorded")
    assert found(res, "football.zero_before_recorded")[0].severity.value == "error"


def test_old_season_nulls_are_not_held_to_modern_coverage(tmp_path: Path) -> None:
    """1970 has no tackles, %P or disposals at all; that is the era, not a defect (control)."""
    root = tmp_path / "var"
    fx.build(root)
    res = audit(root)
    assert not [f for f in res.findings if f.season == 1970]


def test_blank_as_null_in_recorded_era_is_explained(tmp_path: Path) -> None:
    """A stat that is never zero but often null inside its era looks like blanks read as missing."""

    def mutate(r: Rows) -> None:
        for x in r["player_games"]:
            if x["season"] == 2026 and x["player_id"] in ("legacy:p1", "legacy:p3"):
                x["hitouts"] = None

    root = tmp_path / "var"
    fx.rehash(root, mutate)
    res = audit(root, checks=('football.coverage',))
    [f] = found(res, "football.blank_as_null")
    assert (f.entity, f.severity.value) == ("stat:hitouts", "warning")
    assert res.outcome.value == "PASS"


def _config_with(tmp_path: Path, exceptions: str) -> Path:
    cfg = tmp_path / "cfg"
    shutil.copytree(default_config_dir(), cfg)
    p = cfg / "integrity_policy.yaml"
    p.write_text(p.read_text().replace("known_exceptions: []", "known_exceptions:\n" + exceptions))
    return cfg


def test_legitimate_historical_exception_is_accepted_and_visible(tmp_path: Path) -> None:
    def mutate(r: Rows) -> None:
        for x in r["player_games"]:
            if x["season"] == 1970:
                x["tackles"] = 0

    root = tmp_path / "var"
    fx.rehash(root, mutate)
    cfg = _config_with(
        tmp_path,
        '  - {rule_id: football.zero_before_recorded, entity: "stat:tackles:1970", '
        'reason: "documented source quirk"}\n'
        '  - {rule_id: football.zero_before_recorded, entity: "stat:marks:1970", '
        'reason: "no longer occurs"}\n',
    )
    res = audit(root, config_dir=cfg, checks=('football.coverage',))
    [f] = found(res, "football.zero_before_recorded")
    assert (f.status, f.acceptance) == ("accepted", "documented source quirk")
    assert [e["entity"] for e in res.report["exceptions"]["stale"]] == ["stat:marks:1970"]
    assert res.outcome.value == "PASS"  # accepted historical error, stale exception is a warning


def test_current_season_suppression_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "var"
    fx.rehash(root, lambda r: g(r, M26, "legacy:p1").update(brownlow_votes=5))
    ent = f"player_game:{M26}|legacy:p1|alpha"
    cfg = _config_with(tmp_path, f'  - {{rule_id: football.brownlow_range, entity: "{ent}", reason: "hide it"}}\n')
    res = audit(root, config_dir=cfg, checks=('football.arithmetic',))
    assert ent in entities(res, "football.brownlow_range")
    assert ent in entities(res, "policy.exception_rejected_current_season")
    assert res.outcome.value == "FAIL"
    assert res.report["policy"]["sha256"] != audit(root, checks=('football.arithmetic',)).report["policy"]["sha256"]
