"""Dataset validation: schema/keys/FK, stage identity, arithmetic, coverage, exceptions."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from supercoach_via.domain.schemas import CheckOutcome, Severity
from supercoach_via.ingest import legacy, reconcile
from supercoach_via.settings import RunContext, Settings

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "legacy" / "corpus"
REPO_CONFIG = Path(__file__).resolve().parents[3] / "config"


def _import(tmp: Path, mutate: dict[str, tuple[str, str]] | None = None) -> legacy.DatasetCandidate:
    src = tmp / "src"
    shutil.copytree(FIXTURE, src)
    for rel, (old, new) in (mutate or {}).items():
        p = src / rel
        text = p.read_text(encoding="utf-8")
        assert old in text, (rel, old)
        p.write_text(text.replace(old, new), encoding="utf-8")
    ctx = RunContext(settings=Settings(data_root=tmp / "var"), clock=lambda: datetime(2026, 9, 24, tzinfo=UTC))
    return legacy.import_legacy(src, ctx)


@pytest.fixture(scope="module")
def cand(tmp_path_factory: pytest.TempPathFactory) -> legacy.DatasetCandidate:
    return _import(tmp_path_factory.mktemp("rec"))


@pytest.fixture(scope="module")
def default_report(cand: legacy.DatasetCandidate) -> reconcile.ValidationReport:
    return reconcile.validate_dataset(cand, reconcile.load_policy(REPO_CONFIG))


def _rules(report: reconcile.ValidationReport, severity: str | None = None) -> set[str]:
    return {i["rule_id"] for i in report.issues if severity is None or i["severity"] == severity}


class TestPolicy:
    def test_load_repo_policy(self) -> None:
        pol = reconcile.load_policy(REPO_CONFIG)
        assert pol.coverage.recorded_from["goals"] == 1897
        assert pol.coverage.recorded_from["hitouts"] == 1966
        assert pol.coverage.recorded_from["goal_assists"] == 2003
        assert pol.known_exceptions == ()

    def test_coverage_matches_legacy_yaml(self) -> None:
        pol = reconcile.load_policy(REPO_CONFIG)
        legacy_map = reconcile.load_legacy_coverage(REPO_CONFIG / "stat_coverage_eras.yaml")
        assert {pol.coverage.legacy_names[k]: v for k, v in pol.coverage.recorded_from.items()} == legacy_map

    def test_bad_exception_rejected(self, tmp_path: Path) -> None:
        cfg = tmp_path / "coverage.yaml"
        cfg.write_text(
            (REPO_CONFIG / "coverage.yaml")
            .read_text()
            .replace("known_exceptions: []", "known_exceptions:\n  - {rule_id: x}")
        )
        with pytest.raises(ValueError):
            reconcile.load_policy(tmp_path, coverage_path=cfg)


class TestFixtureValidation:
    def test_current_season_defect_blocks(self, default_report: reconcile.ValidationReport) -> None:
        # 'Challenge Final' in 2026 is unrecognized: a current-season defect is blocking
        rep = default_report
        assert rep.outcome is CheckOutcome.FAIL
        blocking = [i for i in rep.issues if i["severity"] == Severity.BLOCKING.value]
        assert any(i["rule_id"] == "stage_unrecognized" and i["season"] == 2026 for i in blocking)
        assert rep.checks["schema"] is CheckOutcome.PASS
        assert rep.checks["keys"] is CheckOutcome.PASS
        assert rep.checks["foreign_keys"] is CheckOutcome.PASS

    def test_exception_cannot_suppress_current_season(
        self, cand: legacy.DatasetCandidate, default_report: reconcile.ValidationReport
    ) -> None:
        base = default_report
        target = next(i for i in base.issues if i["rule_id"] == "stage_unrecognized")
        pol = reconcile.load_policy(REPO_CONFIG)
        pol = pol.with_exceptions([reconcile.KnownException(target["rule_id"], target["row_key"], "test")])
        rep = reconcile.validate_dataset(cand, pol)
        assert rep.outcome is CheckOutcome.FAIL
        assert "exception_rejected_current_season" in _rules(rep, "blocking")

    def test_historical_exception_accepted(self, tmp_path: Path) -> None:
        # the fixture's squads are deliberately partial, so evaluate with no data in the
        # current season (2027); then accept a historical (2025) issue explicitly
        c = _import(tmp_path, {"data/matches/matches_2026.csv": ("Challenge Final", "Grand Final")})
        pol = reconcile.load_policy(REPO_CONFIG, current_season=2027)
        rep = reconcile.validate_dataset(c, pol)
        blocking = [i for i in rep.issues if i["severity"] == "blocking"]
        assert blocking == [], blocking
        assert rep.outcome is CheckOutcome.PASS
        hist = next(i for i in rep.issues if i["season"] == 2025 and i["severity"] in ("warning", "error"))
        rep2 = reconcile.validate_dataset(
            c, pol.with_exceptions([reconcile.KnownException(hist["rule_id"], hist["row_key"], "documented")])
        )
        accepted = [i for i in rep2.issues if i["issue_id"] == hist["issue_id"]]
        assert accepted and accepted[0]["status"] == "accepted" and accepted[0]["acceptance_basis"] == "documented"

    def test_arithmetic_checks(self, tmp_path: Path) -> None:
        c = _import(
            tmp_path,
            {
                "data/matches/matches_2026.csv": ("Challenge Final", "Grand Final"),
                # 2025 R1 Sydney v Hawthorn quarter goals decrease (5 -> 3)
                "data/matches/matches_2025.csv": ("Sydney,1,1,3,3,5,5,7,7", "Sydney,1,1,3,3,5,5,3,7"),
                # disposals != kicks + handballs for a 2026 row
                "data/player_data/lynch_tom_31101990_performance_details.csv": (",10,,5,15,1,", ",10,,5,16,1,"),
            },
        )
        rep = reconcile.validate_dataset(c, reconcile.load_policy(REPO_CONFIG, current_season=2026))
        by_rule = {i["rule_id"]: i for i in rep.issues}
        assert by_rule["quarter_scores_decrease"]["severity"] == "warning"  # historical
        assert by_rule["disposals_arithmetic"]["severity"] == "blocking"  # current season
        assert rep.outcome is CheckOutcome.FAIL

    def test_goal_reconciliation_and_counters(self, default_report: reconcile.ValidationReport) -> None:
        rep = default_report
        assert "match_player_goals_mismatch" in _rules(rep)
        assert rep.counts["players_counter_exceeds_rows"] >= 0
        assert rep.counts["rows:player_games"] > 0

    def test_coverage_reported(self, default_report: reconcile.ValidationReport) -> None:
        rep = default_report
        assert rep.counts["coverage:goals:1965-1986:rows"] == 5  # the 1977 rows
        assert rep.counts["coverage:goals:pre-1965:rows"] == 0
        assert rep.counts["coverage:goals:2003-present:rows"] > 0
        assert rep.counts["coverage:tackles:2003-present:observed"] == 0

    def test_import_alone_is_never_verified(self, cand: legacy.DatasetCandidate) -> None:
        assert cand.candidate.manifest.status.value == "legacy_unverified"


class TestDedup:
    @given(
        st.lists(
            st.tuples(st.sampled_from("abc"), st.sampled_from("xy"), st.booleans(), st.integers(1, 50)), max_size=30
        )
    )
    @settings(max_examples=60, deadline=None)
    def test_dedup_idempotent(self, rows: list[tuple[str, str, bool, int]]) -> None:
        once = reconcile.dedupe_keys(rows)
        assert reconcile.dedupe_keys(once) == once
        keys = [(r[0], r[1]) for r in once]
        assert len(keys) == len(set(keys))
        assert {(r[0], r[1]) for r in rows} == set(keys)
