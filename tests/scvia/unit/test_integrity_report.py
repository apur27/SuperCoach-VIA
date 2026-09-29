"""Integrity report contract: findings, sampling, exceptions, canonical bytes and exit codes."""

from __future__ import annotations

import json
import random

import pytest

from supercoach_via.domain.schemas import Severity
from supercoach_via.integrity import report as R

RULES = {
    "t.contradiction": R.RuleSpec("t.contradiction", "t.check", Severity.BLOCKING, "contradiction", "repair it"),
    "t.anomaly": R.RuleSpec("t.anomaly", "t.check", Severity.WARNING, "anomaly", "review it", current_blocks=True),
    "t.info": R.RuleSpec("t.info", "t.other", Severity.INFO, "note", "none"),
}


def _collector(**kw: object) -> R.Collector:
    return R.Collector(RULES, sample_limit=3, current_season=2026, **kw)  # type: ignore[arg-type]


def test_exact_totals_with_capped_deterministic_samples() -> None:
    keys = [f"match:{i:03d}" for i in range(50)]
    orders = []
    for seed in (1, 2, 3):
        shuffled = keys[:]
        random.Random(seed).shuffle(shuffled)
        c = _collector()
        for k in shuffled:
            c.add("t.contradiction", k, table="matches", field="home_score", expected=1, actual=2)
        s = c.rule_summary()["t.contradiction"]
        assert (s["total"], s["open"], s["truncated"]) == (50, 50, True)
        orders.append([f.entity for f in c.samples()])
    assert orders[0] == orders[1] == orders[2] == ["match:000", "match:001", "match:002"]


def test_issue_id_is_stable_and_independent_of_values() -> None:
    a = R.Finding.make(RULES["t.contradiction"], "match:1", table="matches", field="home_score", expected=1, actual=2)
    b = R.Finding.make(RULES["t.contradiction"], "match:1", table="matches", field="home_score", expected=5, actual=9)
    c = R.Finding.make(RULES["t.contradiction"], "match:2", table="matches", field="home_score", expected=1, actual=2)
    assert a.issue_id == b.issue_id != c.issue_id
    assert a.issue_id.startswith("ic:")


def test_current_season_escalates_and_historical_does_not() -> None:
    c = _collector()
    c.add("t.anomaly", "match:old", season=1990)
    c.add("t.anomaly", "match:new", season=2026)
    sev = {f.entity: f.severity for f in c.samples()}
    assert sev == {"match:old": Severity.WARNING, "match:new": Severity.BLOCKING}


def test_exceptions_accepted_rejected_and_stale_are_visible() -> None:
    exc = (
        R.AcceptedException("t.anomaly", "match:old", "documented source error"),
        R.AcceptedException("t.anomaly", "match:new", "attempted suppression"),
        R.AcceptedException("t.anomaly", "match:gone", "no longer occurs"),
    )
    c = R.Collector(RULES, sample_limit=10, current_season=2026, exceptions=exc)
    c.add("t.anomaly", "match:old", season=1990)
    c.add("t.anomaly", "match:new", season=2026)
    status = {f.entity: (f.status, f.severity) for f in c.samples() if f.rule_id == "t.anomaly"}
    assert status["match:old"] == ("accepted", Severity.WARNING)
    assert status["match:new"] == ("open", Severity.BLOCKING)
    ex = c.exception_summary()
    assert [e["entity"] for e in ex["accepted"]] == ["match:old"]
    assert [e["entity"] for e in ex["rejected"]] == ["match:new"]
    assert [e["entity"] for e in ex["stale"]] == ["match:gone"]
    rejected = [f for f in c.samples() if f.rule_id == R.EXCEPTION_REJECTED]
    assert rejected and rejected[0].severity is Severity.BLOCKING


def test_unknown_rule_is_a_programming_error() -> None:
    with pytest.raises(KeyError):
        _collector().add("t.nope", "x")


def test_non_finite_values_are_rejected_in_canonical_bytes() -> None:
    with pytest.raises(ValueError):
        R.canonical_bytes({"a": float("nan")})
    with pytest.raises(ValueError):
        R.canonical_bytes({"a": [1, float("inf")]})


def test_report_digest_is_self_consistent_and_sensitive() -> None:
    body = {"schema": R.REPORT_SCHEMA, "outcome": "PASS", "report_sha256": ""}
    one = R.seal_report(dict(body))
    assert R.verify_report_digest(one)
    two = R.seal_report({**body, "outcome": "FAIL"})
    assert one["report_sha256"] != two["report_sha256"]
    tampered = {**one, "outcome": "FAIL"}
    assert not R.verify_report_digest(tampered)
    assert R.canonical_bytes(one) == R.canonical_bytes(json.loads(R.canonical_bytes(one)))


@pytest.mark.parametrize(("outcome", "code"), [(R.Outcome.PASS, 0), (R.Outcome.FAIL, 4), (R.Outcome.UNKNOWN, 8)])
def test_exit_codes(outcome: R.Outcome, code: int) -> None:
    assert R.exit_code(outcome) == code


def test_check_status_derivation() -> None:
    c = _collector()
    c.add("t.anomaly", "match:old", season=1990)  # warning only
    assert c.check_status("t.check", unknown=[]) is R.Status.PASS
    assert c.check_status("t.check", unknown=["evidence missing"]) is R.Status.UNKNOWN
    c.add("t.contradiction", "x")
    assert c.check_status("t.check", unknown=["evidence missing"]) is R.Status.FAIL
    assert c.check_status("t.other", unknown=[]) is R.Status.PASS


def test_overall_outcome() -> None:
    P, F, U, N = R.Status.PASS, R.Status.FAIL, R.Status.UNKNOWN, R.Status.NOT_APPLICABLE
    assert R.overall([(P, True), (N, True), (U, False)]) is R.Outcome.PASS
    assert R.overall([(P, True), (U, True)]) is R.Outcome.UNKNOWN
    assert R.overall([(U, True), (F, False)]) is R.Outcome.FAIL
