"""Strict contracts: committed JSON Schemas match the models; completion/validation records validate (D04)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from supercoach_via.reconciliation import schema as S

REPO = Path(__file__).resolve().parents[3]


def test_committed_schemas_equal_the_generated_ones(tmp_path: Path) -> None:
    names = S.export_schemas(tmp_path)
    assert sorted(names) == sorted(p.name for p in (REPO / "schemas" / "reconciliation").glob("*.json"))
    for n in names:
        assert (tmp_path / n).read_bytes() == (REPO / "schemas" / "reconciliation" / n).read_bytes(), n


def _completion(**kw: object) -> dict[str, object]:
    out = {"status": "PENDING", "report_sha256": None, "note": ""}
    base: dict[str, object] = {
        "created_utc": "2026-10-02T00:00:00Z",
        "model": {"requested": "claude-sonnet-5-5", "resolved": "claude-sonnet-5-5", "resolution_source": "session metadata"},
        "design": out, "implementation": out, "execution": out, "data": {"snapshot": out}, "overall_data_verdict": "UNKNOWN",
        "inputs": {}, "reports": {}, "reproducibility": {}, "acceptance": out, "open_items": [],
    }  # fmt: skip
    base.update(kw)
    return base


def test_completion_keeps_design_implementation_execution_and_data_statuses_separate() -> None:
    c = S.Completion.model_validate_json(json.dumps(_completion()))
    assert c.design.status == c.implementation.status == "PENDING" and c.overall_data_verdict is S.Verdict.UNKNOWN
    with pytest.raises(ValidationError):
        S.Completion.model_validate_json(json.dumps(_completion(surprise=1)))  # unknown keys are refused
    with pytest.raises(ValidationError):
        S.Completion.model_validate_json(json.dumps(_completion(overall_data_verdict="MAYBE")))
    with pytest.raises(ValidationError):
        S.Completion.model_validate_json(
            json.dumps(_completion(design={"status": "GREAT", "report_sha256": None, "note": ""}))
        )


def test_models_round_trip_canonically() -> None:
    c = S.Completion.model_validate_json(json.dumps(_completion()))
    again = S.Completion.model_validate_json(S.canonical_dump(c))
    assert S.canonical_dump(again) == S.canonical_dump(c) and json.loads(S.canonical_dump(c))["kind"].startswith(
        "afltables"
    )


def test_finding_schema_covers_every_category_severity() -> None:
    from supercoach_via.reconciliation.findings import CATEGORIES, make_finding

    for cat in CATEGORIES:
        f = make_finding(cat, layer="snapshot", rule_id="R-X", detail="d")
        assert S.Finding.model_validate_json(json.dumps(f)).severity == CATEGORIES[cat][0]


def _step(status: str, **counts: int) -> dict[str, object]:
    base = {"passed": 10, "failed": 0, "errors": 0, "skipped": 0, "deselected": 0}
    base.update(counts)
    return {"name": "fast tier", "command": "pytest", "status": status, "detail": "", "counts": base}


def test_a_validation_step_cannot_say_pass_while_counting_a_failure_or_error() -> None:  # B5
    S.ValidationStep.model_validate_json(json.dumps(_step("PASS")))
    for bad in ({"failed": 1}, {"errors": 1}):
        with pytest.raises(ValidationError, match="PASS"):
            S.ValidationStep.model_validate_json(json.dumps(_step("PASS", **bad)))
    S.ValidationStep.model_validate_json(json.dumps(_step("FAIL", failed=1)))  # an honest FAIL is fine


def test_skips_must_carry_reasons_and_a_baseline_failure_needs_evidence() -> None:  # B5
    with pytest.raises(ValidationError, match="skip_reasons"):
        S.ValidationStep.model_validate_json(json.dumps(_step("PASS", skipped=2)))
    ok = {**_step("PASS", skipped=2), "skip_reasons": {"no repo venv": 2}}
    S.ValidationStep.model_validate_json(json.dumps(ok))
    with pytest.raises(ValidationError, match="baseline"):
        S.ValidationStep.model_validate_json(json.dumps({**_step("BASELINE_FAIL", failed=1)}))
    base = {**_step("BASELINE_FAIL", failed=1), "baseline_evidence": "same test fails at 21ca717 in the same env"}
    S.ValidationStep.model_validate_json(json.dumps(base))


def test_implementation_complete_requires_every_required_step_to_pass() -> None:  # B5
    passing = S.ImplementationValidation.model_validate_json(
        json.dumps(
            {
                "created_utc": "x", "branch": "b", "base_commit": "c", "libraries": {},
                "steps": [_step("PASS")], "baseline_failures": [], "new_failures": [],
            }
        )
    )  # fmt: skip
    assert S.implementation_status(passing) == "COMPLETE"
    failing = passing.model_copy(
        update={"steps": [S.ValidationStep.model_validate_json(json.dumps(_step("FAIL", failed=1)))]}
    )
    assert S.implementation_status(failing) == "INCOMPLETE"
