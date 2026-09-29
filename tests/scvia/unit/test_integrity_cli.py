"""`scvia check-integrity`: exit codes, outputs outside inputs, atomic writes, the findings stream."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from supercoach_via.cli import app
from supercoach_via.integrity import report as R
from supercoach_via.integrity.report_schema import IntegrityReport
from tests.scvia.unit import integrity_fixtures as fx

runner = CliRunner()


def invoke(*args: str) -> Any:
    return runner.invoke(app, ["check-integrity", *args])


def base_args(root: Path, out: Path, *extra: str) -> list[str]:
    return [
        "--data-root",
        str(root),
        "--scope",
        "data",
        "--as-of",
        fx.AS_OF,
        "--report",
        str(out / "report.json"),
        "--json",
        *extra,
    ]


@pytest.fixture
def clean(tmp_path: Path) -> Path:
    fx.with_sources(tmp_path / "var")
    return tmp_path / "var"


def test_clean_audit_exits_zero_and_writes_a_valid_sealed_report(clean: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    res = invoke(*base_args(clean, out))
    assert res.exit_code == 0, res.output
    summary = json.loads(res.stdout.strip().splitlines()[-1])
    assert summary["outcome"] == "PASS" and summary["exit"] == "ok"
    body = json.loads((out / "report.json").read_bytes())
    IntegrityReport.model_validate(body)
    assert R.verify_report_digest(body)
    assert summary["report_sha256"] == body["report_sha256"]
    execution = json.loads((out / "report.execution.json").read_text())
    assert execution["report_sha256"] == body["report_sha256"]
    assert "elapsed_s" in execution and "elapsed_s" not in json.dumps(body)
    assert str(clean) not in (out / "report.json").read_text()  # no absolute paths in the canonical report


def test_violations_exit_four(tmp_path: Path) -> None:
    fx.with_sources(tmp_path / "var", mutate=lambda r: r["player_games"][10].update(kicks=9, disposals=10))
    res = invoke(*base_args(tmp_path / "var", tmp_path / "out"))
    assert res.exit_code == 4, res.output
    assert json.loads(res.stdout.strip().splitlines()[-1])["outcome"] == "FAIL"


def test_incomplete_required_verification_exits_eight(tmp_path: Path) -> None:
    shas = fx.with_sources(tmp_path / "var")
    (tmp_path / "var" / "raw" / "objects" / shas["match"][:2] / shas["match"]).unlink()
    res = invoke(*base_args(tmp_path / "var", tmp_path / "out"))
    assert res.exit_code == 8, res.output


@pytest.mark.parametrize(
    "extra",
    [["--scope", "everything"], ["--as-of", "yesterday"], ["--as-of", "2026-09-28T12:00:00"], ["--workers", "0"]],
)
def test_invalid_usage_exits_two(clean: Path, tmp_path: Path, extra: list[str]) -> None:
    res = invoke(*base_args(clean, tmp_path / "out"), *extra)
    assert res.exit_code == 2, res.output
    assert not (tmp_path / "out" / "report.json").exists()


@pytest.mark.parametrize("where", ["data", "raw", "evidence"])
def test_outputs_inside_an_input_are_refused(clean: Path, tmp_path: Path, where: str) -> None:
    ev = tmp_path / "evidence"
    ev.mkdir()
    target = {"data": clean / "reports", "raw": clean / "raw" / "x", "evidence": ev / "r"}[where]
    before = fx.tree_digest(clean, ev)
    res = invoke(
        "--data-root",
        str(clean),
        "--scope",
        "data",
        "--as-of",
        fx.AS_OF,
        "--evidence",
        str(ev),
        "--report",
        str(target / "report.json"),
        "--json",
    )
    assert res.exit_code == 2, res.output
    assert fx.tree_digest(clean, ev) == before


def test_checker_crash_is_exit_nine_and_never_a_report(clean: Path, tmp_path: Path, monkeypatch: Any) -> None:
    from supercoach_via.integrity import checks_data

    def boom(ctx: Any) -> list[str]:
        raise RuntimeError("simulated checker bug")

    monkeypatch.setattr(
        checks_data,
        "CHECKS",
        [
            c if c.check_id != "aggregates.seasons" else type(c)(c.check_id, c.family, c.summary, boom)
            for c in checks_data.CHECKS
        ],
    )
    out = tmp_path / "out"
    res = invoke(*base_args(clean, out))
    assert res.exit_code == 9, res.output
    assert not (out / "report.json").exists()
    assert "simulated checker bug" in res.output


def test_interrupted_report_write_keeps_the_previous_report(clean: Path, tmp_path: Path, monkeypatch: Any) -> None:
    out = tmp_path / "out"
    assert invoke(*base_args(clean, out)).exit_code == 0
    previous = (out / "report.json").read_bytes()
    real_replace = os.replace

    def failing(src: Any, dst: Any) -> None:
        if str(dst).endswith("report.json"):
            raise OSError("disk full")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing)
    res = invoke(*base_args(clean, out))
    assert res.exit_code == 9, res.output
    assert (out / "report.json").read_bytes() == previous
    assert not [p for p in out.iterdir() if p.name.endswith(".tmp")]


def test_complete_findings_stream(tmp_path: Path) -> None:
    def many(r: dict[str, list[dict[str, Any]]]) -> None:
        for g in r["player_games"]:
            if g["season"] == 2026 and g["match_id"] == "m:2026:r02:alpha:beta:0":  # not source-captured
                g.update(kicks=40, handballs=30, disposals=70)  # unusual (warning) on four rows

    fx.with_sources(tmp_path / "var", mutate=many)
    out = tmp_path / "out"
    res = invoke(*base_args(tmp_path / "var", out, "--sample-limit", "2", "--findings-stream", str(out / "all.jsonl")))
    body = json.loads((out / "report.json").read_text())
    rule = next(r for r in body["rules"] if r["rule_id"] == "football.unusual_value")
    lines = [json.loads(x) for x in (out / "all.jsonl").read_text().splitlines()]
    assert rule["total"] == 4 and rule["sampled"] == 2 and rule["truncated"] is True
    assert sum(1 for x in lines if x["rule_id"] == "football.unusual_value") == 4
    stream = body["findings_stream"]
    assert stream["count"] == len(lines) and stream["complete"] is True
    import hashlib

    assert stream["sha256"] == hashlib.sha256((out / "all.jsonl").read_bytes()).hexdigest()
    assert res.exit_code == 0  # warnings only


def test_report_schema_file_is_generated_from_the_contract() -> None:
    from supercoach_via.integrity.report_schema import schema_json

    path = Path(__file__).resolve().parents[3] / "schemas" / "integrity" / "integrity-report.schema.json"
    assert path.read_text() == schema_json()


# -- output destinations must be distinct (follow-up finding: execution metadata overwrote the report) --


def _collision_cases(out: Path) -> list[list[str]]:
    link = out.parent / "link-to-out"
    if not link.exists():
        link.symlink_to(out, target_is_directory=True)
    rel = os.path.relpath(out / "report.json")
    return [
        ["--report", str(out / "report.json"), "--execution", str(out / "report.json")],
        ["--report", str(out / "report.json"), "--execution", rel],
        ["--report", str(out / "report.json"), "--execution", str(link / "report.json")],
        ["--report", str(out / "report.json"), "--findings-stream", str(out / "report.json")],
        ["--report", str(out / "r.json"), "--execution", str(out / "x.json"), "--findings-stream", str(link / "x.json")],
        ["--report", str(out / "report.json"), "--findings-stream", str(out / "report.execution.json")],
    ]


@pytest.mark.parametrize("case", range(6))
def test_output_destinations_that_alias_each_other_are_refused(clean: Path, tmp_path: Path, case: int) -> None:
    out = tmp_path / "out"
    out.mkdir()
    for name in ("report.json", "r.json", "x.json", "report.execution.json"):
        (out / name).write_bytes(b"previous " + name.encode())
    before = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    args = _collision_cases(out)[case]
    res = invoke("--data-root", str(clean), "--scope", "data", "--as-of", fx.AS_OF, "--json", *args)
    assert res.exit_code == 2, res.output
    assert "same file" in res.output
    assert {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()} == before


def test_distinct_destinations_are_all_written(clean: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    res = invoke("--data-root", str(clean), "--scope", "data", "--as-of", fx.AS_OF, "--json",
                 "--report", str(out / "a.json"), "--execution", str(out / "b.json"),
                 "--findings-stream", str(out / "c.jsonl"))  # fmt: skip
    assert res.exit_code == 0, res.output
    assert json.loads((out / "a.json").read_text())["schema"] == R.REPORT_SCHEMA
    assert json.loads((out / "b.json").read_text())["schema"] == "scvia.integrity-execution/1"
    assert (out / "c.jsonl").exists()


def test_failure_between_outputs_reports_what_was_written(clean: Path, tmp_path: Path, monkeypatch: Any) -> None:
    out = tmp_path / "out"
    real_replace = os.replace

    def failing(src: Any, dst: Any) -> None:
        if str(dst).endswith("report.execution.json"):
            raise OSError("disk full")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing)
    res = invoke(*base_args(clean, out, "--findings-stream", str(out / "all.jsonl")))
    assert res.exit_code == 9, res.output
    assert "no report was written" not in res.output
    assert "written before the failure" in res.output
    assert str(out / "report.json") in res.output and str(out / "all.jsonl") in res.output
    assert "report.execution.json" in res.output.split("not written")[1]
