"""`scvia reconcile-afltables`: wiring, exit codes and refusal paths (DESIGN section 11; T29)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
from typer.testing import CliRunner

from supercoach_via.cli import app
from supercoach_via.reconciliation import cli as rcli
from tests.scvia.unit import integrity_fixtures as fx
from tests.scvia.unit.recon_site import FakeClock, FakeSite, site_pages, small_world

runner = CliRunner()


def _plan_args(tmp_path: Path, *extra: str) -> list[str]:
    data = tmp_path / "data"
    if not data.exists():
        fx.build(data)
    return [
        "reconcile-afltables",
        "plan",
        "--data-root",
        str(data),
        "--snapshot",
        "current",
        "--through-date",
        "2026-09-30",
        "--scope",
        "all",
        "--run-dir",
        str(tmp_path / "run"),
        *extra,
    ]


def test_command_group_is_registered_and_documents_its_subcommands() -> None:
    res = runner.invoke(app, ["reconcile-afltables", "--help"])
    assert res.exit_code == 0
    for sub in ("plan", "capture", "capture-evidence", "compare"):
        assert sub in res.output


def test_plan_writes_plan_json_and_reports_its_identity(tmp_path: Path) -> None:
    res = runner.invoke(app, [*_plan_args(tmp_path), "--json"])
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout.strip().splitlines()[-1])
    plan = json.loads((tmp_path / "run" / "plan.json").read_text())
    assert out["plan_id"] == plan["plan_id"] and out["full_population"] is True
    assert plan["inputs"]["snapshot"]["snapshot_id"].startswith("sha256:")


def test_plan_makes_no_network_access(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import socket

    def deny(*_a: object, **_k: object) -> None:
        raise AssertionError("plan must not touch the network")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)
    assert runner.invoke(app, _plan_args(tmp_path)).exit_code == 0


def test_plan_refuses_unsafe_or_invalid_invocations_with_exit_2(tmp_path: Path) -> None:
    bad_date = runner.invoke(app, [*_plan_args(tmp_path)[:-6], "--through-date", "2026-13-40", "--scope", "all",
                                   "--run-dir", str(tmp_path / "run")])  # fmt: skip
    assert bad_date.exit_code == 2
    inside = runner.invoke(app, ["reconcile-afltables", "plan", "--data-root", str(tmp_path / "data"),
                                 "--through-date", "2026-09-30", "--scope", "all",
                                 "--run-dir", str(tmp_path / "data" / "fragments" / "run")])  # fmt: skip
    assert inside.exit_code == 2 and "inside an input root" in inside.output


def test_capture_requires_explicit_network_consent_and_a_valid_plan(tmp_path: Path) -> None:
    assert runner.invoke(app, _plan_args(tmp_path)).exit_code == 0
    plan = str(tmp_path / "run" / "plan.json")
    no_flag = runner.invoke(app, ["reconcile-afltables", "capture", "--plan", plan, "--resume"])
    assert no_flag.exit_code == 2 and "--allow-network" in no_flag.output
    doc = json.loads((tmp_path / "run" / "plan.json").read_text())
    doc["scope"]["through_date"] = "2020-01-01"
    (tmp_path / "tampered.json").write_text(json.dumps(doc))
    tampered = runner.invoke(
        app, ["reconcile-afltables", "capture", "--plan", str(tmp_path / "tampered.json"), "--allow-network"]
    )
    assert tampered.exit_code == 2 and "identity" in tampered.output
    missing = runner.invoke(
        app, ["reconcile-afltables", "capture", "--plan", str(tmp_path / "nope.json"), "--allow-network"]
    )
    assert missing.exit_code == 2


def test_capture_runs_resumes_and_maps_exit_codes(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # shrink the audit to the two fake seasons the fake site serves (a legitimate, identity-bearing scope change)
    monkeypatch.setattr("supercoach_via.reconciliation.schema.FIRST_SEASON", 2025)
    assert runner.invoke(app, _plan_args(tmp_path)).exit_code == 0
    plan_path = tmp_path / "run" / "plan.json"
    clock = FakeClock()
    site = FakeSite(clock, site_pages(small_world()))
    monkeypatch.setattr(rcli, "make_client", lambda _policies: site.client())
    monkeypatch.setattr(rcli, "SYSTEM_CLOCK", clock)
    args = ["reconcile-afltables", "capture", "--plan", str(plan_path), "--allow-network"]
    first = runner.invoke(app, [*args, "--max-requests", "5"])
    assert first.exit_code == 8, first.output
    existing = runner.invoke(app, args)  # a checkpoint exists: resuming must be explicit
    assert existing.exit_code == 2 and "--resume" in existing.output
    done = runner.invoke(app, [*args, "--resume"])
    assert done.exit_code == 0, done.output
    assert "capture_complete: true" in done.output.replace('"', "")


def test_scope_sample_requires_explicit_profiles(tmp_path: Path) -> None:
    res = runner.invoke(app, [*_plan_args(tmp_path)[:-4], "--scope", "sample", "--run-dir", str(tmp_path / "run")])
    assert res.exit_code == 2 and "sample" in res.output
    ok = runner.invoke(
        app,
        [*_plan_args(tmp_path)[:-4], "--scope", "sample", "--run-dir", str(tmp_path / "run2"),
         "--sample-profile", "https://afltables.com/afl/stats/players/S/Scott_Pendlebury.html"],
    )  # fmt: skip
    assert ok.exit_code == 0, ok.output
    assert json.loads((tmp_path / "run2" / "plan.json").read_text())["scope"]["full_population"] is False


def test_http_client_factory_uses_the_reconciliation_policy_and_a_descriptive_agent() -> None:
    from supercoach_via.reconciliation.urls import load_reconciliation_policies

    client = rcli.make_client(load_reconciliation_policies())
    try:
        assert "reconciliation" in client._client.headers["user-agent"]
        assert client.archive is None  # capture archives bytes itself and never writes validators
    finally:
        client.close()
    assert isinstance(httpx.Client, type)


def test_capture_evidence_needs_consent_then_acquires_robots_and_the_page_into_its_own_archive(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    from supercoach_via.reconciliation import evidence as EV
    from tests.scvia.unit.test_reconciliation_evidence import brownlow_page

    args = ["reconcile-afltables", "capture-evidence", "--run-dir", str(tmp_path / "run")]
    refused = runner.invoke(app, args)
    assert refused.exit_code == 2 and "--allow-network" in refused.output
    clock = FakeClock()
    site = FakeSite(clock, {"/afl/brownlow/brownlow_idx.html": brownlow_page()})
    monkeypatch.setattr(
        rcli, "make_evidence_client", lambda _policies: site.client(EV.evidence_policies(), EV.USER_AGENT)
    )
    monkeypatch.setattr(rcli, "SYSTEM_CLOCK", clock)
    done = runner.invoke(app, [*args, "--allow-network", "--json"])
    assert done.exit_code == 0, done.output
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["state"] == "complete" and out["requests"] == 2 and len(out["sha256"]) == 64
    assert (tmp_path / "run" / "evidence" / "receipt.json").is_file()
    assert not (tmp_path / "run" / "capture").exists()  # the frozen corpus archive is never touched


def test_the_evidence_client_uses_its_own_policy_and_a_descriptive_agent() -> None:
    from supercoach_via.reconciliation import evidence as EV

    client = rcli.make_evidence_client(EV.evidence_policies())
    try:
        assert client._client.headers["user-agent"] == EV.USER_AGENT and client.archive is None
        assert set(client.policies.sources) == {EV.EVIDENCE_POLICY_NAME}
    finally:
        client.close()
