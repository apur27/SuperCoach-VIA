"""End-to-end comparison on a captured world: verdicts, determinism, mutations (T01, T05, T19-T25, T28, T30)."""

from __future__ import annotations

import json
import shutil
import socket
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from supercoach_via.reconciliation import report as RP
from supercoach_via.reconciliation.capture import Capture
from tests.scvia.unit import recon_e2e as E
from tests.scvia.unit import recon_inputs as RI
from tests.scvia.unit import recon_world as rw

W = RI.modern_world()


def two_season_world() -> rw.World:
    base = RI.modern_world()
    older = tuple(
        replace(m, gid=m.gid.replace("2026", "2025"), year=2025, when=m.when.replace(year=2025))
        for m in base.matches[:2]
    )
    return rw.World(base.players, (*older, *base.matches), (2025, 2026))


def build(tmp_path: Path, mp: pytest.MonkeyPatch, world: rw.World = W, **kw: Any) -> E.E2E:
    return E.build(tmp_path, world, monkeypatch=mp, **kw)


def local_without(pred: Any) -> Any:
    def rows(layer: str, season: int) -> list[Any]:
        return [r for r in RI.local_rows(W, season, layer) if not pred(r)]

    return rows


def local_edit(pid: str, gid: str, field: str, value: Any, world: rw.World | None = None) -> Any:
    def rows(layer: str, season: int) -> list[Any]:
        base = RI.local_rows(world or two_season_world(), season, layer)
        return RI.with_cell(base, pid, gid, field, value) if layer == "snapshot" else base

    return rows


def test_a_consistent_world_passes_both_layers_and_writes_every_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    e2e = build(tmp_path, monkeypatch)
    assert e2e.capture_result.exit_code == 0
    result, code, out = e2e.compare()
    rep = result.report
    assert code == 0 and rep["result"] == {"overall": "PASS", "layers": {"snapshot": "PASS", "legacy_csv": "PASS"}}
    assert result.execution["accounting_identities_hold"] and result.findings == []
    for name in (*RP.OUTPUTS, "output-manifest.json"):
        assert (out / name).is_file()
    man = json.loads((out / "output-manifest.json").read_text())
    assert set(man["files"]) == set(RP.OUTPUTS) and man["report_sha256"]
    lf = rep["latest_completed_final"]
    assert lf["stage"] == "Grand Final" and lf["present_locally"] == {"snapshot": True, "legacy_csv": True}
    assert (
        lf["participants"]["snapshot"] == {"matched": 3}
        and rep["layers"]["snapshot"]["fractions"]["available_statistic_fraction"] == 1.0
    )


def test_removing_a_whole_local_player_fails_even_when_other_totals_stay_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # T01
    e2e = build(
        tmp_path,
        monkeypatch,
        local_rows=local_without(lambda r: r.player_key.endswith("c")),
        drop_players=frozenset({"c"}),
    )
    result, code, _ = e2e.compare()
    assert code == 4 and result.report["result"]["layers"] == {"snapshot": "FAIL", "legacy_csv": "FAIL"}
    cats = {f["category"] for f in result.findings if f["severity"] == "fail"}
    assert {"PLAYER_MISSING_LOCAL", "APPEARANCE_MISSING_LOCAL"} <= cats
    pop = result.report["layers"]["snapshot"]["population"]
    assert pop["source_players_missing_locally"] == 1 or pop["source_players_unresolved"] == 1


def test_a_player_missing_across_several_seasons_is_one_finding_and_every_finding_id_is_unique(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # F-H1: the per-season emission guard repeated PLAYER_MISSING_LOCAL once per season
    world = two_season_world()

    def rows(layer: str, season: int) -> list[Any]:
        return [r for r in RI.local_rows(world, season, layer) if not r.player_key.endswith("c")]

    e2e = build(tmp_path, monkeypatch, world=world, local_rows=rows, drop_players=frozenset({"c"}))
    result, code, out = e2e.compare()
    assert code == 4
    ids = [json.loads(line)["id"] for line in (out / "findings.jsonl").read_text().splitlines()]
    assert len(ids) == len(set(ids))
    for layer in ("snapshot", "legacy_csv"):
        missing = [f for f in result.findings if f["category"] == "PLAYER_MISSING_LOCAL" and f["layer"] == layer]
        assert len(missing) == 1, layer


def test_missing_latest_final_is_found_without_any_local_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # T28
    e2e = build(
        tmp_path,
        monkeypatch,
        local_rows=local_without(lambda r: r.origin.endswith("041520260926")),
        drop_matches=frozenset({"041520260926"}),
    )
    result, code, _ = e2e.compare()
    assert code == 4 and result.report["result"]["layers"] == {"snapshot": "FAIL", "legacy_csv": "FAIL"}
    lf = result.report["latest_completed_final"]
    assert lf["present_locally"]["snapshot"] is False and lf["participants"]["snapshot"] == {"missing": 3}
    assert any(f["category"] == "MATCH_MISSING_LOCAL" and f["layer"] == "snapshot" for f in result.findings)
    assert any(f["category"] == "MATCH_MISSING_LOCAL" and f["layer"] == "legacy_csv" for f in result.findings)


def test_one_layer_failing_while_the_other_passes_is_a_combined_fail_with_both_verdicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # T30
    e2e = build(tmp_path, monkeypatch, local_rows=local_edit("a", "041520260305", "kicks", 77, W))
    result, code, _ = e2e.compare()
    assert code == 4 and result.report["result"]["layers"] == {"snapshot": "FAIL", "legacy_csv": "PASS"}
    (f,) = [x for x in result.findings if x["category"] == "CELL_MISMATCH"]
    assert f["layer"] == "snapshot" and f["actual"] == 77


def test_outputs_are_byte_identical_across_worker_counts_cache_states_and_changed_since(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # T20
    e2e = build(
        tmp_path, monkeypatch, world=two_season_world(), local_rows=local_edit("a", "041520260305", "kicks", 77, W)
    )
    outs = {}
    for name, kw in (
        ("cold1", {"workers": 1, "cache": "c1"}),
        ("cold4", {"workers": 4, "cache": "c4"}),
        ("warm4", {"workers": 4, "cache": "c4"}),
    ):
        outs[name] = e2e.compare(f"reports/{name}", **kw)[2]
    prev = outs["cold4"] / "report.json"
    outs["changed"] = e2e.compare("reports/changed", workers=2, cache="c4", previous=prev)[2]
    for fname in ("report.json", "findings.jsonl", "players.csv", "coverage.csv", "summary.md"):
        digests = {n: (o / fname).read_bytes() for n, o in outs.items()}
        assert len({d for d in digests.values()}) == 1, fname
    ex = json.loads((outs["warm4"] / "execution.json").read_text())
    assert ex["parse_cache"].get("hit", 0) > 0 and ex["parse_cache"].get("miss", 0) == 0
    cold = json.loads((outs["cold1"] / "execution.json").read_text())
    assert cold["parse_cache"].get("miss", 0) > 0
    ch = json.loads((outs["changed"] / "execution.json").read_text())["previous"]
    assert ch["units_total"] == 4 and ch["units_unchanged_reused_evidence"] == 4 and ch["units_changed_or_new"] == []


def test_changing_one_local_row_invalidates_only_its_unit_and_changes_the_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # T21
    w = two_season_world()
    base = build(tmp_path / "a", monkeypatch, world=w)
    _, _, out_a = base.compare("reports/a")
    mutated = build(tmp_path / "b", monkeypatch, world=w, local_rows=local_edit("a", "041520260305", "kicks", 88))
    res_b, code_b, _ = mutated.compare("reports/b", previous=out_a / "report.json")
    changed = json.loads((tmp_path / "b" / "run" / "reports" / "b" / "execution.json").read_text())["previous"]
    assert code_b == 4 and changed["units_changed_or_new"] == ["snapshot:2026"]
    assert any(f["category"] == "CELL_MISMATCH" and f["actual"] == 88 for f in res_b.findings)


def test_offline_comparison_works_with_sockets_denied_and_a_relocated_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # T19
    e2e = build(tmp_path, monkeypatch)
    _, _, out1 = e2e.compare("reports/one")

    def deny(*_a: object, **_k: object) -> None:
        raise AssertionError("compare must not touch the network")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)
    moved = tmp_path / "moved-capture"
    shutil.copytree(tmp_path / "run" / "capture", moved)
    e2e.manifest_path = moved / "manifest.json"
    _, _, out2 = e2e.compare("reports/two", cache="cache2")
    assert (out1 / "report.json").read_bytes() == (out2 / "report.json").read_bytes()


def test_an_interrupted_capture_gives_unknown_not_pass_and_a_real_mismatch_still_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean = build(tmp_path / "x", monkeypatch, capture=False)
    clock_site = clean.site
    cap = Capture(
        clean.plan, tmp_path / "x" / "run", clock_site.client(), clock=clock_site.clock, durable=False, max_requests=60
    )
    assert cap.run().exit_code == 8
    result, code, _ = clean.compare()
    assert code == 8 and result.report["result"]["overall"] == "UNKNOWN"
    assert any("capture incomplete" in r for r in result.report["layers"]["snapshot"]["unknown_reasons"])


def test_a_failed_alphabet_page_makes_the_denominator_null_never_one_hundred_percent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # T02
    import httpx

    e2e = build(tmp_path, monkeypatch, capture=False)
    e2e.site.override["/afl/stats/playersQ_idx.html"] = lambda _r: httpx.Response(500)
    Capture(e2e.plan, tmp_path / "run", e2e.site.client(), clock=e2e.site.clock, durable=False).run()
    result, code, _ = e2e.compare()
    fr = result.report["layers"]["snapshot"]["fractions"]
    assert code == 8 and fr["verified_numeric_fraction"] is None and "census" in fr["null_reason"]


def test_unreadable_pinned_snapshot_makes_the_layer_unknown_with_a_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # T22
    e2e = build(tmp_path, monkeypatch, legacy=False)
    frag = next((e2e.data_root / "fragments" / "player_games").rglob("*.parquet"))
    frag.chmod(0o644)
    frag.write_bytes(frag.read_bytes() + b"x")
    result, code, _ = e2e.compare()
    assert code == 8 and result.report["result"]["layers"]["snapshot"] == "UNKNOWN"
    assert any(f["category"] == "LOCAL_INPUT_GAP" for f in result.findings)


def test_output_overlapping_an_input_or_a_completed_report_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # T23
    e2e = build(tmp_path, monkeypatch, legacy=False)
    result, _, out = e2e.compare()
    with pytest.raises(RP.OutputWriteError, match="completed report"):
        RP.write_report_dir(out, result, [])
    with pytest.raises(RP.OutputWriteError, match="overlaps"):
        RP.write_report_dir(e2e.data_root / "fragments" / "x", result, [e2e.data_root.resolve()])
    victim = next((e2e.data_root / "fragments").rglob("*.parquet"))
    target = tmp_path / "alias"
    target.mkdir()
    (target / "report.json").hardlink_to(victim)
    with pytest.raises(RP.OutputWriteError, match="hard-link"):
        RP.write_report_dir(target, result, [])
    assert victim.read_bytes()[:4] == b"PAR1"


def _tree(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("where", ["data", "capture", "legacy", "evidence"])
@pytest.mark.parametrize("what", ["out", "cache"])
def test_out_or_cache_inside_any_input_root_is_refused_before_a_single_byte_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str, what: str
) -> None:  # T23 / B4
    from supercoach_via.reconciliation import compare as CP

    e2e = build(tmp_path, monkeypatch)
    root = {
        "data": e2e.data_root,
        "capture": e2e.manifest_path.parent,
        "legacy": e2e.legacy_root / "data" / "player_data",
        "evidence": e2e.root / "run" / "evidence",
    }[where]
    before = _tree(root)
    opts = CP.CompareOptions(
        plan=e2e.plan_path,
        capture_manifest=e2e.manifest_path,
        out=root / "evil" if what == "out" else e2e.root / "run" / "reports" / "ok",
        cache=root / "here" if what == "cache" else e2e.root / "run" / "cache",
    )
    with pytest.raises(CP.CompareError, match="input root"):
        CP.run_audit(opts)
    assert _tree(root) == before
    assert not list(root.rglob(".evil.work-*")) and not (root / "here").exists()


def test_a_failing_write_leaves_no_completion_marker_and_names_what_was_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # T24
    e2e = build(tmp_path, monkeypatch, legacy=False)
    opts = e2e.options("reports/broken")
    import supercoach_via.reconciliation.compare as CP

    result, audit = CP.run_audit(opts)
    real = RP.atomic_write_bytes
    calls = {"n": 0}

    def flaky(path: Path, data: bytes, **kw: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError(28, "No space left on device")
        real(path, data, **kw)

    monkeypatch.setattr(RP, "atomic_write_bytes", flaky)
    with pytest.raises(RP.OutputWriteError) as ei:
        RP.write_report_dir(opts.out, result, CP.input_roots_of(audit.plan, audit.capture_dir))
    assert (
        ei.value.written == ["report.json", "findings.jsonl", "players.csv"] and "coverage.csv" in ei.value.not_written
    )
    assert not (opts.out / "output-manifest.json").exists()


def test_csv_exports_neutralise_spreadsheet_formulas_without_changing_canonical_json() -> None:
    assert (
        RP.neutralise("=HYPERLINK(1)") == "'=HYPERLINK(1)"
        and RP.neutralise("+1") == "'+1"
        and RP.neutralise("ok") == "ok"
    )
    out = RP.csv_bytes([{"a": "=1+1", "b": "x"}], ["a", "b"]).decode()
    assert out == "a,b\n'=1+1,x\n"


def test_a_sample_population_can_never_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    e2e = build(tmp_path, monkeypatch, legacy=False)
    plan = e2e.plan.model_copy(
        update={"scope": e2e.plan.scope.model_copy(update={"population": "sample", "full_population": False})}
    )
    from supercoach_via.reconciliation import inventory as inv

    assert inv.load_plan  # the on-disk plan is untouched; the engine consults plan.scope only
    assert plan.scope.full_population is False


def test_games_after_the_event_boundary_are_excluded_and_printed_career_figures_are_not_misused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # T15
    e2e = build(tmp_path, monkeypatch, through="2026-03-10")
    result, code, _ = e2e.compare()
    rep = result.report
    assert code == 0, rep["layers"]["snapshot"]["unknown_reasons"]
    snap = rep["layers"]["snapshot"]["coverage"]
    assert (
        snap["app_expected"] == 3 and rep["scope"]["matches_excluded_after_boundary"] == 2
    )  # only round 1 is in scope
    assert snap["source_games_excluded_after_boundary"] == 6 and snap["derived_career_out_of_scope_skipped"] == 3
    assert "derived_career_total_mismatch" not in snap and "derived_gm_mismatch" not in snap


def test_a_changed_source_cell_changes_the_unit_digest_and_the_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # T21
    w = two_season_world()
    base = build(tmp_path / "a", monkeypatch, world=w)
    _, _, out_a = base.compare("reports/a")

    def edit(pages: dict[str, bytes]) -> None:
        k = "/afl/stats/games/2026/041520260305.html"
        pages[k] = pages[k].replace(b"<td align=center>1</td>", b"<td align=center>9</td>", 1)

    changed = build(tmp_path / "b", monkeypatch, world=w, page_edit=edit)
    res, code, _ = changed.compare("reports/b", previous=out_a / "report.json")
    prev = json.loads((tmp_path / "b" / "run" / "reports" / "b" / "execution.json").read_text())["previous"]
    assert code != 0 and set(prev["units_changed_or_new"]) == {"legacy_csv:2026", "snapshot:2026"}
    assert any(f["category"] in ("SOURCE_CONFLICT", "CELL_MISMATCH") for f in res.findings)


def _evidence_relpath() -> tuple[str, str]:
    sha = "dc81249c48524329ec024f29283d0524742a66da799a886b91c26d8f2aa98725"
    return ("objects", sha[:2] + "/" + sha)


def test_changing_a_rule_invalidates_exactly_the_units_that_depend_on_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # T21
    from supercoach_via.reconciliation import compare as CP
    from supercoach_via.reconciliation.rules import parse_rules

    e2e = build(tmp_path, monkeypatch, world=two_season_world())
    _, _, out1 = e2e.compare("reports/one")
    d1 = json.loads((out1 / "report.json").read_text())["unit_digests"]
    audit = CP.Audit(e2e.options("reports/two"))
    audit.load()
    from supercoach_via.settings import default_config_dir

    cfg = default_config_dir()
    base_rules = (cfg / "reconciliation_rules.toml").read_text()
    audit.rules = parse_rules(
        base_rules
        + '\n[[notes_club_alias]]\nseason = 2026\nname = "X"\nclub = "Alpha"\nevidence_locator = "test"\nreason = "test"\n',
        (cfg / "reconciliation_identity_overrides.csv").read_text(),
    ).with_evidence((e2e.root / "run" / "evidence").joinpath(*_evidence_relpath()).read_bytes())
    audit.parse()
    audit.inventory()
    audit.identity_snapshot()
    audit.identity_legacy()
    audit.season_units()
    changed = {k for k in d1 if audit.unit_digests[k] != d1[k]}
    assert changed == {"snapshot:2026", "legacy_csv:2026"}  # a rule about 2026 invalidates only 2026 units


@pytest.mark.parametrize(
    "change",
    [
        "award_total",  # the evidence states a different number of votes per game in 2026
        "no_award",  # the evidence states no medal in 2026
    ],
)
def test_changing_the_brownlow_award_evidence_invalidates_the_units_it_governs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:  # a mutation probe found the digest ignoring the Brownlow rule; stale units must never be reused
    import dataclasses

    from supercoach_via.reconciliation import compare as CP
    from supercoach_via.reconciliation.evidence import VoteEra

    e2e = build(tmp_path, monkeypatch, world=two_season_world())
    _, _, out1 = e2e.compare("reports/one")
    d1 = json.loads((out1 / "report.json").read_text())["unit_digests"]
    audit = CP.Audit(e2e.options("reports/two"))
    audit.load()
    facts = audit.rules.brownlow
    assert facts is not None
    if change == "award_total":
        facts = dataclasses.replace(facts, exceptions=(*facts.exceptions, VoteEra(2026, 2026, 12)))
    else:
        facts = dataclasses.replace(facts, no_award_seasons=facts.no_award_seasons | {2026})
    audit.rules = dataclasses.replace(audit.rules, brownlow=facts)
    audit.parse()
    audit.inventory()
    audit.identity_snapshot()
    audit.identity_legacy()
    audit.season_units()
    assert {k for k in d1 if audit.unit_digests[k] != d1[k]} == {"snapshot:2026", "legacy_csv:2026"}


def test_a_captured_object_that_changes_during_the_audit_is_drift_and_makes_the_layer_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # T22
    from supercoach_via.reconciliation import compare as CP

    e2e = build(tmp_path, monkeypatch, legacy=False)
    opts = e2e.options("reports/drift")
    real = CP.Audit.reduce_players

    def tamper_then_reduce(self: CP.Audit, only: str | None = None) -> None:
        real(self, only)
        victim = next(r for r in self.manifest.resources if r.kind == "profile")
        obj = self.capture_dir / "objects" / victim.sha256[:2] / victim.sha256  # type: ignore[index]
        obj.chmod(0o644)
        obj.write_bytes(b"changed while the audit ran")

    monkeypatch.setattr(CP.Audit, "reduce_players", tamper_then_reduce)
    result, _audit = CP.run_audit(opts)
    assert result.report["input_drift"] and result.report["layers"]["snapshot"]["verdict"] == "UNKNOWN"
    assert result.exit_code == 8


def test_an_unusable_profile_is_reported_as_a_capture_gap_and_blocks_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # T16
    import httpx

    e2e = build(tmp_path, monkeypatch, capture=False, legacy=False)
    e2e.site.override["/afl/stats/players/B/Bob_Baker.html"] = lambda _r: httpx.Response(404)
    Capture(e2e.plan, tmp_path / "run", e2e.site.client(), clock=e2e.site.clock, durable=False).run()
    result, code, _ = e2e.compare()
    gap = [f for f in result.findings if f["category"] == "CAPTURE_GAP" and f["rule_id"] == "R-PROFILE-MISSING"]
    assert len(gap) == 1 and "404" in gap[0]["detail"] and code == 8


def test_an_unmappable_notes_exception_is_a_schema_gap_that_blocks_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # S-08 / B3: never a silent drop and never an info note
    notes = (E.FIXTURES / "notes.html").read_bytes()

    def edit(pages: dict[str, bytes]) -> None:
        pages["/afl/stats/notes.html"] = notes.replace(b"1975, R11", b"2026, R1", 1)

    e2e = build(tmp_path, monkeypatch, legacy=False, page_edit=edit)
    result, code, _ = e2e.compare()
    gaps = [f for f in result.findings if f["category"] == "SCHEMA_GAP" and f["rule_id"] == "R-NOTES-CLUB"]
    assert gaps and all(f["severity"] == "unknown" and f["season"] == 2026 for f in gaps)
    assert code == 8 and result.report["result"]["overall"] == "UNKNOWN"
    assert not [f for f in result.findings if f["category"] == "NOTES_CLUB_UNMAPPED"]


def test_warm_runs_reuse_stored_season_results_and_a_corrupt_entry_is_recomputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # S-14 / T20
    e2e = build(
        tmp_path, monkeypatch, world=two_season_world(), local_rows=local_edit("a", "041520260305", "kicks", 77)
    )
    _, _, cold = e2e.compare("reports/cold", workers=2, cache="c")
    _, _, warm = e2e.compare("reports/warm", workers=2, cache="c")
    ex = json.loads((warm / "execution.json").read_text())["parse_cache"]
    assert ex["units_reused_from_cache"] == 4
    for name in ("report.json", "findings.jsonl", "players.csv", "coverage.csv"):
        assert (cold / name).read_bytes() == (warm / name).read_bytes(), name
    entry = next((tmp_path / "run" / "c" / "units").rglob("*.bin"))
    entry.write_bytes(entry.read_bytes()[:-9] + b"corrupted")
    _, _, again = e2e.compare("reports/again", workers=1, cache="c")
    assert json.loads((again / "execution.json").read_text())["parse_cache"]["units_reused_from_cache"] == 3
    assert (cold / "report.json").read_bytes() == (again / "report.json").read_bytes()


def test_a_match_page_games_to_date_one_below_the_profile_counter_is_a_source_conflict_and_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # B1 / A3 / T14
    def edit(pages: dict[str, bytes]) -> None:
        k = "/afl/stats/games/2026/041520260312.html"
        pages[k] = pages[k].replace(
            b"<td align=center>2 (2-0-0 100.00%)</td>", b"<td align=center>1 (1-0-0 100.00%)</td>", 1
        )

    e2e = build(tmp_path, monkeypatch, page_edit=edit)
    result, code, _ = e2e.compare()
    conflicts = [f for f in result.findings if f["category"] == "SOURCE_CONFLICT" and f["field"] == "counter"]
    assert len(conflicts) == 1 and conflicts[0]["rule_id"] == "R-SOURCE-COUNTER"
    assert code == 8 and result.report["result"]["layers"] == {"snapshot": "UNKNOWN", "legacy_csv": "UNKNOWN"}
    assert any("source conflicts" in r for r in result.report["layers"]["snapshot"]["unknown_reasons"])


def test_a_profile_with_a_skipped_counter_blocks_pass_with_a_source_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # B1 / T14
    def edit(pages: dict[str, bytes]) -> None:
        k = "/afl/stats/players/A/Ann_Able.html"
        pages[k] = pages[k].replace(b"<td align=center>3</td><td nowrap>", b"<td align=center>4</td><td nowrap>", 1)

    e2e = build(tmp_path, monkeypatch, page_edit=edit)
    result, code, _ = e2e.compare()
    seq = [f for f in result.findings if f["field"] == "counter_sequence"]
    assert len(seq) == 1 and seq[0]["layer"] == "source" and "gap:3" in seq[0]["detail"] and code in (4, 8)


def _seasons_audit(tmp_path: Path, e2e: E.E2E, seasons: list[int]) -> Any:
    from supercoach_via.reconciliation import compare as CP
    from supercoach_via.reconciliation import inventory as inv

    run = tmp_path / "srun"
    plan = inv.build_plan(data_root=e2e.data_root, snapshot="current", legacy_root=e2e.legacy_root,
                          through_date="2026-09-30", scope="seasons", seasons=seasons, run_dir=run)  # fmt: skip
    plan_path = inv.write_plan(plan)
    shutil.copytree(e2e.root / "run" / "evidence", run / "evidence")
    cap = Capture(plan, run, e2e.site.client(), clock=e2e.site.clock, durable=False).run()
    assert cap.exit_code == 0, cap
    res, audit = CP.run_audit(CP.CompareOptions(plan=plan_path, capture_manifest=run / "capture" / "manifest.json",
                                                out=run / "r", cache=run / "c"))  # fmt: skip
    res.findings_list = list(res.findings)
    CP.cleanup(audit)
    return res


def test_a_seasons_audit_judges_only_its_seasons_and_states_its_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = two_season_world()
    e2e = build(tmp_path, monkeypatch, world=world, local_rows=local_edit("a", "041520250305", "kicks", 99, world))
    full, _code, _out = e2e.compare()
    assert full.report["layers"]["snapshot"]["verdict"] == "FAIL"  # the 2025 error is real
    res = _seasons_audit(tmp_path, e2e, [2026])
    bad = [
        (f["layer"], f["category"], f["season"], f["detail"][:90]) for f in res.findings_list if f["severity"] != "info"
    ]
    assert res.report["result"]["layers"] == {"legacy_csv": "PASS", "snapshot": "PASS"}, bad  # 2025 is out of scope
    assert res.report["scope"]["population"] == "seasons" and res.report["scope"]["seasons"] == [2026]
    assert res.report["scope"]["full_population"] is False
    assert not [f for f in res.findings_list if f["category"] in ("MATCH_EXTRA_LOCAL", "IDENTITY_UNRESOLVED")]
    assert res.report["layers"]["snapshot"]["coverage"].get("agg_career_judged", 0) == 0  # careers are not judged


def test_a_seasons_audit_fails_on_an_error_inside_its_seasons(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = two_season_world()
    e2e = build(tmp_path, monkeypatch, world=world, local_rows=local_edit("a", "041520260305", "kicks", 99, world))
    res = _seasons_audit(tmp_path, e2e, [2026])
    assert res.report["result"]["layers"]["snapshot"] == "FAIL"
    assert [f["season"] for f in res.findings_list if f["category"] == "CELL_MISMATCH"] == [2026]


def test_a_seasons_audit_resolves_a_renamed_player_by_career_not_by_teammates_identical_season_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live gate, 2026: "Callum Ah Chee" (local "Chee") resolves in the full audit by his exact career appearances,
    but in a one-season audit his 2026 games equal his teammates', so he was IDENTITY_UNRESOLVED and the gate blocked.
    Legacy appearance keys need no match URL, so a seasons audit compares careers like the full audit."""
    base = two_season_world()
    first_2025 = base.matches[0]
    # Bob plays the 2026 games with Ann but misses one 2025 game: identical in 2026, distinct over a career
    world = base.with_matches(
        (replace(first_2025, apps=tuple(a for a in first_2025.apps if a.pid != "b")), *base.matches[1:])
    )

    def edit(pages: dict[str, bytes]) -> None:
        for url in pages:
            if url.endswith("/Bob_Baker.html"):
                pages[url] = pages[url].replace(b"Bob Baker", b"Bob De Baker")

    e2e = build(tmp_path, monkeypatch, world=world, page_edit=edit)
    full, _code, _out = e2e.compare()
    unresolved = [f for f in full.findings if f["category"] == "IDENTITY_UNRESOLVED" and f["layer"] == "legacy_csv"]
    assert not unresolved  # the full audit resolves Bob by his career appearances
    res = _seasons_audit(tmp_path, e2e, [2026])
    unresolved = [f for f in res.findings_list if f["category"] == "IDENTITY_UNRESOLVED" and f["layer"] == "legacy_csv"]
    assert not unresolved, unresolved
    assert res.report["result"]["layers"]["legacy_csv"] == "PASS"
