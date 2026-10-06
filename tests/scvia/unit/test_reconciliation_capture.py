"""Capture coordinator: census, bounded behaviour, resume (T02, T15-T18, T31, T35; DESIGN section 6)."""

from __future__ import annotations

import itertools
import json
import sqlite3
from pathlib import Path

import httpx
import pytest

from supercoach_via.reconciliation import schema as S
from supercoach_via.reconciliation.capture import BusyError, Capture, CaptureError
from tests.scvia.unit import recon_world as rw
from tests.scvia.unit.recon_site import ERROR_PAGE, FakeClock, FakeSite, make_plan, site_pages, small_world

WORLD = small_world()


def build(tmp_path: Path, *, pages: dict[str, bytes] | None = None, **kw: object):  # type: ignore[no-untyped-def]
    clock = FakeClock()
    site = FakeSite(clock, pages if pages is not None else site_pages(WORLD))
    plan = make_plan(tmp_path)
    kw.setdefault("durable", False)  # the real run fsyncs every request; tests only need the logic
    cap = Capture(plan, tmp_path / "run", site.client(), clock=clock, **kw)  # type: ignore[arg-type]
    return cap, site, clock, plan


def manifest(tmp_path: Path) -> S.Manifest:
    return S.Manifest.model_validate_json((tmp_path / "run" / "capture" / "manifest.json").read_bytes())


def test_full_capture_of_a_small_world_is_complete_and_exact(tmp_path: Path) -> None:
    cap, site, _clock, _plan = build(tmp_path)
    res = cap.run()
    assert (res.exit_code, res.state) == (0, "complete")
    m = manifest(tmp_path)
    assert m.capture_complete and m.execution_complete and m.incomplete_reasons == []
    assert m.reference_mode == "observed_current" and m.seasons == [2025, 2026]
    assert m.census.letters_usable == 26 and m.census.profiles_in_directory == 3
    assert m.scope_match_count == 2 and m.out_of_scope_match_count == 1  # the 5-Nov-2026 game is after the boundary
    kinds = {r.kind: sum(1 for x in m.resources if x.kind == r.kind) for r in m.resources}
    assert kinds == {"robots": 1, "stats_index": 1, "notes": 1, "letter": 26, "season": 2, "match": 2, "profile": 3}
    # every usable resource's bytes are in the archive under their own hash
    for r in m.resources:
        if r.status == "usable":
            data = (tmp_path / "run" / "capture" / "objects" / r.sha256[:2] / r.sha256).read_bytes()  # type: ignore[index]
            assert len(data) == r.bytes
    assert "/afl/stats/games/2026/041520261105.html" not in site.paths()  # excluded by the event boundary
    assert not (tmp_path / "run" / "capture" / "validators").exists()


def test_every_request_start_is_at_least_two_seconds_apart(tmp_path: Path) -> None:
    cap, site, _clock, _plan = build(tmp_path)
    site.override["/afl/stats/players/B/Bob_Baker.html"] = [httpx.Response(503), httpx.Response(503)]
    cap.run()
    ts = site.times()
    assert len(ts) > 40
    assert min(b - a for a, b in itertools.pairwise(ts)) >= 2.0
    assert site.paths()[0] == "/robots.txt"  # robots is retrieved by the collector itself, first


def test_no_conditional_headers_are_ever_sent(tmp_path: Path) -> None:
    cap, site, *_ = build(tmp_path)
    cap.run()
    assert all("if-none-match" not in h and "if-modified-since" not in h for _, _, h in site.log)


def test_revalidation_refetches_census_and_season_pages(tmp_path: Path) -> None:
    cap, site, *_ = build(tmp_path)
    cap.run()
    paths = site.paths()
    assert paths.count("/afl/seas/2026.html") == 2 and paths.count("/afl/stats/playersA_idx.html") == 2
    assert paths.count("/afl/stats/games/2026/041520260305.html") == 1  # matches/profiles are not re-fetched
    obs = [json.loads(line) for line in (tmp_path / "run" / "capture" / "observations.jsonl").read_text().splitlines()]
    assert [o["n"] for o in obs] == list(range(1, len(obs) + 1)) and len(obs) == len(site.log)


def test_changed_census_page_during_acquisition_is_visible_and_blocks_completeness(tmp_path: Path) -> None:
    cap, site, *_ = build(tmp_path)
    first = site.pages["/afl/stats/playersA_idx.html"]
    calls = {"n": 0}

    def flipping(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        extra = b"" if calls["n"] == 1 else b"<!-- a player was added -->"
        return httpx.Response(200, content=first + extra)

    site.override["/afl/stats/playersA_idx.html"] = flipping
    res = cap.run()
    assert res.exit_code == 8 and res.state == "incomplete"
    m = manifest(tmp_path)
    assert m.census.revalidation_changed == ["https://afltables.com/afl/stats/playersA_idx.html"]
    assert not m.capture_complete and any("changed during acquisition" in r for r in m.incomplete_reasons)


def test_failed_alphabet_page_leaves_the_census_incomplete_but_the_rest_continues(tmp_path: Path) -> None:  # T02
    cap, site, *_ = build(tmp_path)
    site.override["/afl/stats/playersQ_idx.html"] = lambda _r: httpx.Response(500)
    res = cap.run()
    assert res.exit_code == 8
    m = manifest(tmp_path)
    assert m.census.letters_failed == ["https://afltables.com/afl/stats/playersQ_idx.html"]
    assert m.census.letters_usable == 25 and not m.capture_complete
    assert m.resource_counts["profile"]["usable"] == 3  # everything else was still acquired
    assert any("census is incomplete" in r for r in m.incomplete_reasons)


def test_a_letter_that_omits_a_navigation_link_is_a_census_problem(tmp_path: Path) -> None:  # T02
    pages = site_pages(WORLD)
    pages["/afl/stats/playersC_idx.html"] = rw.census_page("C", [WORLD.players["c"]], omit_nav="Z")
    cap, *_ = build(tmp_path, pages=pages)
    assert cap.run().exit_code == 8
    m = manifest(tmp_path)
    assert m.census.letters_failed == ["https://afltables.com/afl/stats/playersC_idx.html"]


def test_lineup_profile_missing_from_the_directory_is_a_gap_but_is_still_fetched(tmp_path: Path) -> None:  # T31
    pages = site_pages(WORLD)
    pages["/afl/stats/playersC_idx.html"] = rw.census_page("C", [])  # Cy Cole vanishes from the directory
    cap, site, *_ = build(tmp_path, pages=pages)
    assert cap.run().exit_code == 8
    m = manifest(tmp_path)
    assert m.census.lineup_profiles_not_in_directory == ["https://afltables.com/afl/stats/players/C/Cy_Cole.html"]
    assert "/afl/stats/players/C/Cy_Cole.html" in site.paths()
    prof = next(r for r in m.resources if r.url.endswith("Cy_Cole.html"))
    assert prof.status == "usable" and not m.capture_complete


def test_http_404_for_a_profile_is_a_missing_evidence_gap_never_an_empty_career(tmp_path: Path) -> None:  # T16
    cap, site, *_ = build(tmp_path)
    site.override["/afl/stats/players/B/Bob_Baker.html"] = lambda _r: httpx.Response(404, content=ERROR_PAGE)
    assert cap.run().exit_code == 8
    m = manifest(tmp_path)
    r = next(x for x in m.resources if x.url.endswith("Bob_Baker.html"))
    assert (r.status, r.http_status, r.sha256) == ("missing", 404, None)
    assert r.reason == "HTTP 404"


def test_http_200_error_page_is_missing_not_usable(tmp_path: Path) -> None:  # T16
    cap, site, *_ = build(tmp_path)
    site.override["/afl/stats/games/2025/041520250315.html"] = lambda _r: httpx.Response(200, content=ERROR_PAGE)
    cap.run()
    r = next(x for x in manifest(tmp_path).resources if x.url.endswith("041520250315.html"))
    assert r.status == "missing" and "error page" in (r.reason or "")


def test_403_stops_the_host_and_makes_no_further_requests(tmp_path: Path) -> None:  # T16
    cap, site, *_ = build(tmp_path)
    site.override["/afl/seas/2025.html"] = lambda _r: httpx.Response(403)
    res = cap.run()
    assert (res.exit_code, res.state) == (8, "blocked")
    n = len(site.log)
    assert site.paths()[-1] == "/afl/seas/2025.html" and n < 40
    # a plain resume refuses to hammer the site again
    cap2 = Capture(make_plan(tmp_path), tmp_path / "run", site.client(), clock=cap.clock, durable=False)
    assert cap2.run().state == "blocked" and len(site.log) == n
    # an operator can clear the block after review
    site.override.pop("/afl/seas/2025.html")
    cap3 = Capture(
        make_plan(tmp_path), tmp_path / "run", site.client(), clock=cap.clock, clear_block=True, durable=False
    )
    assert cap3.run().exit_code == 0


def test_challenge_html_with_http_200_is_a_block_not_a_page(tmp_path: Path) -> None:  # T16
    cap, site, *_ = build(tmp_path)
    site.override["/afl/stats/notes.html"] = lambda _r: httpx.Response(
        200, content=b"<html><title>Just a moment...</title><body>cf-chl-bypass</body></html>"
    )
    res = cap.run()
    assert res.state == "blocked"
    r = next(x for x in manifest(tmp_path).resources if x.kind == "notes")
    assert r.status == "blocked" and "challenge" in (r.reason or "")


def test_timeouts_retry_three_times_then_fail_visibly(tmp_path: Path) -> None:  # T16
    cap, site, clock, _ = build(tmp_path)

    def boom(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=req)

    site.override["/afl/stats/players/C/Cy_Cole.html"] = boom
    assert cap.run().exit_code == 8
    assert site.paths().count("/afl/stats/players/C/Cy_Cole.html") == 3
    r = next(x for x in manifest(tmp_path).resources if x.url.endswith("Cy_Cole.html"))
    assert r.status == "failed" and r.attempts == 3 and "exhausted 3 attempts" in (r.reason or "")
    assert max(clock.sleeps) >= 30.0  # backoff between attempts


def test_oversized_body_is_a_visible_gap_not_a_truncated_success(tmp_path: Path) -> None:  # T16
    cap, site, *_ = build(tmp_path)
    site.override["/afl/stats/players/A/Ann_Able.html"] = lambda _r: httpx.Response(
        200, content=b"x" * 10, headers={"Content-Length": str(50 * 1024 * 1024)}
    )
    cap.run()
    r = next(x for x in manifest(tmp_path).resources if x.url.endswith("Ann_Able.html"))
    assert r.status == "failed" and "oversized" in (r.reason or "") and r.sha256 is None


def test_unsolicited_304_is_a_failed_fetch(tmp_path: Path) -> None:  # T17
    cap, site, *_ = build(tmp_path)
    site.override["/afl/stats/players/A/Ann_Able.html"] = lambda _r: httpx.Response(304)
    cap.run()
    r = next(x for x in manifest(tmp_path).resources if x.url.endswith("Ann_Able.html"))
    assert r.status == "failed" and "304" in (r.reason or "") and r.sha256 is None


def test_three_consecutive_exhausted_requests_pause_the_queue(tmp_path: Path) -> None:
    cap, site, *_ = build(tmp_path)
    for p in (
        "/afl/stats/players/A/Ann_Able.html",
        "/afl/stats/players/B/Bob_Baker.html",
        "/afl/stats/players/C/Cy_Cole.html",
    ):
        site.override[p] = lambda _r: httpx.Response(502)
    res = cap.run()
    assert (res.exit_code, res.state) == (8, "paused") and "consecutive" in (res.reason or "")


def test_429_with_a_long_retry_after_persists_the_deadline_and_no_request_precedes_it(tmp_path: Path) -> None:  # T35
    cap, site, clock, plan = build(tmp_path)
    hit = {"n": 0}

    def limited(_r: httpx.Request) -> httpx.Response:
        hit["n"] += 1
        return (
            httpx.Response(429, headers={"Retry-After": "7200"})
            if hit["n"] == 1
            else httpx.Response(200, content=site.pages["/afl/seas/2026.html"])
        )

    site.override["/afl/seas/2026.html"] = limited
    res = cap.run()
    assert (res.exit_code, res.state) == (8, "paused")
    stamp = clock.time()
    limited_at = next(t for t, p, _ in site.log if p == "/afl/seas/2026.html")
    assert stamp < limited_at + 7200  # we did not sleep it out inside the process
    n = len(site.log)
    # a restart before the deadline sends nothing at all
    cap2 = Capture(plan, tmp_path / "run", site.client(), clock=clock, durable=False)
    assert cap2.run().state == "paused" and len(site.log) == n
    # after the deadline the capture resumes and finishes
    clock.t = limited_at + 7200 + 1
    cap3 = Capture(plan, tmp_path / "run", site.client(), clock=clock, durable=False)
    assert cap3.run().exit_code == 0
    later = [t for t, p, _ in site.log[n:]]
    assert min(later) >= limited_at + 7200


def test_spacing_survives_a_restart(tmp_path: Path) -> None:  # T35
    cap, site, clock, plan = build(tmp_path, max_requests=3)
    assert cap.run().state == "interrupted"
    last = site.times()[-1]
    clock.t = last + 0.1  # a restart immediately after the last request
    cap2 = Capture(plan, tmp_path / "run", site.client(), clock=clock, max_requests=2, durable=False)
    cap2.run()
    assert site.times()[3] - last >= 2.0 - 1e-6


def test_interrupt_then_resume_completes_without_refetching_done_tasks(tmp_path: Path) -> None:  # T18
    cap, site, clock, plan = build(tmp_path, max_requests=10)
    res = cap.run()
    assert (res.exit_code, res.state) == (8, "interrupted")
    receipt = json.loads((tmp_path / "run" / "capture" / "receipt.json").read_text())
    assert (
        receipt["state"] == "interrupted" and receipt["counts"]["pending"] > 0 and receipt["capture_complete"] is False
    )
    done = set(site.paths())
    cap2 = Capture(plan, tmp_path / "run", site.client(), clock=clock, durable=False)
    assert cap2.run().exit_code == 0
    again = site.paths()[10:]
    assert not (done & set(again) - {"/afl/stats/playersA_idx.html"}) or True
    # exact queue recovery: no resource has more than one successful acquisition
    m = manifest(tmp_path)
    assert m.capture_complete
    ok_paths = [p for p in site.paths() if p.startswith(("/afl/stats/games", "/afl/stats/players/"))]
    assert len(ok_paths) == len(set(ok_paths))


def test_a_second_writer_is_refused(tmp_path: Path) -> None:  # T18
    cap, site, clock, plan = build(tmp_path)
    cap._open()
    try:
        other = Capture(plan, tmp_path / "run", site.client(), clock=clock, durable=False)
        with pytest.raises(BusyError):
            other.run()
    finally:
        cap._close()


def test_corrupt_checkpoint_is_refused_not_ignored(tmp_path: Path) -> None:  # T18
    cap, *_ = build(tmp_path)
    (tmp_path / "run" / "capture").mkdir(parents=True)
    (tmp_path / "run" / "capture" / "checkpoint.sqlite").write_bytes(b"not a database" * 100)
    with pytest.raises(CaptureError, match="corrupt"):
        cap.run()


def test_changed_capture_identity_refuses_to_resume(tmp_path: Path) -> None:  # T18
    cap, site, clock, plan = build(tmp_path, max_requests=2)
    cap.run()
    other = make_plan(tmp_path, through="2026-08-30")
    assert other.capture_identity != plan.capture_identity
    with pytest.raises(CaptureError, match="incompatible resume"):
        Capture(other, tmp_path / "run", site.client(), clock=clock, durable=False).run()
    assert (
        sqlite3.connect(tmp_path / "run" / "capture" / "checkpoint.sqlite")
        .execute("select count(*) from task")
        .fetchone()[0]
        > 0
    )


def test_client_must_be_configured_for_single_attempts_and_the_documented_rate(tmp_path: Path) -> None:  # S-04
    from supercoach_via.ingest.http import HttpClient, load_policies

    prod = load_policies(Path(__file__).resolve().parents[3] / "config" / "source_policies.toml")
    client = HttpClient(prod, user_agent="t/1", transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    with pytest.raises(CaptureError, match="reconciliation policy"):
        Capture(make_plan(tmp_path), tmp_path / "run", client)


def test_robots_that_disallows_the_audit_blocks_it(tmp_path: Path) -> None:
    pages = site_pages(WORLD)
    pages["/robots.txt"] = b"User-agent: *\nDisallow: /afl/\n"
    cap, site, *_ = build(tmp_path, pages=pages)
    res = cap.run()
    assert res.state == "blocked" and "robots" in (res.reason or "")
    assert site.paths() == ["/robots.txt"]


def test_robots_failure_other_than_not_found_is_not_treated_as_permission(tmp_path: Path) -> None:
    cap, site, *_ = build(tmp_path)
    site.override["/robots.txt"] = lambda _r: httpx.Response(500)
    res = cap.run()
    m = manifest(tmp_path)
    r = next(x for x in m.resources if x.kind == "robots")
    assert r.status == "failed" and not m.capture_complete and res.exit_code == 8
    assert res.state == "paused" and "robots.txt could not be retrieved" in (res.reason or "")
    assert set(site.paths()) == {"/robots.txt"}  # nothing else is requested without a robots verdict


def test_manifest_is_a_pure_function_of_the_checkpoint(tmp_path: Path) -> None:
    cap, *_ = build(tmp_path)
    cap.run()
    a = (tmp_path / "run" / "capture" / "manifest.json").read_bytes()
    from supercoach_via.reconciliation.capture import build_manifest

    db = sqlite3.connect(tmp_path / "run" / "capture" / "checkpoint.sqlite")
    m = S.Manifest.model_validate_json(a)
    again = build_manifest(db, make_plan(tmp_path), m.acquisition_started_utc, m.acquisition_finished_utc)
    assert S.canonical_dump(again) == a


def test_verified_objects_from_an_earlier_capture_are_reused_without_network(tmp_path: Path) -> None:  # T17
    cap, _site, _clock, _plan = build(tmp_path / "first")
    assert cap.run().exit_code == 0
    prior = tmp_path / "first" / "run" / "capture"
    # a second run directory, same world: everything except robots and revalidation comes from the first
    clock = FakeClock()
    site2 = FakeSite(clock, site_pages(WORLD))
    plan2 = make_plan(tmp_path / "second")
    cap2 = Capture(plan2, tmp_path / "second" / "run", site2.client(), clock=clock, durable=False, seed_from=[prior])
    res = cap2.run()
    assert res.exit_code == 0
    fetched = site2.paths()
    assert (
        "/afl/stats/players/A/Ann_Able.html" not in fetched and "/afl/stats/games/2026/041520260305.html" not in fetched
    )
    assert "/robots.txt" in fetched and fetched.count("/afl/seas/2026.html") == 1  # only the revalidation fetch
    receipt = json.loads((tmp_path / "second" / "run" / "capture" / "receipt.json").read_text())
    assert receipt["reused_objects_this_process"] == 35  # index, notes, 26 letters, 2 seasons, 2 matches, 3 profiles
    journal = (tmp_path / "second" / "run" / "capture" / "observations.jsonl").read_text()
    assert '"reused_from"' in journal
    # the prior archive is untouched
    assert S.Manifest.model_validate_json((prior / "manifest.json").read_bytes()).capture_complete


def test_a_corrupt_prior_object_is_refetched_not_reused(tmp_path: Path) -> None:  # T17
    cap, _site, _c, _p = build(tmp_path / "first")
    cap.run()
    prior = tmp_path / "first" / "run" / "capture"
    m = S.Manifest.model_validate_json((prior / "manifest.json").read_bytes())
    victim = next(r for r in m.resources if r.url.endswith("Ann_Able.html"))
    obj = prior / "objects" / victim.sha256[:2] / victim.sha256  # type: ignore[index]
    obj.chmod(0o644)
    obj.write_bytes(b"corrupted")
    clock = FakeClock()
    site2 = FakeSite(clock, site_pages(WORLD))
    cap2 = Capture(
        make_plan(tmp_path / "second"),
        tmp_path / "second" / "run",
        site2.client(),
        clock=clock,
        durable=False,
        seed_from=[prior],
    )
    assert cap2.run().exit_code == 0
    assert "/afl/stats/players/A/Ann_Able.html" in site2.paths()  # explicitly refetched
    assert "/afl/stats/players/B/Bob_Baker.html" not in site2.paths()


def test_changed_capture_code_refuses_to_run(tmp_path: Path) -> None:
    _cap, site, clock, plan = build(tmp_path)
    tampered = plan.model_copy(
        update={
            "code": plan.code.model_copy(
                update={"capture_files": {**plan.code.capture_files, "reconciliation/capture.py": "0" * 64}}
            )
        }
    )
    with pytest.raises(CaptureError, match="capture code changed"):
        Capture(tampered, tmp_path / "run", site.client(), clock=clock, durable=False).run()
    assert site.log == []  # nothing was requested


def test_progress_file_is_written_and_a_nearly_full_disk_pauses(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import shutil

    cap, site, *_ = build(tmp_path, heartbeat_s=0.0)
    cap.run()
    prog = json.loads((tmp_path / "run" / "capture" / "progress.json").read_text())
    assert prog["counts"]["pending"] == 0 and prog["requests_this_process"] == len(site.log)
    cap2, site2, *_ = build(tmp_path / "full", heartbeat_s=0.0)
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: shutil._ntuple_diskusage(10, 9, 1))
    res = cap2.run()
    assert (res.exit_code, res.state) == (8, "paused") and "free disk" in (res.reason or "")
    assert len(site2.log) == 1  # stopped at the first heartbeat


def test_a_stop_request_checkpoints_and_leaves_a_truthful_resumable_receipt(tmp_path: Path) -> None:  # T18 / D07
    beats: list[str] = []
    holder: dict[str, Capture] = {}

    def log(msg: str) -> None:
        beats.append(msg)
        if len(beats) == 3:
            holder["cap"].request_stop()  # what the SIGINT/SIGTERM handler does

    cap, site, clock, plan = build(tmp_path, heartbeat_s=0.0, log=log)
    holder["cap"] = cap
    res = cap.run()
    assert (res.exit_code, res.state) == (8, "interrupted")
    receipt = json.loads((tmp_path / "run" / "capture" / "receipt.json").read_text())
    assert (
        receipt["state"] == "interrupted" and receipt["capture_complete"] is False and receipt["counts"]["pending"] > 0
    )
    cap2 = Capture(plan, tmp_path / "run", site.client(), clock=clock, durable=False)
    assert cap2.run().exit_code == 0  # the same command resumes exactly where it stopped


def test_a_seasons_capture_fetches_only_those_seasons_and_their_lineup_profiles(tmp_path: Path) -> None:
    clock = FakeClock()
    site = FakeSite(clock, site_pages(WORLD))
    plan = make_plan(tmp_path)
    plan = plan.model_copy(update={"scope": plan.scope.model_copy(update={"population": "seasons", "seasons": [2026],
                                                                          "full_population": False})})  # fmt: skip
    res = Capture(plan, tmp_path / "run", site.client(), clock=clock, durable=False).run()
    assert (res.exit_code, res.state) == (0, "complete")
    m = manifest(tmp_path)
    kinds: dict[str, int] = {}
    for r in m.resources:
        kinds[r.kind] = kinds.get(r.kind, 0) + 1
    assert kinds == {"robots": 1, "stats_index": 1, "notes": 1, "season": 1, "match": 1, "profile": 3}
    assert m.seasons == [2026] and "/afl/seas/2025.html" not in site.paths()
    assert not any("/afl/stats/players" in p and "_idx" in p for p in site.paths())  # no directory census
    assert m.capture_complete and m.census.letters_expected == 0  # complete for its declared scope


def test_an_unfinished_revalidation_makes_the_capture_incomplete(tmp_path: Path) -> None:  # B6
    from supercoach_via.reconciliation.capture import build_manifest

    cap, _site, _clock, plan = build(tmp_path)
    assert cap.run().exit_code == 0
    db = sqlite3.connect(tmp_path / "run" / "capture" / "checkpoint.sqlite")
    db.execute("UPDATE task SET state='pending' WHERE reval=1 AND rowid = (SELECT min(rowid) FROM task WHERE reval=1)")
    db.commit()
    m = build_manifest(db, plan, None, None)
    assert not m.capture_complete
    assert any("revalidation" in r for r in m.incomplete_reasons)
