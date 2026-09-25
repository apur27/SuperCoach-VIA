"""AFL.com.au / ZeroHanger contract adapters and honest manual-fixture fallback."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from supercoach_via.domain.schemas import TABLES, CheckOutcome, SourceMode
from supercoach_via.ingest import contracts
from supercoach_via.ingest.http import FetchResult

RAW = Path(__file__).resolve().parents[1] / "fixtures" / "raw" / "contracts"
FIXTURE_DATE = date(2026, 6, 19)


def _now() -> datetime:
    return datetime(2026, 9, 24, tzinfo=UTC)


def _failed(url: str) -> FetchResult:
    return FetchResult(
        url=url, final_url=url, source="afl_com_au", outcome=CheckOutcome.FAIL,
        source_mode=SourceMode.LIVE, freshness="unknown", fetched_at=_now(), http_status=403,
        error="http 403",
    )  # fmt: skip


def test_parse_afl_fa_list() -> None:
    res = contracts.parse_afl_fa((RAW / "afl_fa_2026.html").read_bytes(), contract_end=2026)
    assert res.outcome is CheckOutcome.PASS
    borlase = res.rows[0]
    assert (borlase.player_name, borlase.club, borlase.fa_category) == ("James Borlase", "Adelaide", "restricted")
    assert len(res.rows) > 20


def test_parse_zerohanger() -> None:
    res = contracts.parse_zerohanger((RAW / "zerohanger_offcontract_2026.html").read_bytes(), contract_end=2026)
    assert res.outcome is CheckOutcome.PASS
    assert ("Aliir Aliir", "Port Adelaide") in {(r.player_name, r.club) for r in res.rows}
    assert all(r.fa_category is None for r in res.rows)


def test_live_failure_falls_back_to_labelled_manual_fixture() -> None:
    fixture = (RAW / "afl_fa_2026.html").read_bytes()
    chosen = contracts.choose_payload(
        _failed(contracts.AFL_FA_URL), fixture=fixture, fixture_date=FIXTURE_DATE, source="afl_com_au", now=_now
    )
    assert chosen.source_mode is SourceMode.MANUAL_FIXTURE
    assert chosen.outcome is CheckOutcome.UNKNOWN and chosen.source_date == FIXTURE_DATE
    obs = contracts.observations_from(chosen, contracts.parse_afl_fa(chosen.content or b"", contract_end=2026))
    assert obs.outcome is CheckOutcome.UNKNOWN  # never a verified-fresh observation
    row = obs.rows[0]
    assert set(row) == set(TABLES["contract_observations"].column_names)
    assert row["source_type"] == "manual_fixture" and row["observed_at"] == FIXTURE_DATE
    assert row["confidence"] == "low"


def test_no_fixture_and_failed_fetch_is_unknown_with_no_rows() -> None:
    chosen = contracts.choose_payload(
        _failed(contracts.AFL_FA_URL), fixture=None, fixture_date=None, source="afl_com_au", now=_now
    )
    assert chosen.outcome is not CheckOutcome.PASS and chosen.content is None
    obs = contracts.observations_from(chosen, None)
    assert obs.outcome is not CheckOutcome.PASS and obs.rows == []


def test_live_success_is_live_observation() -> None:
    body = (RAW / "zerohanger_offcontract_2026.html").read_bytes()
    live = FetchResult(
        url=contracts.ZEROHANGER_URL, final_url=contracts.ZEROHANGER_URL, source="zerohanger",
        outcome=CheckOutcome.PASS, source_mode=SourceMode.LIVE, freshness="fresh", fetched_at=_now(),
        http_status=200, content=body, sha256="b" * 64, bytes=len(body),
    )  # fmt: skip
    chosen = contracts.choose_payload(live, fixture=b"x", fixture_date=FIXTURE_DATE, source="zerohanger", now=_now)
    assert chosen is live
    obs = contracts.observations_from(chosen, contracts.parse_zerohanger(body, contract_end=2026))
    assert obs.outcome is CheckOutcome.PASS
    assert obs.rows[0]["source_type"] == "live" and obs.rows[0]["observed_at"] == date(2026, 9, 24)


def test_empty_list_is_fail() -> None:
    assert contracts.parse_zerohanger(b"<html><body></body></html>", contract_end=2026).outcome is CheckOutcome.FAIL
