"""Contract / free-agency adapters (AFL.com.au, ZeroHanger) with honest provenance.

Both sources are often bot-blocked. When the live fetch fails, :func:`choose_payload`
may substitute a reviewed manual fixture, but only as ``manual_fixture`` with its own
source date and ``UNKNOWN`` outcome (PLAN 5.1): it is never presented as fetched today
and never yields a ``PASS`` observation.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from bs4 import BeautifulSoup

from supercoach_via.domain.schemas import CheckOutcome, Provenance, SourceMode
from supercoach_via.ingest.http import FetchResult, fit_table_row, manual_fixture_result

ADAPTER_VERSION = "1"
AFL_FA_URL = "https://www.afl.com.au/news/1484077/2026afl-free-agentslist"
ZEROHANGER_URL = "https://www.zerohanger.com/afl/players/off-contract-2026/"


@dataclass(frozen=True)
class ContractRow:
    player_name: str
    club: str
    contract_end: int
    fa_category: str | None  # restricted | unrestricted | None (not stated)
    notes: str | None


@dataclass
class ContractParse:
    outcome: CheckOutcome
    rows: list[ContractRow] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def _clean(text: str) -> str:
    return " ".join(text.replace("\xa0", " ").split()).strip()


def _soup(content: bytes | str) -> BeautifulSoup:
    return BeautifulSoup(content.decode("utf-8", "replace") if isinstance(content, bytes) else content, "html.parser")


def parse_afl_fa(content: bytes | str, *, contract_end: int) -> ContractParse:
    out = ContractParse(outcome=CheckOutcome.FAIL)
    for section in _soup(content).select("div.club-section"):
        h3 = section.find("h3")
        club = _clean(str(section.get("data-club") or "")) or _clean(h3.get_text() if h3 else "")
        for tr in section.find_all("tr"):
            player, status = tr.find("td", class_="player"), tr.find("td", class_="status")
            if player is None or status is None:
                continue
            raw_status = _clean(status.get_text()).lower()
            category = raw_status if raw_status in ("restricted", "unrestricted") else None
            if category is None:
                out.issues.append(f"unrecognised FA status {raw_status[:30]!r}")
            out.rows.append(
                ContractRow(_clean(player.get_text()), club, contract_end, category, f"{raw_status} FA (AFL.com.au)")
            )
    if not out.rows:
        out.issues.append("no free-agent rows (layout drift, block page or empty)")
    out.outcome = CheckOutcome.FAIL if out.issues else CheckOutcome.PASS
    return out


def parse_zerohanger(content: bytes | str, *, contract_end: int) -> ContractParse:
    out = ContractParse(outcome=CheckOutcome.FAIL)
    for section in _soup(content).select("div.club"):
        club = _clean(str(section.get("data-club") or ""))
        for li in section.find_all("li"):
            name = _clean(li.get_text())
            if name:
                out.rows.append(ContractRow(name, club, contract_end, None, "off-contract (ZeroHanger)"))
    if not out.rows:
        out.issues.append("no off-contract rows (layout drift, block page or empty)")
    out.outcome = CheckOutcome.FAIL if out.issues else CheckOutcome.PASS
    return out


def choose_payload(
    live: FetchResult,
    *,
    fixture: bytes | None,
    fixture_date: date | None,
    source: str,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> FetchResult:
    """Prefer a verified live payload; otherwise a labelled manual fixture; else the failure."""
    if live.ok:
        return live
    if fixture is None or fixture_date is None:
        return live
    return manual_fixture_result(live.url, fixture, source=source, source_date=fixture_date, now=now)


@dataclass
class ContractObservations:
    outcome: CheckOutcome
    rows: list[dict[str, Any]]
    source_mode: SourceMode
    issues: list[str]


def observations_from(payload: FetchResult, parsed: ContractParse | None) -> ContractObservations:
    """Build ``contract_observations`` rows; outcome is PASS only for a live, clean parse."""
    if payload.content is None or parsed is None:
        return ContractObservations(
            outcome=CheckOutcome.UNKNOWN if payload.outcome is not CheckOutcome.FAIL else CheckOutcome.FAIL,
            rows=[],
            source_mode=payload.source_mode,
            issues=[payload.error or "no payload"],
        )
    manual = payload.source_mode is SourceMode.MANUAL_FIXTURE
    observed = payload.source_date if manual else payload.fetched_at.date()
    source_type = "manual_fixture" if manual else "live"
    rows: list[dict[str, Any]] = []
    for i, r in enumerate(parsed.rows, start=1):
        key = f"{payload.url}|{r.club}|{r.player_name}|{observed}"
        row = {
            "observation_id": f"contract:{hashlib.sha256(key.encode()).hexdigest()[:20]}",
            "player_name": r.player_name,
            "player_id": None,
            "club_id": None,
            "contract_end": r.contract_end,
            "fa_category": r.fa_category,
            "observed_at": observed,
            "source_type": source_type,
            "source_name": payload.source,
            "position": None,
            "notes": re.sub(r"\s+", " ", f"{r.notes or ''}; club as sourced: {r.club}").strip("; "),
            "confidence": "low" if manual else "medium",
            "provenance": (Provenance.MANUAL_FIXTURE if manual else Provenance.SOURCE_FETCH).value,
            "source_path": payload.url,
            "source_sha256": payload.sha256,
            "source_row": i,
        }
        rows.append(fit_table_row("contract_observations", row))
    if parsed.outcome is not CheckOutcome.PASS:
        outcome = CheckOutcome.FAIL
    elif manual or not payload.ok:
        outcome = CheckOutcome.UNKNOWN
    else:
        outcome = CheckOutcome.PASS
    return ContractObservations(outcome, rows, payload.source_mode, list(parsed.issues))
