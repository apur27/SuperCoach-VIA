"""Deterministic numeric verification and bound review verdicts (PLAN 10; AUDIT S03).

Draft grammar: a number may appear only as ``{{claim:ID}}`` (rendered from the claim's
value) or as a literal immediately followed by ``{{ref:ID}}`` whose claim value it must
equal at the literal's precision. Any other number -- including one carrying a
``[data]`` tag -- is an invented number and fails verification.

A review verdict is bound to the content, packet (snapshot + claims) and policy hashes
and applies only while all three match. ``PASS_WITH_CONCERNS`` is an editorial verdict,
never numeric certification: ``numeric_certified`` comes only from the deterministic
check. :func:`run_editorial` never raises; any failure yields an unpublished draft and
leaves the numeric release untouched.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, ConfigDict

from supercoach_via.domain.schemas import CheckOutcome, ReviewVerdict
from supercoach_via.editorial.adapter import AdapterError, DraftResult, EditorialAdapter
from supercoach_via.editorial.evidence import EvidencePacket

_CLAIM_RE = re.compile(r"\{\{claim:([^{}\s]+)\}\}")
_REF_RE = re.compile(r"([-+]?\d[\d,]*(?:\.\d+)?)(%?)\s*\{\{ref:([^{}\s]+)\}\}")
_NUMBER_RE = re.compile(r"(?<![A-Za-z_])[-+]?\d[\d,]*(?:\.\d+)?")


def _sha(text: str | bytes) -> str:
    return hashlib.sha256(text if isinstance(text, bytes) else text.encode()).hexdigest()


def format_value(value: int | float | str | None) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool | str):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return f"{value:.2f}".rstrip("0").rstrip(".")


class NumericCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: CheckOutcome
    numbers_checked: int
    violations: tuple[str, ...] = ()


def _literal_matches(literal: str, value: int | float | str | None) -> bool:
    if value is None or isinstance(value, str):
        return False
    try:
        lit = Decimal(literal.replace(",", ""))
    except InvalidOperation:
        return False
    exponent = lit.as_tuple().exponent
    places = -exponent if isinstance(exponent, int) and exponent < 0 else 0
    return bool(abs(Decimal(str(value)) - lit) <= Decimal(5) * Decimal(10) ** -(places + 1))


def verify_draft(markdown: str, packet: EvidencePacket) -> NumericCheck:
    violations: list[str] = []
    checked = 0
    for m in _CLAIM_RE.finditer(markdown):
        checked += 1
        if packet.claim(m.group(1)) is None:
            violations.append(f"unknown claim {m.group(1)!r}")
    rest = _CLAIM_RE.sub(" ", markdown)
    for m in _REF_RE.finditer(rest):
        checked += 1
        claim = packet.claim(m.group(3))
        if claim is None:
            violations.append(f"unknown claim {m.group(3)!r}")
        elif not _literal_matches(m.group(1), claim.value):
            violations.append(f"{m.group(1)} does not match claim {claim.claim_id} = {format_value(claim.value)}")
    rest = _REF_RE.sub(" ", rest)
    for m in _NUMBER_RE.finditer(rest):
        checked += 1
        violations.append(f"unreferenced number {m.group(0)!r} (a [data] tag is not evidence)")
    return NumericCheck(
        outcome=CheckOutcome.FAIL if violations else CheckOutcome.PASS,
        numbers_checked=checked,
        violations=tuple(violations[:100]),
    )


def render_draft(markdown: str, packet: EvidencePacket) -> str:
    """Replace claim references with deterministic values; ``{{ref:..}}`` markers are dropped."""

    def claim(m: re.Match[str]) -> str:
        c = packet.claim(m.group(1))
        if c is None:
            raise ValueError(f"unknown claim {m.group(1)!r}")
        return format_value(c.value)

    out = _CLAIM_RE.sub(claim, markdown)
    return re.sub(r"\s*\{\{ref:[^{}\s]+\}\}", "", out)


class VerdictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: ReviewVerdict
    content_sha256: str
    packet_sha256: str
    snapshot_id: str
    policy_sha256: str
    reviewer: str
    reasons: tuple[str, ...] = ()
    numeric_certified: bool


def bind_verdict(
    verdict: ReviewVerdict,
    *,
    content: str,
    packet: EvidencePacket,
    policy_text: str,
    reviewer: str,
    reasons: list[str] | tuple[str, ...] = (),
    numeric: NumericCheck,
) -> VerdictRecord:
    return VerdictRecord(
        verdict=verdict,
        content_sha256=_sha(content),
        packet_sha256=packet.sha256,
        snapshot_id=packet.snapshot_id,
        policy_sha256=_sha(policy_text),
        reviewer=reviewer,
        reasons=tuple(str(r)[:300] for r in reasons),
        numeric_certified=numeric.outcome is CheckOutcome.PASS,
    )


def verdict_applies(record: VerdictRecord, *, content: str, packet: EvidencePacket, policy_text: str) -> bool:
    return (
        record.content_sha256 == _sha(content)
        and record.packet_sha256 == packet.sha256
        and record.snapshot_id == packet.snapshot_id
        and record.policy_sha256 == _sha(policy_text)
    )


@dataclass
class EditorialOutcome:
    status: Literal["publishable", "unpublished_draft"]
    draft_markdown: str
    rendered_markdown: str | None
    numeric: NumericCheck
    verdict: VerdictRecord | None
    reasons: list[str] = field(default_factory=list)


def run_editorial(adapter: EditorialAdapter, packet: EvidencePacket, *, policy_text: str) -> EditorialOutcome:
    """Draft + review + deterministic checks. Never raises; failures stay unpublished."""
    empty = NumericCheck(outcome=CheckOutcome.UNKNOWN, numbers_checked=0)
    try:
        draft: DraftResult = adapter.draft(packet)
    except (AdapterError, ValueError) as exc:
        return EditorialOutcome("unpublished_draft", "", None, empty, None, [f"draft failed: {exc}"])
    numeric = verify_draft(draft.markdown, packet)
    reasons = list(numeric.violations)
    try:
        review = adapter.review(packet, draft)
        verdict = bind_verdict(review.verdict, content=draft.markdown, packet=packet, policy_text=policy_text,
                               reviewer=review.reviewer, reasons=review.reasons, numeric=numeric)  # fmt: skip
    except (AdapterError, ValueError) as exc:
        verdict = bind_verdict(ReviewVerdict.UNKNOWN, content=draft.markdown, packet=packet, policy_text=policy_text,
                               reviewer=adapter.name, reasons=[f"review failed: {exc}"], numeric=numeric)  # fmt: skip
    reasons += list(verdict.reasons)
    ok = numeric.outcome is CheckOutcome.PASS and verdict.verdict in (
        ReviewVerdict.PASS,
        ReviewVerdict.PASS_WITH_CONCERNS,
    )
    rendered = render_draft(draft.markdown, packet) if numeric.outcome is CheckOutcome.PASS else None
    return EditorialOutcome("publishable" if ok else "unpublished_draft", draft.markdown, rendered, numeric,
                            verdict, reasons)  # fmt: skip
