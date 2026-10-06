"""Finding construction: stable ids, severities and the evidence/locator shape (DESIGN section 9).

A finding id is a hash of its rule, layer, entity, field and evidence identity: re-running on the
same inputs produces the same ids, and nothing here is random. Locators let a reviewer open the
exact evidence (source URL + body hash + table/row/column; local fragment or file/row).
"""

from __future__ import annotations

import hashlib
from typing import Any

from supercoach_via.integrity.report import canonical_bytes

#: category -> (severity, one-line investigation hint)
CATEGORIES: dict[str, tuple[str, str]] = {
    # confirmed discrepancies in a requested local layer
    "PLAYER_MISSING_LOCAL": ("fail", "the source lists a player with in-scope appearances that the local layer lacks"),
    "APPEARANCE_MISSING_LOCAL": ("fail", "the source has this appearance; the local layer has no accepted row for it"),
    "APPEARANCE_QUARANTINED": (
        "fail",
        "the source appearance exists only as a quarantined local row, not an accepted one",
    ),
    "APPEARANCE_EXTRA_LOCAL": ("fail", "the local layer has an appearance the source does not show for this player"),
    "APPEARANCE_DUPLICATE_LOCAL": ("fail", "more than one local row represents one source appearance"),
    "MATCH_LINK_MISMATCH": (
        "fail",
        "a local row is attached to a different match than the source row it corresponds to",
    ),
    "MATCH_MISSING_LOCAL": ("fail", "an in-scope source match has no local match"),
    "MATCH_EXTRA_LOCAL": ("fail", "a local match inside the audited scope has no source match"),
    "MATCH_ATTR_MISMATCH": ("fail", "a local match attribute (date, stage, teams, score) differs from the source"),
    "APPEARANCE_ATTR_MISMATCH": (
        "fail",
        "a local appearance attribute (club, opponent, result, jersey, counter, date) differs",
    ),
    "APPEARANCE_DATE_MISMATCH": (
        "fail",
        "local rows store a match date that differs from the source match date (grouped per player, season and declared"
        " date quality; local.rows is the exact appearance count)",
    ),
    "CELL_MISMATCH": ("fail", "a local statistic differs from the source value"),
    "CELL_LOCAL_NULL": ("fail", "the source proves a recorded zero; the local value is null"),
    "LOCAL_CELL_MALFORMED": ("fail", "a local cell is not a valid number"),
    "AGGREGATE_MISMATCH": ("fail", "a local season/stint/career total differs from the source reference"),
    "LOCAL_MISSING_SUMMARY_VALUE": (
        "fail",
        "the source publishes a season-summary value the local layer cannot reproduce",
    ),
    # evidence missing, conflicting or unresolved
    "SOURCE_CONFLICT": ("unknown", "two source facts that index the same appearance disagree; open both"),
    "CELL_UNRESOLVED": ("unknown", "the source cell cannot be resolved to a value, zero or unavailable"),
    "LOCAL_UNSUPPORTED_NUMERIC": ("unknown", "the local layer stores a number where the source records none"),
    "IDENTITY_UNRESOLVED": ("unknown", "no identity rule mapped this player uniquely"),
    "IDENTITY_CONFLICT": ("unknown", "identity evidence conflicts or two local players claim one profile"),
    "CAPTURE_GAP": ("unknown", "required source evidence is missing, failed, blocked or rejected"),
    "SCHEMA_GAP": ("unknown", "a source page has a column, shape or label this audit does not support"),
    "LOCAL_INPUT_GAP": ("unknown", "a local input could not be read as pinned"),
    "MATCH_UNRESOLVED": ("unknown", "a local and source match could not be paired without guessing"),
    # reported, never changes a layer verdict
    "SOURCE_DERIVED_INCONSISTENCY": ("info", "a printed average/total disagrees with the source's own game cells"),
    "IDENTITY_VARIANCE": ("info", "identity resolved although names differ"),
    "REPRESENTATION": ("info", "null versus zero for a did-not-take-the-field cell"),
}


def severity_of(category: str) -> str:
    return CATEGORIES[category][0]


def finding_id(parts: list[Any]) -> str:
    return hashlib.sha256(canonical_bytes(parts)).hexdigest()[:24]


def make_finding(
    category: str,
    *,
    layer: str,
    rule_id: str,
    player: dict[str, Any] | None = None,
    season: int | None = None,
    match: dict[str, Any] | None = None,
    field: str | None = None,
    expected: Any = None,
    actual: Any = None,
    expected_raw: Any = None,
    actual_raw: Any = None,
    evidence: dict[str, Any] | None = None,
    local: dict[str, Any] | None = None,
    detail: str = "",
    extra_id: Any = None,
) -> dict[str, Any]:
    """One finding as a plain, JSON-ready dict with a deterministic id and the category's investigation hint."""
    sev, hint = CATEGORIES[category]
    player = player or {}
    match = match or {}
    fid = finding_id(
        [
            category,
            layer,
            rule_id,
            player.get("source_url"),
            player.get("local_id"),
            season,
            match.get("source_url"),
            match.get("local_id"),
            field,
            (local or {}).get("origin"),
            (evidence or {}).get("locator"),
            extra_id,
        ]
    )
    return {
        "id": fid,
        "category": category,
        "severity": sev,
        "layer": layer,
        "rule_id": rule_id,
        "player": player,
        "season": season,
        "match": match,
        "field": field,
        "expected": expected,
        "actual": actual,
        "expected_raw": expected_raw,
        "actual_raw": actual_raw,
        "evidence": evidence or {},
        "local": local or {},
        "detail": detail,
        "suggestion": hint,
    }


def sort_key(f: dict[str, Any]) -> tuple[Any, ...]:
    return (
        f["layer"],
        f["category"],
        f["player"].get("source_url") or f["player"].get("local_id") or "",
        f["season"] if f["season"] is not None else -1,
        f["match"].get("source_url") or f["match"].get("local_id") or "",
        f["field"] or "",
        f["id"],
    )
