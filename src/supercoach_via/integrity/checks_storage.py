"""Family A: stored bytes and contracts (pointer, manifest, fragments, keys, values)."""

from __future__ import annotations

from supercoach_via.domain.schemas import (
    TABLES,
    BirthDateQuality,
    CheckOutcome,
    DatePrecision,
    DateQuality,
    IdentityStatus,
    MatchStatus,
    Origin,
    Provenance,
    Severity,
    SourceMode,
    StageType,
)
from supercoach_via.integrity.capture import strict_json
from supercoach_via.integrity.context import AuditContext, CheckSkipped, CheckSpec, rule
from supercoach_via.integrity.report import Status

B = Severity.BLOCKING
CORE_TABLES = ("players", "clubs", "matches", "player_games", "seasons", "quality_issues", "quarantine", "source_files")

RULES = [
    rule(
        "storage.pointer",
        "storage.identity",
        B,
        "current.json is missing, invalid or names another manifest",
        "restore current.json from backup or re-promote the intended snapshot",
    ),
    rule(
        "storage.manifest_missing",
        "storage.identity",
        B,
        "the selected snapshot manifest is missing",
        "restore snapshots/<id>.json from backup",
    ),
    rule(
        "storage.manifest_invalid",
        "storage.identity",
        B,
        "the snapshot manifest does not parse under the contract",
        "restore the manifest from backup; never edit manifests by hand",
    ),
    rule(
        "storage.manifest_identity",
        "storage.identity",
        B,
        "the manifest's content does not hash to its snapshot id",
        "treat the snapshot as tampered; restore it from backup or re-import",
    ),
    rule(
        "storage.table_unknown",
        "storage.fragments",
        B,
        "the manifest lists a table outside the contract",
        "re-import with the current code; unknown tables are not readable by consumers",
    ),
    rule(
        "storage.table_missing",
        "storage.fragments",
        B,
        "a core table is absent from the manifest",
        "re-import; a snapshot without core tables cannot be promoted",
    ),
    rule(
        "storage.table_row_count",
        "storage.fragments",
        B,
        "a table's row_count differs from its fragments' rows",
        "restore the manifest; the declared size is not the stored size",
    ),
    rule(
        "storage.fragment_missing",
        "storage.fragments",
        B,
        "a referenced fragment is missing or not a regular file",
        "restore fragments/ from backup; do not promote this snapshot",
    ),
    rule(
        "storage.fragment_escape",
        "storage.fragments",
        B,
        "a fragment path escapes the fragment store",
        "treat the manifest as hostile; restore it from backup",
    ),
    rule(
        "storage.fragment_size",
        "storage.fragments",
        B,
        "a fragment's byte size differs from the manifest",
        "restore the fragment from backup (truncated or replaced bytes)",
    ),
    rule(
        "storage.fragment_hash",
        "storage.fragments",
        B,
        "a fragment's bytes do not hash to the manifest sha256",
        "restore the fragment from backup (bytes changed after sealing)",
    ),
    rule(
        "storage.fragment_unreadable",
        "storage.fragments",
        B,
        "a fragment is not readable Parquet",
        "restore the fragment from backup",
    ),
    rule(
        "storage.fragment_rows",
        "storage.fragments",
        B,
        "a fragment's Parquet row count differs from the manifest",
        "restore the manifest or fragment; the declared row count is wrong",
    ),
    rule(
        "storage.fragment_schema",
        "storage.fragments",
        B,
        "a fragment's column names or types differ from the table contract",
        "re-import with the current contract; do not cast by hand",
    ),
    rule(
        "storage.partition_declaration",
        "storage.fragments",
        B,
        "partition labels are missing, duplicated or present on an unpartitioned table",
        "rebuild the snapshot so each partition is declared once",
    ),
    rule(
        "storage.partition_mismatch",
        "storage.fragments",
        B,
        "a fragment holds rows of a different partition",
        "rebuild the snapshot; a misplaced row is invisible to partition-scoped readers",
    ),
    rule(
        "contract.key_duplicate",
        "contract.keys",
        B,
        "two rows share a table key",
        "re-import; resolve the duplicate identity at the source",
    ),
    rule(
        "contract.null_required",
        "contract.keys",
        B,
        "a non-nullable column holds nulls",
        "re-import; the contract forbids nulls here",
    ),
    rule(
        "contract.enum",
        "contract.values",
        B,
        "a column holds a value outside its enumeration",
        "re-import with a mapped value; never widen the enumeration to hide it",
    ),
    rule(
        "contract.non_finite",
        "contract.values",
        B,
        "a float column holds NaN or infinity",
        "re-import; a missing value must be null, not NaN",
    ),
    rule(
        "contract.json_column",
        "contract.values",
        B,
        "a JSON column holds invalid JSON, duplicate keys or non-finite numbers",
        "re-import the row from its source",
    ),
]

ENUMS: dict[tuple[str, str], tuple[str, ...]] = {
    ("matches", "stage_type"): tuple(s.value for s in StageType),
    ("matches", "status"): tuple(s.value for s in MatchStatus),
    ("matches", "date_precision"): tuple(s.value for s in DatePrecision),
    ("matches", "provenance"): tuple(s.value for s in Provenance),
    ("player_games", "date_quality"): tuple(s.value for s in DateQuality),
    ("player_games", "provenance"): tuple(s.value for s in Provenance),
    ("player_games", "link_method"): ("key", "date_tiebreak", "row_order", "source_url"),
    ("player_games", "result"): ("W", "L", "D"),
    ("players", "identity_status"): tuple(s.value for s in IdentityStatus),
    ("players", "birth_date_quality"): tuple(s.value for s in BirthDateQuality),
    ("players", "provenance"): tuple(s.value for s in Provenance),
    ("source_observations", "outcome"): tuple(s.value for s in CheckOutcome),
    ("source_observations", "source_mode"): tuple(s.value for s in SourceMode),
    ("quality_issues", "severity"): tuple(s.value for s in Severity),
    ("quality_issues", "status"): ("open", "accepted", "resolved"),
    ("legacy_predictions", "origin"): tuple(s.value for s in Origin),
}


def check_identity(ctx: AuditContext) -> list[str]:
    snap = ctx.snapshot
    assert snap is not None
    for rule_id, entity, message in snap.problems:
        ctx.add(rule_id, entity, message=message)
    ctx.count("manifests", 1 if snap.manifest_raw is not None else 0)
    return []


def check_fragments(ctx: AuditContext) -> list[str]:
    snap = ctx.snapshot
    assert snap is not None
    m = snap.manifest
    if m is None:
        raise CheckSkipped(Status.UNKNOWN, "no parseable manifest")
    for name in CORE_TABLES:
        if name not in m.tables:
            ctx.add("storage.table_missing", f"table:{name}")
    for name, entry in sorted(m.tables.items()):
        spec = TABLES.get(name)
        if spec is None:
            ctx.add("storage.table_unknown", f"table:{name}")
        declared = sum(f.rows for f in entry.fragments)
        if declared != entry.row_count:
            ctx.add("storage.table_row_count", f"table:{name}", expected=declared, actual=entry.row_count)
        parts = [f.partition for f in entry.fragments]
        if spec is not None and spec.partition_by:
            if any(p is None for p in parts) or len(set(parts)) != len(parts):
                ctx.add("storage.partition_declaration", f"table:{name}", actual=sorted(str(p) for p in parts))
        elif len(entry.fragments) > 1 or any(p is not None for p in parts):
            ctx.add("storage.partition_declaration", f"table:{name}", actual=[str(p) for p in parts])
    for frag in snap.fragments:
        ctx.count("fragments")
        ev = {"path": frag.ref.path, "sha256": frag.ref.sha256}
        if frag.data is None:
            rid = (
                "storage.fragment_escape" if frag.problem and "escapes" in frag.problem else "storage.fragment_missing"
            )
            ctx.add(rid, frag.entity, table=frag.table, evidence=ev, message=frag.problem or "")
            continue
        ctx.count("bytes", len(frag.data))
        if frag.problem:
            rid = "storage.fragment_size" if frag.problem.startswith("size") else "storage.fragment_hash"
            ctx.add(
                rid,
                frag.entity,
                table=frag.table,
                evidence=ev,
                message=frag.problem,
                expected=frag.ref.bytes if rid == "storage.fragment_size" else frag.ref.sha256,
            )
        if frag.parquet_error:
            ctx.add(
                "storage.fragment_unreadable", frag.entity, table=frag.table, evidence=ev, message=frag.parquet_error
            )
            continue
        ctx.count("rows", frag.num_rows or 0)
        if frag.num_rows != frag.ref.rows:
            ctx.add(
                "storage.fragment_rows",
                frag.entity,
                table=frag.table,
                expected=frag.ref.rows,
                actual=frag.num_rows,
                evidence=ev,
            )
        spec = TABLES.get(frag.table)
        if spec is None or frag.schema is None:
            continue
        want = spec.arrow_schema()
        got_cols = [(f.name, str(f.type)) for f in frag.schema]
        want_cols = [(f.name, str(f.type)) for f in want]
        if got_cols != want_cols:
            diff = sorted(set(got_cols) ^ set(want_cols))
            ctx.add(
                "storage.fragment_schema",
                frag.entity,
                table=frag.table,
                actual=[f"{n}:{t}" for n, t in diff][:20],
                evidence=ev,
            )
        if spec.partition_by and frag.ref.partition is not None and frag.partition_values is not None:
            wrong = [str(v) for v in frag.partition_values if str(v) != frag.ref.partition]
            if wrong:
                ctx.add(
                    "storage.partition_mismatch",
                    frag.entity,
                    table=frag.table,
                    field=spec.partition_by,
                    expected=frag.ref.partition,
                    actual=wrong[:10],
                    evidence=ev,
                )
    return []


def _ident(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"unsafe identifier {name!r}")
    return f'"{name}"'


def _available(ctx: AuditContext) -> list[str]:
    """Contract tables whose fragments all verified (others are UNKNOWN for value checks)."""
    assert ctx.snapshot is not None and ctx.snapshot.manifest is not None
    out, unknown = [], []
    for name in sorted(ctx.snapshot.manifest.tables):
        if name not in TABLES:
            continue
        try:
            ctx.need(name)
            out.append(name)
        except CheckSkipped:
            unknown.append(name)
    if unknown:
        ctx.coverage.setdefault("unverifiable_tables", sorted(set(unknown)))
    return out


def check_keys(ctx: AuditContext) -> list[str]:
    if ctx.snapshot is None or ctx.snapshot.manifest is None:
        raise CheckSkipped(Status.UNKNOWN, "no parseable manifest")
    tables = _available(ctx)
    for name in tables:
        spec = TABLES[name]
        n = ctx.rows(f"SELECT count(*) FROM {_ident(name)}")[0][0]  # noqa: S608 - identifier validated
        ctx.count("rows", n)
        key = ", ".join(_ident(k) for k in spec.key)
        for (k,) in ctx.rows(
            f"SELECT concat_ws('|', {key}) FROM {_ident(name)} GROUP BY {key} HAVING count(*) > 1 ORDER BY 1"  # noqa: S608
        ):
            ctx.add("contract.key_duplicate", f"{name}:{k}", table=name, field=",".join(spec.key))
        required = [c.name for c in spec.columns if not c.nullable]
        if required:
            sel = ", ".join(f"count(*) FILTER (WHERE {_ident(c)} IS NULL)" for c in required)
            for c, nulls in zip(required, ctx.rows(f"SELECT {sel} FROM {_ident(name)}")[0], strict=True):  # noqa: S608
                if nulls:
                    ctx.add("contract.null_required", f"table:{name}", table=name, field=c, actual=int(nulls))
    unknown = sorted(set(ctx.snapshot.manifest.tables) & set(TABLES) - set(tables))
    return [f"table {t} not verifiable" for t in unknown]


def check_values(ctx: AuditContext) -> list[str]:
    if ctx.snapshot is None or ctx.snapshot.manifest is None:
        raise CheckSkipped(Status.UNKNOWN, "no parseable manifest")
    tables = set(_available(ctx))
    for (table, column), allowed in sorted(ENUMS.items()):
        if table not in tables:
            continue
        marks = ",".join("?" * len(allowed))
        for value, n in ctx.rows(
            f"SELECT {_ident(column)}, count(*) FROM {_ident(table)} WHERE {_ident(column)} IS NOT NULL "  # noqa: S608
            f"AND {_ident(column)} NOT IN ({marks}) GROUP BY 1 ORDER BY 1",
            list(allowed),
        ):
            ctx.add(
                "contract.enum",
                f"table:{table}",
                table=table,
                field=column,
                actual=value,
                evidence={"rows": int(n), "allowed": list(allowed)},
            )
    for table in sorted(tables):
        spec = TABLES[table]
        floats = [c.name for c in spec.columns if c.type == "float64"]
        if floats:
            sel = ", ".join(f"count(*) FILTER (WHERE isnan({_ident(c)}) OR isinf({_ident(c)}))" for c in floats)
            for c, n in zip(floats, ctx.rows(f"SELECT {sel} FROM {_ident(table)}")[0], strict=True):  # noqa: S608
                if n:
                    ctx.add("contract.non_finite", f"table:{table}", table=table, field=c, actual=int(n))
        for c in (c.name for c in spec.columns if c.type == "json"):
            key = ", ".join(_ident(k) for k in spec.key)
            bad = 0
            first: list[str] = []
            for k, text in ctx.rows(
                f"SELECT concat_ws('|', {key}), {_ident(c)} FROM {_ident(table)} "  # noqa: S608 - identifiers validated
                f"WHERE {_ident(c)} IS NOT NULL ORDER BY 1"
            ):
                ctx.count("json_cells")
                try:
                    strict_json(str(text).encode())
                except ValueError:
                    bad += 1
                    if len(first) < 5:
                        first.append(str(k))
            if bad:
                ctx.add(
                    "contract.json_column",
                    f"table:{table}",
                    table=table,
                    field=c,
                    actual=bad,
                    evidence={"first_keys": first},
                )
    return []


CHECKS = [
    CheckSpec("storage.identity", "storage", "pointer and manifest identity", check_identity),
    CheckSpec("storage.fragments", "storage", "fragment bytes, rows, schema and partitions", check_fragments),
    CheckSpec("contract.keys", "contract", "table keys unique and required columns non-null", check_keys),
    CheckSpec("contract.values", "contract", "enumerations, finite floats and JSON columns", check_values),
]
