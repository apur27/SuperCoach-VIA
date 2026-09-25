"""Build tiny canonical snapshots that conform to domain.schemas.TABLES (test helper)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa

from supercoach_via.domain.schemas import TABLES, CheckOutcome, DatasetStatus, SnapshotManifest, ValidationReport
from supercoach_via.storage import snapshots


def table(name: str, rows: list[dict[str, Any]]) -> pa.Table:
    spec = TABLES[name]
    schema = spec.arrow_schema()
    defaults: dict[str, Any] = {"string": "unspecified", "json": "[]", "int32": 0, "int64": 0, "float64": 0.0, "bool": False}
    cols = {
        c.name: [r.get(c.name, None if c.nullable else defaults.get(c.type)) for r in rows] for c in spec.columns
    }
    return pa.table(cols, schema=schema)


def build(root: Path, clock: Callable[[], datetime], tables: dict[str, list[dict[str, Any]]]) -> SnapshotManifest:
    b = snapshots.SnapshotBuilder(root, clock=clock, code_version="test")
    for name, rows in tables.items():
        t = table(name, rows)
        if TABLES[name].partition_by:
            b.add_partitioned(name, t, TABLES[name].partition_by)
        else:
            b.add(name, t)
    cand = b.finish(status=DatasetStatus.DEMO)
    snapshots.promote(root, cand, ValidationReport(outcome=CheckOutcome.PASS))
    return cand.manifest
