"""Safe human-facing CSV and deterministic ZIP packaging."""

from __future__ import annotations

import csv
import io
import re
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

_FORMULA_PREFIX = ("=", "+", "-", "@", "\t", "\r")
_NUMERIC = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_SAFE_MEMBER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-./]{0,200}$")
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def neutralise_cell(value: Any) -> Any:
    """Prefix spreadsheet-formula-looking TEXT with a quote; numbers are untouched."""
    if value is None or isinstance(value, bool | int | float):
        return value
    text = str(value)
    if text.startswith(_FORMULA_PREFIX) and not _NUMERIC.match(text):
        return "'" + text
    return text


def safe_csv_bytes(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow([neutralise_cell(h) for h in header])
    for row in rows:
        writer.writerow(["" if v is None else neutralise_cell(v) for v in row])
    return buf.getvalue().encode("utf-8")


def check_member(name: str) -> str:
    if (
        not _SAFE_MEMBER.match(name)
        or ".." in name.split("/")
        or name.startswith("/")
        or "//" in name
        or name.endswith("/")
    ):
        raise ValueError(f"unsafe archive member name: {name!r}")
    return name


def deterministic_zip(files: Mapping[str, bytes]) -> bytes:
    """ZIP with sorted, validated member names, fixed timestamps and permissions."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in sorted(files):
            info = zipfile.ZipInfo(check_member(name), date_time=ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            zf.writestr(info, files[name])
    return buf.getvalue()
