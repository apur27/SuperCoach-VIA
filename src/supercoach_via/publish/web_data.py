"""Public JSON resources: schema export, canonical serialization and hashing.

Release building (sharding, manifest assembly) lives in ``publish.release``; this module
owns the byte-level contract every public JSON file obeys:

- UTF-8, sorted keys, compact separators, trailing newline -> deterministic hashes.
- NaN/Infinity are rejected (``allow_nan=False``).
- Every document validates against its view model before serialization.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from supercoach_via.publish.view_models import PUBLIC_MODELS, PUBLIC_SCHEMA_VERSION

SCHEMA_BASE_ID = "https://supercoach-via.local/schemas/v1/"


_EXACT_INT = 2.0**53


def _compact_numbers(value: Any) -> Any:
    """Write integral floats as integers (8.0 -> 8): same JSON number, fewer bytes.

    Leaves are handled inline (no call per number): a release has tens of millions of them.
    """
    t = type(value)
    if t is not float and t not in _PLAIN and isinstance(value, float):  # e.g. numpy.float64
        value, t = float(value), float
    if t is float:
        return int(value) if value.is_integer() and -_EXACT_INT <= value <= _EXACT_INT else value
    if t is list:
        return [
            (int(v) if v.is_integer() and -_EXACT_INT <= v <= _EXACT_INT else v)
            if type(v) is float
            else (v if type(v) in _PLAIN else _compact_numbers(v))
            for v in value
        ]
    if t is dict:
        return {
            k: (int(v) if v.is_integer() and -_EXACT_INT <= v <= _EXACT_INT else v)
            if type(v) is float
            else (v if type(v) in _PLAIN else _compact_numbers(v))
            for k, v in value.items()
        }
    return value


_PLAIN = (str, int, bool, type(None))


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` deterministically; raises ValueError on NaN/Infinity."""
    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    payload = _compact_numbers(payload)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def export_json_schemas(out_dir: Path) -> dict[str, Path]:
    """Write one JSON Schema per public model; returns key -> written path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for key, model in sorted(PUBLIC_MODELS.items()):
        schema = model.model_json_schema(mode="serialization")
        schema["$id"] = f"{SCHEMA_BASE_ID}{key}.schema.json"
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["x-public-schema-version"] = PUBLIC_SCHEMA_VERSION
        path = out_dir / f"{key}.schema.json"
        path.write_text(json.dumps(schema, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written[key] = path
    return written
