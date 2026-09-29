"""Semantic-result cache for integrity units, keyed only by verified content identity.

An entry is reused only when every input that could change its result is identical:
checker code, rules, policy, scope and ``as_of`` (the salt), the unit's canonical
dependencies (fragment hashes and the rows it reads) and the SHA-256 of every public file
it reads, as measured in this run. File names, sizes and modification times are never
part of a key. Each entry carries a digest of its own body; a corrupt, truncated or edited
entry is discarded and recomputed, never trusted.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from supercoach_via.domain.schemas import Severity
from supercoach_via.integrity.public_compare import UnitResult
from supercoach_via.integrity.report import Finding, Kind, canonical_bytes

_FORMAT = "scvia-integrity-cache/1"


def unit_key(salt: str, unit_id: str, dependencies: str, resources: dict[str, tuple[str, int]]) -> str:
    body = {
        "salt": salt,
        "unit": unit_id,
        "deps": dependencies,
        "resources": sorted([rel, sha] for rel, (sha, _n) in resources.items()),
    }
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def _finding_from(d: dict[str, Any]) -> Finding:
    return Finding(
        rule_id=d["rule_id"],
        check_id=d["check_id"],
        kind=Kind(d["kind"]),
        severity=Severity(d["severity"]),
        status=d["status"],
        entity=d["entity"],
        table=d["table"],
        field=d["field"],
        season=d["season"],
        expected=d["expected"],
        actual=d["actual"],
        evidence=d["evidence"],
        message=d["message"],
        action=d["action"],
        acceptance=d["acceptance"],
    )


class SemanticCache:
    def __init__(self, root: Path, salt: str):
        self.root = root
        self.salt = salt
        self.hits = 0
        self.misses = 0
        self.invalid = 0
        self.stored = 0

    def _path(self, key: str) -> Path:
        return self.root / "units" / key[:2] / f"{key}.json"

    def get(self, key: str) -> UnitResult | None:
        path = self._path(key)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            self.misses += 1
            return None
        except OSError:
            self.invalid += 1
            return None
        try:
            doc = json.loads(raw)
            body = doc["body"]
            if doc.get("format") != _FORMAT or doc.get("key") != key:
                raise ValueError("wrong format or key")
            if hashlib.sha256(canonical_bytes(body)).hexdigest() != doc.get("body_sha256"):
                raise ValueError("body digest mismatch")
            result = UnitResult(
                unit_id=body["unit_id"],
                findings=[_finding_from(f) for f in body["findings"]],
                examined={str(k): int(v) for k, v in body["examined"].items()},
                expected_paths=[str(p) for p in body["expected_paths"]],
            )
        except (ValueError, KeyError, TypeError):
            self.invalid += 1
            return None
        self.hits += 1
        return result

    def put(self, key: str, result: UnitResult) -> None:
        body = {
            "unit_id": result.unit_id,
            "findings": [{**f.as_dict(), "kind": f.kind.value} for f in result.findings],
            "examined": result.examined,
            "expected_paths": result.expected_paths,
        }
        doc = {
            "format": _FORMAT,
            "key": key,
            "body_sha256": hashlib.sha256(canonical_bytes(body)).hexdigest(),
            "body": body,
        }
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(canonical_bytes(doc))
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self.stored += 1

    def stats(self) -> dict[str, int]:
        return {
            "reused": self.hits,
            "computed": self.misses + self.invalid,
            "invalidated": self.invalid,
            "stored": self.stored,
        }
