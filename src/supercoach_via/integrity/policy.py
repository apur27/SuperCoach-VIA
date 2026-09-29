"""Integrity policy: thresholds, accepted exceptions and the policy identity hashed into reports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from supercoach_via.domain.schemas import PLAYER_STAT_COLUMNS
from supercoach_via.integrity.report import AcceptedException
from supercoach_via.settings import default_config_dir

POLICY_FILE = "integrity_policy.yaml"
#: every file whose bytes can change a verdict is part of the policy identity
POLICY_INPUTS = (POLICY_FILE, "coverage.yaml", "stat_coverage_eras.yaml")


class PolicyError(ValueError):
    """The integrity policy file is missing or malformed (exit 2)."""


@dataclass(frozen=True)
class IntegrityPolicy:
    version: str
    sha256: str
    files: dict[str, str]
    sample_limit: int
    max_fixture_age_hours: float
    plausible_max: dict[str, float]
    exceptions: tuple[AcceptedException, ...]
    config_dir: Path


def load_policy(config_dir: Path | None = None) -> IntegrityPolicy:
    cfg = config_dir or default_config_dir()
    files: dict[str, str] = {}
    for name in POLICY_INPUTS:
        path = cfg / name
        if not path.is_file():
            raise PolicyError(f"policy input missing: {path}")
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        raw = yaml.safe_load((cfg / POLICY_FILE).read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise PolicyError(f"{POLICY_FILE}: {exc}") from exc
    if raw.get("schema_version") != 1:
        raise PolicyError(f"{POLICY_FILE}: unsupported schema_version {raw.get('schema_version')!r}")
    plausible = {str(k): float(v) for k, v in (raw.get("plausible_max") or {}).items()}
    unknown = set(plausible) - set(PLAYER_STAT_COLUMNS)
    if unknown:
        raise PolicyError(f"{POLICY_FILE}: plausible_max names non-canonical stats {sorted(unknown)}")
    exceptions = []
    for e in raw.get("known_exceptions") or []:
        if not isinstance(e, dict) or not all(e.get(k) for k in ("rule_id", "entity", "reason")):
            raise PolicyError(f"{POLICY_FILE}: an exception needs rule_id, entity and reason: {e!r}")
        exceptions.append(AcceptedException(str(e["rule_id"]), str(e["entity"]), str(e["reason"])))
    limit = int(raw.get("sample_limit", 20))
    if limit < 1:
        raise PolicyError(f"{POLICY_FILE}: sample_limit must be positive")
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return IntegrityPolicy(
        version=str(raw.get("policy_version", "unknown")),
        sha256=digest,
        files=files,
        sample_limit=limit,
        max_fixture_age_hours=float((raw.get("freshness") or {}).get("max_fixture_age_hours", 192)),
        plausible_max=plausible,
        exceptions=tuple(exceptions),
        config_dir=cfg,
    )
