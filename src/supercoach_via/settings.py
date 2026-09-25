"""Validated settings, resolved roots and the run context passed to services.

Precedence: defaults < TOML file (``--config`` or ``SCVIA_CONFIG``) < narrowly named
environment overrides (``SCVIA_DATA_ROOT``, ``SCVIA_OUTPUT_ROOT``, ``SCVIA_SOURCE_ROOT``,
``SCVIA_PUBLIC_BASE``, ``SCVIA_SITE_URL``, ``SCVIA_TIMEZONE``). Unknown keys fail.
"""

from __future__ import annotations

import logging
import os
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator

ENV_OVERRIDES: dict[str, str] = {
    "SCVIA_DATA_ROOT": "data_root",
    "SCVIA_OUTPUT_ROOT": "output_root",
    "SCVIA_SOURCE_ROOT": "source_root",
    "SCVIA_PUBLIC_BASE": "public_base",
    "SCVIA_SITE_URL": "site_url",
    "SCVIA_TIMEZONE": "timezone",
}


class SettingsError(ValueError):
    """Invalid configuration (exit code 2)."""


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data_root: Path = Path("var")
    output_root: Path = Path("dist")
    source_root: Path = Path()
    public_base: str = "/"
    site_url: str | None = None
    timezone: str = "Australia/Melbourne"
    season: int | None = Field(default=None, ge=1897, le=2100)
    source_policy_path: Path = Path("config/source_policies.toml")
    user_agent: str = "supercoach-via/0.1 (+https://github.com/; research; contact via repository)"
    editorial_enabled: bool = False
    retain_releases: int = Field(default=3, ge=1, le=50)

    @field_validator("public_base")
    @classmethod
    def _base(cls, v: str) -> str:
        if not v.startswith("/") or not v.endswith("/") or "//" in v or ".." in v:
            raise ValueError("public_base must look like '/' or '/SuperCoach-VIA/'")
        return v

    @field_validator("site_url")
    @classmethod
    def _site(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith("https://"):
            raise ValueError("site_url must be https")
        return v

    @field_validator("timezone")
    @classmethod
    def _tz(cls, v: str) -> str:
        ZoneInfo(v)  # raises for unknown zones
        return v


def load_settings(
    config_path: Path | None = None,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Settings:
    env = os.environ if env is None else env
    data: dict[str, Any] = {}
    path = config_path or (Path(env["SCVIA_CONFIG"]) if env.get("SCVIA_CONFIG") else None)
    if path is not None:
        try:
            data.update(tomllib.loads(path.read_text(encoding="utf-8")))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise SettingsError(f"cannot read config {path}: {exc}") from exc
    for var, key in ENV_OVERRIDES.items():
        if env.get(var):
            data[key] = env[var]
    for key, value in (overrides or {}).items():
        if value is not None:
            data[key] = value
    try:
        return Settings.model_validate(data)
    except Exception as exc:  # pydantic.ValidationError -> config error
        raise SettingsError(str(exc)) from exc


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class RunContext:
    """Explicit dependencies for services: settings, clock, logger and optional HTTP client."""

    settings: Settings
    clock: Callable[[], datetime] = utc_now
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("scvia"))
    http: Any = None  # ingest.http.HttpClient when network use is explicitly enabled
    run: Any = None  # storage.runs.RunStore when inside a run

    @property
    def data_root(self) -> Path:
        return self.settings.data_root

    @property
    def output_root(self) -> Path:
        return self.settings.output_root
