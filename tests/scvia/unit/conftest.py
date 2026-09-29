"""Fixtures shared by unit modules (built once per pytest worker)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class IntegrityDemo:
    """The DEMO snapshot, model, forecast and a sealed release with a site (read-only: copy before editing)."""

    env: object
    data_root: Path
    release_dir: Path


@pytest.fixture(scope="session")
def integrity_demo(tmp_path_factory: pytest.TempPathFactory) -> IntegrityDemo:
    from tests.scvia.unit import integrity_fixtures as fx
    from tests.scvia.unit.demo_release_env import demo_env, full_release

    base = tmp_path_factory.mktemp("integrity-demo")
    env = demo_env(base)
    cand = full_release(base)
    fx.add_site(cand.release_dir)
    return IntegrityDemo(env=env, data_root=env.data_root, release_dir=cand.release_dir)
