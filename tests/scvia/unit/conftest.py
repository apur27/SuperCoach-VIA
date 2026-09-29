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

    import shutil

    # the same per-worker DEMO build the builder tests use (built once), copied so the site
    # added here never changes the release those tests read
    shared = tmp_path_factory.getbasetemp()
    env = demo_env(shared)
    cand = full_release(shared)
    release_dir = tmp_path_factory.mktemp("integrity-demo") / cand.release_dir.name
    shutil.copytree(cand.release_dir, release_dir)
    fx.add_site(release_dir)
    return IntegrityDemo(env=env, data_root=env.data_root, release_dir=release_dir)
