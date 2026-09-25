"""Shared fixtures for the new-package (supercoach_via) test tiers.

Tiers are assigned by directory:
- unit/, contract/: hermetic. Network is blocked; fixtures own their roots and clocks.
- integration/, performance/: real-corpus checks against a snapshot copy; marked
  ``integration`` so the pre-commit fast tier (``-m "not integration"``) excludes them.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        parts = Path(str(item.fspath)).parts
        if "scvia" not in parts:
            continue
        if "integration" in parts:
            item.add_marker(pytest.mark.integration)
        elif "performance" in parts:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.performance)
        elif "contract" in parts:
            item.add_marker(pytest.mark.contract)


class NetworkBlockedError(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    parts = Path(str(request.node.fspath)).parts
    if "unit" in parts or "contract" in parts:

        def guard(*_a: object, **_k: object) -> None:
            raise NetworkBlockedError("network access is blocked in hermetic tests")

        monkeypatch.setattr(socket.socket, "connect", guard)
        monkeypatch.setattr(socket, "create_connection", guard)
        monkeypatch.setattr(socket, "getaddrinfo", guard)
    yield


@pytest.fixture(scope="session")
def real_snapshot_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Data root holding a promoted real-corpus snapshot (integration/performance tiers only).

    ``SCVIA_SNAPSHOT_ROOT`` reuses an existing root. Otherwise the checked-in corpus plus the
    archived B1 repair evidence is imported, validated and promoted once per session under
    pytest's tmp dir, so a fresh clone needs no pre-built ``var/`` state.
    """
    import os

    override = os.environ.get("SCVIA_SNAPSHOT_ROOT")
    if override:
        return Path(override)
    from supercoach_via import pipeline
    from supercoach_via.settings import RunContext, Settings

    root = tmp_path_factory.mktemp("real-corpus") / "var"
    ctx = RunContext(settings=Settings(data_root=root))
    res = pipeline.ingest(ctx, source_root=REPO_ROOT, repairs=[(REPO_ROOT / "docs/rewrite/evidence/b1", 2026)])
    assert res.exit_code == 0 and res.promoted, res.as_dict()
    return root


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
