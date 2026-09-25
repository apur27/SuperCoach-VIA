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


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
