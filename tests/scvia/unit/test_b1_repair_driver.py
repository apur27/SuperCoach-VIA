"""The archived B1 driver (docs/rewrite/evidence/b1/b1_repair_fetch.py) stays within its authorization.

It is evidence of what ran, not a pipeline entry point; these checks pin the properties
the owner authorised: <=10 requests, targets named, earlier payloads re-served locally.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[3]
DRIVER = REPO / "docs" / "rewrite" / "evidence" / "b1" / "b1_repair_fetch.py"


def _driver():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("b1_repair_fetch", DRIVER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_driver_cap_and_targets() -> None:
    mod = _driver()
    assert mod.MAX_REQUESTS <= 10
    assert set(mod.TARGETS) == {"Flynn Perez", "Jack Dalton", "Will Brodie"}
    assert mod.TARGETS["Jack Dalton"][1] is None  # new identity, never the 1876 namesake


def test_reuse_transport_serves_archived_bytes_without_network() -> None:
    mod = _driver()
    t = mod.ReuseTransport({"https://afltables.com/afl/seas/2026.html": b"cached"})
    resp = t.handle_request(httpx.Request("GET", "https://afltables.com/afl/seas/2026.html"))
    assert resp.status_code == 200 and resp.read() == b"cached"
    assert not t.network_requests and t.served == ["https://afltables.com/afl/seas/2026.html"]


def test_recorded_evidence_is_within_the_cap() -> None:
    manifest = json.loads((DRIVER.parent / "fetch-manifest.json").read_text(encoding="utf-8"))
    assert manifest["requests_total"] <= 10 and manifest["outcome"] == "PASS"
