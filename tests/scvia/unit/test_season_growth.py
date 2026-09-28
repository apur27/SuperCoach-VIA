"""Season-byte attribution must follow canonical match-detail keys, not only legacy filenames."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "season_growth",
    Path(__file__).resolve().parents[3] / "docs/rewrite/evidence/season_growth.py",
)
assert _SPEC and _SPEC.loader
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def test_canonical_match_detail_is_attributed_to_its_season(tmp_path: Path) -> None:
    site = tmp_path / "site"
    detail = site / "data" / "rel1" / "matches" / "detail"
    detail.mkdir(parents=True)
    body = json.dumps({"summary": {"season": 2024, "match_id": "legacy:2024:r01:a-b"}}).encode()
    (detail / "k.abc.json").write_bytes(body)
    (site / "index.html").write_text("shell")
    out = _MOD.attribute(site)
    assert out["seasons"]["2024"]["bytes"] == len(body)
    assert out["base_bytes"] == len(b"shell")
