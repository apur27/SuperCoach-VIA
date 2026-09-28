"""The smoke fingerprint covers executed source and ignores generated caches."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _inventory():
    path = ROOT / "docs/rewrite/evidence/source_inventory.py"
    spec = importlib.util.spec_from_file_location("source_inventory", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "web" / "src").mkdir(parents=True)
    (root / "web" / "src" / "app.tsx").write_text("view")
    (root / "web" / "dist").mkdir()
    (root / "web" / "dist" / "built.html").write_text("cache")
    (root / "src" / "__pycache__").mkdir(parents=True)
    (root / "src" / "__pycache__" / "mod.pyc").write_bytes(b"pyc")
    (root / "top_players_comprehensive.py").write_text("print(1)\n")
    (root / "config").mkdir()
    (root / "config" / "public_content.toml").write_text('[[article]]\npath = "docs/news/one.md"\n')
    (root / "docs" / "news").mkdir(parents=True)
    (root / "docs" / "news" / "one.md").write_text("article")
    return root


def test_web_source_and_the_legacy_ranker_change_the_digest_and_caches_do_not(tmp_path: Path) -> None:
    module = _inventory()
    root = _tree(tmp_path)
    original = module.inventory(root)["sha256"]
    (root / "web" / "src" / "app.tsx").write_text("edited view")
    assert module.inventory(root)["sha256"] != original
    (root / "web" / "src" / "app.tsx").write_text("view")
    (root / "top_players_comprehensive.py").write_text("print(2)\n")
    assert module.inventory(root)["sha256"] != original
    (root / "top_players_comprehensive.py").write_text("print(1)\n")
    (root / "web" / "dist" / "built.html").write_text("rebuilt")
    (root / "src" / "__pycache__" / "mod.pyc").write_bytes(b"other")
    assert module.inventory(root)["sha256"] == original


def test_scratch_must_match_the_selected_bytes(tmp_path: Path) -> None:
    module = _inventory()
    source = _tree(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    for path in source.rglob("*"):
        if path.is_file():
            target = scratch / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    assert module.compare_roots(source, scratch)["scratch_differences"] == []
    (scratch / "web" / "src" / "app.tsx").write_text("stale")
    kinds = {item["path"]: item["kind"] for item in module.compare_roots(source, scratch)["scratch_differences"]}
    assert kinds["web/src/app.tsx"] == "bytes_differ"
