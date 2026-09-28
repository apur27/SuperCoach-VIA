"""Fingerprint the code and inputs a numeric smoke actually executes.

Usage: python source_inventory.py --compare SOURCE_ROOT SCRATCH_ROOT OUTPUT.json
"""

from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

PREFIXES = (
    "src/",
    "scripts/",
    "web/",
    "tests/",
    "schemas/",
    "config/",
    ".github/",
    ".githooks/",
    "docs/rewrite/switch-candidate/",
    "docs/rewrite/evidence/",
)
ROOT_NAMES = {
    ".node-version",
    ".python-version",
    "pyproject.toml",
    "uv.lock",
    "package.json",
    "package-lock.json",
    "CLAUDE.md",
    "AGENTS.md",
    ".gitignore",
}
SKIP_DIRS = {
    "node_modules",
    "dist",
    ".astro",
    ".e2e-dist",
    "test-results",
    "playwright-report",
    "coverage",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def _selected(root: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}

    def add(path: Path) -> None:
        if not path.is_file() or path.suffix in SKIP_SUFFIXES:
            return
        rel = path.relative_to(root).as_posix()
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise SystemExit(f"unsafe inventory path: {rel}")
        data = path.read_bytes()
        files[rel] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    for name in sorted(ROOT_NAMES):
        add(root / name)
    for path in sorted(root.glob("*.py")) + sorted(root.glob("*.sh")):
        add(path)
    for prefix in PREFIXES:
        base = root / prefix
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
                continue
            add(path)
    manifest = root / "config" / "public_content.toml"
    if manifest.is_file():
        body = tomllib.loads(manifest.read_text(encoding="utf-8"))
        for item in body.get("article", []):
            add(root / str(item["path"]))
    return files


def inventory(root: Path) -> dict[str, Any]:
    files = _selected(root)
    canonical = json.dumps({"files": files}, sort_keys=True, separators=(",", ":")).encode()
    return {
        "files": len(files),
        "bytes": sum(item["bytes"] for item in files.values()),
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "paths": files,
    }


def compare_roots(source: Path, scratch: Path) -> dict[str, Any]:
    left, right = _selected(source), _selected(scratch)
    differences: list[dict[str, str]] = []
    for name in sorted(set(left) | set(right)):
        if name not in right:
            differences.append({"path": name, "kind": "missing_from_scratch"})
        elif name not in left:
            differences.append({"path": name, "kind": "only_in_scratch"})
        elif left[name]["sha256"] != right[name]["sha256"]:
            differences.append({"path": name, "kind": "bytes_differ"})
    canonical = json.dumps({"files": left}, sort_keys=True, separators=(",", ":")).encode()
    return {
        "source_root": str(source),
        "scratch_root": str(scratch),
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": len(left),
        "bytes": sum(item["bytes"] for item in left.values()),
        "scratch_differences": differences,
        "paths": left,
    }


def main() -> None:
    if len(sys.argv) != 5 or sys.argv[1] != "--compare":
        raise SystemExit("usage: source_inventory.py --compare SOURCE SCRATCH OUTPUT.json")
    report = compare_roots(Path(sys.argv[2]), Path(sys.argv[3]))
    Path(sys.argv[4]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "files": report["files"],
        "bytes": report["bytes"],
        "inventory_sha256": report["inventory_sha256"],
        "differences": len(report["scratch_differences"]),
    }))
    if report["scratch_differences"]:
        raise SystemExit("source inventory does not match the scratch checkout")


if __name__ == "__main__":
    main()
