"""scripts/reconciliation_mutation_probes.py helpers: edits are exact, new files, originals untouched."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "reconciliation_mutation_probes.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("recon_probes", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_replace_once_changes_only_the_first_match_after_the_anchor() -> None:
    m = _load()
    body = b"<td>5</td> A <td>5</td> B <td>5</td>"
    assert m.replace_once(body, b"<td>5</td>", b"<td>9</td>", after=b"A") == b"<td>5</td> A <td>9</td> B <td>5</td>"


def test_replace_once_missing_anchor_raises() -> None:
    m = _load()
    with pytest.raises(ValueError):
        m.replace_once(b"abc", b"b", b"x", after=b"zzz")


def test_link_tree_copy_then_replace_never_writes_through(tmp_path: Path) -> None:
    m = _load()
    src = tmp_path / "src"
    (src / "d").mkdir(parents=True)
    (src / "d" / "f.txt").write_text("orig")
    before = m.tree_digest(src)
    dst = tmp_path / "dst"
    m.link_tree(src, dst)
    assert os.stat(src / "d" / "f.txt").st_nlink == 2
    victim = dst / "d" / "f.txt"
    victim.unlink()  # the documented pattern: break the link, then write a new file
    victim.write_text("changed")
    assert m.tree_digest(src) == before
    assert m.tree_digest(dst) != before
