"""Facts cache: content-and-parser keyed, corruption is a miss (T20, T21; DESIGN section 10)."""

from __future__ import annotations

from pathlib import Path

import pytest

from supercoach_via.reconciliation.cache import FactsCache, parser_identity

SHA = "ab" * 32


def test_round_trip_and_hit_miss_accounting(tmp_path: Path) -> None:
    c = FactsCache(tmp_path, "p1")
    assert c.read("match", SHA) is None and c.misses == 1
    c.write("match", SHA, b'{"x":1}')
    assert c.read("match", SHA) == b'{"x":1}' and c.hits == 1 and c.stats() == {"hits": 1, "misses": 1, "corrupt": 0}


def test_a_different_parser_identity_never_reads_another_parsers_entry(tmp_path: Path) -> None:
    FactsCache(tmp_path, "p1").write("match", SHA, b"old")
    assert FactsCache(tmp_path, "p2").read("match", SHA) is None


def test_corrupt_truncated_or_swapped_entries_are_misses(tmp_path: Path) -> None:
    c = FactsCache(tmp_path, "p1")
    c.write("profile", SHA, b'{"games":[1,2,3]}')
    path = c.path("profile", SHA)
    raw = path.read_bytes()
    for bad in (raw[:-3], raw.replace(b"games", b"gamez"), b"garbage", b"", raw.split(b"\n")[0]):
        path.write_bytes(bad)
        fresh = FactsCache(tmp_path, "p1")
        assert fresh.read("profile", SHA) is None and fresh.corrupt + fresh.misses >= 1
    other = "cd" * 32
    c.write("profile", other, b"other")
    path.write_bytes(c.path("profile", other).read_bytes())  # a valid file for a different body
    assert FactsCache(tmp_path, "p1").read("profile", SHA) is None


def test_parts_are_separate_entries(tmp_path: Path) -> None:
    c = FactsCache(tmp_path, "p1")
    c.write("profile", SHA, b"core", "core")
    c.write("profile", SHA, b"s2026", "s2026")
    assert c.read("profile", SHA, "core") == b"core" and c.read("profile", SHA, "s2026") == b"s2026"
    assert c.read("profile", SHA, "s2025") is None


def test_parser_identity_changes_with_extra_and_is_stable() -> None:
    a = parser_identity()
    assert a == parser_identity() and len(a) == 12
    assert parser_identity("notes-alias") != a


def test_code_closure_follows_every_package_import_transitively() -> None:  # F-M2
    from supercoach_via.reconciliation.cache import code_closure

    closure = code_closure(("reconciliation/season.py",))
    for rel in (
        "reconciliation/season.py",
        "reconciliation/cells.py",
        "reconciliation/aggregate.py",
        "reconciliation/rules.py",
        "reconciliation/evidence.py",
        "reconciliation/source.py",
        "reconciliation/findings.py",
        "reconciliation/schema.py",
        "integrity/sourcepages.py",
    ):
        assert rel in closure, rel
    assert closure == tuple(sorted(closure))  # stable order: the salt never depends on traversal order


def test_the_parser_identity_covers_the_facts_builder() -> None:  # F-M2: facts.py shapes cached headers
    from supercoach_via.reconciliation.cache import parser_files

    files = parser_files()
    assert "reconciliation/facts.py" in files and "reconciliation/source.py" in files


def test_editing_any_module_in_the_closure_changes_the_unit_salt(monkeypatch: pytest.MonkeyPatch) -> None:  # F-M2
    from supercoach_via.reconciliation import cache as CA

    real = CA._file_bytes
    base = CA.code_digest(("reconciliation/season.py",))
    monkeypatch.setattr(
        CA, "_file_bytes", lambda rel: real(rel) + (b"#x" if rel == "reconciliation/aggregate.py" else b"")
    )
    assert CA.code_digest(("reconciliation/season.py",)) != base
