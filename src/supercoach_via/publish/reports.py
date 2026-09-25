"""Deterministic Markdown report rendering and generated-section replacement."""

from __future__ import annotations

import re


class MarkerError(ValueError):
    """Generated-section markers are missing, duplicated or out of order."""


def _markers(name: str) -> tuple[str, str]:
    if not re.fullmatch(r"[a-z0-9_\-]+", name):
        raise MarkerError(f"invalid marker name {name!r}")
    return f"<!-- GEN:{name} START -->", f"<!-- GEN:{name} END -->"


def replace_marked_section(document: str, name: str, body: str) -> str:
    """Replace exactly one generated section; never append a duplicate."""
    start, end = _markers(name)
    if document.count(start) != 1 or document.count(end) != 1:
        raise MarkerError(f"expected exactly one {start!r} and one {end!r}")
    i, j = document.index(start), document.index(end)
    if j < i:
        raise MarkerError("END marker precedes START marker")
    return document[: i + len(start)] + "\n" + body.strip("\n") + "\n" + document[j:]
