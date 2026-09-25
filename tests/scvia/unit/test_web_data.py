"""Canonical public JSON serialization (payload size and determinism)."""

from __future__ import annotations

import json
import math

import pytest

from supercoach_via.publish.web_data import canonical_json_bytes


def test_integral_floats_are_written_as_integers() -> None:
    out = canonical_json_bytes({"a": 8.0, "b": [1.0, None, 2.5, -0.0], "c": {"d": 1e15}, "e": True, "f": 3})
    assert out == b'{"a":8,"b":[1,null,2.5,0],"c":{"d":1000000000000000},"e":true,"f":3}\n'


def test_values_are_preserved_exactly() -> None:
    payload = {"x": [0.1, 1 / 3, 2.0**53, 2.0**60, 1234.5, -7.0]}
    back = json.loads(canonical_json_bytes(payload))
    assert back["x"] == payload["x"]
    assert isinstance(back["x"][3], float)  # beyond 2**53 stays a float literal


def test_non_finite_values_are_refused() -> None:
    with pytest.raises(ValueError):
        canonical_json_bytes({"x": math.nan})
