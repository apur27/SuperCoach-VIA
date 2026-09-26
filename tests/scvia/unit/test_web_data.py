"""Canonical public JSON serialization (payload size and determinism)."""

from __future__ import annotations

import json
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

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


def _reference_compact(value: object) -> object:
    """The original, obviously-correct recursive definition."""
    if isinstance(value, float):
        return int(value) if value.is_integer() and abs(value) <= 2.0**53 else value
    if isinstance(value, list):
        return [_reference_compact(v) for v in value]
    if isinstance(value, dict):
        return {k: _reference_compact(v) for k, v in value.items()}
    return value


json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text(max_size=5),
    lambda children: st.lists(children, max_size=5) | st.dictionaries(st.text(max_size=3), children, max_size=5),
    max_leaves=40,
)


@given(json_values)
def test_fast_compaction_matches_the_reference(value: object) -> None:
    from supercoach_via.publish.web_data import _compact_numbers

    got = _compact_numbers(value)
    want = _reference_compact(value)
    assert json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True)
    assert canonical_json_bytes(value) == canonical_json_bytes(want)


def test_float_subclasses_are_compacted_too() -> None:
    import numpy as np

    assert canonical_json_bytes({"a": [np.float64(8.0), np.float64(2.5)], "b": np.float64(3.0)}) == b'{"a":[8,2.5],"b":3}\n'
