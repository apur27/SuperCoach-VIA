"""Injective public path keys: canonical base64url, plus legacy colon aliases."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from supercoach_via.publish.resources import KeyCodecError, parse_public_key, public_key

_ALPHABET = st.sampled_from("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:_-.")


@given(st.text(alphabet=_ALPHABET, min_size=1, max_size=80))
@settings(max_examples=80)
def test_canonical_keys_round_trip_and_hide_colons(identifier: str) -> None:
    if ".." in identifier:
        return
    key = public_key(identifier)
    assert key.startswith("k.")
    assert ":" not in key
    assert len(key) <= 200
    assert parse_public_key(key) == identifier
    assert "__" not in key or key.startswith("k.")


def test_known_collisions_and_the_triple_underscore_id() -> None:
    assert public_key("a__b") != public_key("id.a_x_b")
    assert parse_public_key(public_key("legacy:a___b")) == "legacy:a___b"
    assert parse_public_key("legacy__x_y_01011990") == "legacy:x_y_01011990"


@given(st.lists(st.text(alphabet=_ALPHABET, min_size=1, max_size=40), min_size=2, max_size=30, unique=True))
@settings(max_examples=40)
def test_canonical_keys_are_injective(identifiers: list[str]) -> None:
    safe = [i for i in identifiers if ".." not in i]
    keys = [public_key(i) for i in safe]
    assert len(set(keys)) == len(safe)


def test_oversized_ids_are_refused_instead_of_truncated() -> None:
    huge = "legacy:" + ("a" * 180)
    with pytest.raises(KeyCodecError, match="alias"):
        public_key(huge)
