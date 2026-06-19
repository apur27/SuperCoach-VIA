"""
Unit tests for the AFL squad-builder name/draft resolution layer in
``scrapers/squad_builder.py``.

These cover the pure normalisation/matching helpers and, most importantly, the
``pick_best`` cross-club same-name collision fix: when two different players
share a namekey (e.g. Bailey Williams / Western Bulldogs 2015 AND Bailey
Williams / West Coast 2018) the resolver must return the record for the club
actually being looked up, not just the earliest draft entry.

No network calls. File I/O uses ``tmp_path`` only -- real data files untouched.
Written BEFORE the implementation (TDD).
"""

import os
import sys

import pytest

# Make ``scrapers`` importable regardless of the directory pytest runs from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scrapers.squad_builder import (  # noqa: E402
    namekey,
    loosekey,
    parse_jersey,
    canon_team,
    firstname_compat,
    loose_ok,
    pick_best,
    lookup_draft,
    load_draft_index,
)


# ---------------------------------------------------------------------------
# namekey
# ---------------------------------------------------------------------------

def test_namekey_strips_nonalpha_and_lowercases():
    assert namekey("Bailey Williams") == "baileywilliams"
    assert namekey("O'Brien") == "obrien"
    assert namekey("Jy-Simpkin 7") == "jysimpkin"
    assert namekey("") == ""


# ---------------------------------------------------------------------------
# loosekey  (first initial + full surname)
# ---------------------------------------------------------------------------

def test_loosekey_first_initial_plus_surname():
    assert loosekey("Matthew Crouch") == "mcrouch"
    assert loosekey("Bailey Williams") == "bwilliams"


def test_loosekey_single_token_uses_initial_plus_token():
    # A single-token name has first==last word, so loosekey is initial+token.
    assert loosekey("Cripps") == "ccripps"


def test_loosekey_nonalpha_token_falls_back_to_namekey():
    # First token has no alpha chars -> fall back to namekey of the whole input.
    assert loosekey("7 Cripps") == namekey("7 Cripps") == "cripps"


def test_loosekey_empty_string():
    assert loosekey("") == ""
    assert loosekey("   ") == ""


# ---------------------------------------------------------------------------
# parse_jersey
# ---------------------------------------------------------------------------

def test_parse_jersey_integer_string():
    assert parse_jersey("7") == 7


def test_parse_jersey_float_string():
    assert parse_jersey("7.0") == 7


def test_parse_jersey_empty_returns_none():
    assert parse_jersey("") is None
    assert parse_jersey(None) is None


def test_parse_jersey_non_numeric_returns_none():
    assert parse_jersey("rushed") is None
    assert parse_jersey("abc") is None


# ---------------------------------------------------------------------------
# canon_team
# ---------------------------------------------------------------------------

def test_canon_team_known_variants_collapse():
    assert canon_team("West Coast") == "westcoast"
    assert canon_team("West Coast Eagles") == "westcoast"
    assert canon_team("Western Bulldogs") == "westernbulldogs"
    assert canon_team("Footscray") == "westernbulldogs"


def test_canon_team_never_collapses_port_into_adelaide():
    assert canon_team("Port Adelaide") == "portadelaide"
    assert canon_team("Adelaide") == "adelaide"
    assert canon_team("Port Adelaide") != canon_team("Adelaide")


def test_canon_team_unknown_falls_back_to_alpha_only():
    assert canon_team("Made Up FC") == "madeupfc"


# ---------------------------------------------------------------------------
# firstname_compat
# ---------------------------------------------------------------------------

def test_firstname_compat_exact():
    assert firstname_compat("matt", "matt") is True


def test_firstname_compat_prefix_matt_matthew():
    assert firstname_compat("matt", "matthew") is True
    assert firstname_compat("matthew", "matt") is True


def test_firstname_compat_three_char_common_prefix_lachie_lachlan():
    assert firstname_compat("lachie", "lachlan") is True


def test_firstname_compat_rejects_unrelated():
    assert firstname_compat("jack", "sam") is False
    assert firstname_compat("sam", "sid") is False  # only 's' in common


def test_firstname_compat_empty_is_false():
    assert firstname_compat("", "matt") is False
    assert firstname_compat("matt", "") is False


# ---------------------------------------------------------------------------
# loose_ok
# ---------------------------------------------------------------------------

def test_loose_ok_surname_mismatch_rejected():
    assert loose_ok("Matt Crouch", "Matt Rowell") is False


def test_loose_ok_firstname_mismatch_rejected():
    assert loose_ok("Jack Crisp", "Sam Crisp") is False


def test_loose_ok_valid_match_passes():
    assert loose_ok("Matt Crouch", "Matthew Crouch") is True


# ---------------------------------------------------------------------------
# pick_best  -- THE REGRESSION
# ---------------------------------------------------------------------------

def _bailey_williams_records():
    """Two distinct players that collide on namekey 'baileywilliams'."""
    bulldogs = {"year": "2015", "pick": "48", "club": "Western Bulldogs",
                "player_name": "Bailey Williams"}
    west_coast = {"year": "2018", "pick": "35", "club": "West Coast",
                  "player_name": "Bailey Williams"}
    return bulldogs, west_coast


def test_pick_best_resolves_collision_by_team():
    bulldogs, west_coast = _bailey_williams_records()
    recs = [bulldogs, west_coast]
    chosen = pick_best(recs, cteam="westcoast")
    assert chosen is west_coast
    assert chosen["pick"] == "35"


def test_pick_best_without_team_returns_earliest():
    bulldogs, west_coast = _bailey_williams_records()
    # order shouldn't matter -- earliest by year wins
    recs = [west_coast, bulldogs]
    chosen = pick_best(recs)
    assert chosen is bulldogs
    assert chosen["pick"] == "48"


def test_pick_best_team_given_but_no_match_falls_back_to_earliest():
    bulldogs, west_coast = _bailey_williams_records()
    recs = [west_coast, bulldogs]
    # Carlton isn't either player's club -> fall back to earliest (Bulldogs 2015)
    chosen = pick_best(recs, cteam="carlton")
    assert chosen is bulldogs


def test_pick_best_single_record():
    bulldogs, _ = _bailey_williams_records()
    assert pick_best([bulldogs], cteam="westcoast") is bulldogs


# ---------------------------------------------------------------------------
# lookup_draft
# ---------------------------------------------------------------------------

def _index(records, name_col="player_name"):
    """Build (by_nk, by_lk) dicts the way load_draft_index would."""
    from collections import defaultdict
    by_nk = defaultdict(list)
    by_lk = defaultdict(list)
    for r in records:
        nm = r.get(name_col, "")
        by_nk[namekey(nm)].append(r)
        by_lk[loosekey(nm)].append(r)
    return by_nk, by_lk


def test_lookup_draft_exact_match():
    rec = {"year": "2014", "pick": "7", "club": "Adelaide",
           "player_name": "Matt Crouch"}
    by_nk, by_lk = _index([rec])
    out, how = lookup_draft("Matt Crouch", "mattcrouch", by_nk, by_lk)
    assert out is rec
    assert how == "exact"


def test_lookup_draft_alias_cam_to_cameron():
    rec = {"year": "2017", "pick": "1", "club": "Brisbane Lions",
           "player_name": "Cameron Rayner"}
    by_nk, by_lk = _index([rec])
    # perf-file display namekey is 'camrayner'; alias maps it to 'cameronrayner'
    out, how = lookup_draft("Cam Rayner", "camrayner", by_nk, by_lk)
    assert out is rec
    assert how == "alias"


def test_lookup_draft_loose_match():
    rec = {"year": "2014", "pick": "7", "club": "Adelaide",
           "player_name": "Matthew Crouch"}
    by_nk, by_lk = _index([rec])
    # display namekey 'mattcrouch' has no exact entry; loosekey 'mcrouch' does
    out, how = lookup_draft("Matt Crouch", "mattcrouch", by_nk, by_lk)
    assert out is rec
    assert how == "loose"


def test_lookup_draft_no_match():
    by_nk, by_lk = _index([])
    out, how = lookup_draft("Nobody Here", "nobodyhere", by_nk, by_lk)
    assert out is None
    assert how == "none"


def test_lookup_draft_collision_resolved_by_team():
    bulldogs, west_coast = _bailey_williams_records()
    by_nk, by_lk = _index([bulldogs, west_coast])
    # Without cteam, exact lookup would return the earliest (Bulldogs);
    # with cteam='westcoast' it must return the West Coast record.
    out, how = lookup_draft("Bailey Williams", "baileywilliams",
                            by_nk, by_lk, cteam="westcoast")
    assert out is west_coast
    assert how == "exact"


# ---------------------------------------------------------------------------
# load_draft_index
# ---------------------------------------------------------------------------

def test_load_draft_index_reads_csv(tmp_path):
    csv_path = tmp_path / "draft.csv"
    csv_path.write_text(
        "year,round,pick,club,player_name,recruited_from\n"
        "2015,3,48,Western Bulldogs,Bailey Williams,Glenelg\n"
        "2018,2,35,West Coast,Bailey Williams,Dandenong Stingrays\n"
        "2014,1,7,Adelaide,Matthew Crouch,North Adelaide\n",
        encoding="utf-8",
    )
    by_nk, by_lk = load_draft_index(str(csv_path), "player_name")

    # Both Bailey Williams rows collapse under the same namekey.
    assert len(by_nk["baileywilliams"]) == 2
    clubs = {r["club"] for r in by_nk["baileywilliams"]}
    assert clubs == {"Western Bulldogs", "West Coast"}

    # Loose index keyed by first-initial + surname.
    assert len(by_lk["bwilliams"]) == 2
    assert len(by_nk["matthewcrouch"]) == 1
    assert by_lk["mcrouch"][0]["player_name"] == "Matthew Crouch"
