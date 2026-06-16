"""
AFL draft school-affiliation classifier.

Turns the slash-separated pathway chain in ``draftguru_enrichment.csv``'s
``original_club`` field (e.g. "Balwyn Greythorn / Carey Grammar / Oakleigh U18")
into a single elite-school affiliation tag.

Two pure string steps, then an end-to-end CSV build:

  1. ``extract_school`` -- scan the " / "-separated pathway and return the first
     segment that looks like a school (College/Grammar/School/Academy/...).
  2. ``classify_school`` -- map that school name to an affiliation bucket
     (APS / GPS_QLD / GPS_WA / GPS_NSW / GPS_SA / other_private / state /
     unknown) via a hardcoded, case-insensitive, substring lookup.

Matching is substring-based so "Scotch College, Melbourne" still matches the
"Scotch College" key. Ordering inside ``classify_school`` is load-bearing: the
WA-suffixed "Wesley College (WA)" must be tested before the bare Victorian
"Wesley College", and the "state high school" indicators before any private
key, so a substring of a longer private name never wins by accident.

Output: ``data/drafts/afl_draft_schools.csv`` with columns
  year, pick, player_name, original_club, extracted_school, school_type,
  grade, games, goals

Style mirrors ``scrapers/draft_scraper.py``: pure helpers up top, a thin CLI at
the bottom, pandas for the delta-friendly CSV write.
"""

import sys
from typing import List, Optional

import pandas as pd

DEFAULT_INPUT = "data/drafts/draftguru_enrichment.csv"
DEFAULT_OUTPUT = "data/drafts/afl_draft_schools.csv"

OUTPUT_COLUMNS = [
    "year", "pick", "player_name", "original_club",
    "extracted_school", "school_type", "grade", "games", "goals",
]

# Tokens that mark a pathway segment as a school rather than a club/league.
_SCHOOL_TOKENS = ("college", "grammar", "school", "academy", "institute", "collegiate")

# Tokens that mark a school as a state (public) school.
_STATE_TOKENS = ("high school", "secondary college", "state high")

# Affiliation lookup. Keys are matched case-insensitively as substrings of the
# extracted school name. Within each bucket the longest / most-specific keys
# come first; across buckets, ``classify_school`` checks STATE, then the
# WA-suffixed Wesley, then GPS_*, then APS, then other_private (see that
# function for the exact precedence and why it matters).
_APS = [
    "melbourne grammar",
    "scotch college",
    "geelong grammar",
    "haileybury college",
    "xavier college",
    "brighton grammar",
    "caulfield grammar",
    "st kevin's college",
    "carey grammar",
    "assumption college",
    "trinity grammar (kew)",
    "trinity grammar",
    "wesley college",  # Victorian Wesley -- WA variant handled before this list
]

_GPS_QLD = [
    "brisbane grammar",
    "the southport school",
    "southport school",
    "st joseph's nudgee college",
    "nudgee college",
    "anglican church grammar",
    "churchie",
    "ipswich grammar",
    "gregory terrace",
    "toowoomba grammar",
    "villanova college",
    "padua college",
]

_GPS_WA = [
    "wesley college (wa)",
    "christ church grammar",
    "aquinas college (wa)",
    "aquinas college",
    "hale school",
    "guildford grammar",
    "trinity college (wa)",
    "presbyterian ladies' college (wa)",
]

_GPS_NSW: List[str] = []

_GPS_SA = [
    "prince alfred college",
    "st peter's college (sa)",
    "st peter's college",
    "rostrevor college",
    "scotch college (sa)",
    "princes' school",
    "st ignatius' college (sa)",
    "christian brothers college adelaide",
]

_OTHER_PRIVATE = [
    "emmanuel college",
    "st patrick's college",
    "marist college",
    "de la salle college",
    "parade college",
    "salesian college",
    "sacred heart college",
    "marcellin college",
    "st bernard's college",
    "st joseph's college",
    "whitefriars college",
    "st bede's college",
    "ivanhoe grammar",
    "yarra valley grammar",
    "ballarat clarendon college",
    "geelong college",
]


def extract_school(original_club: Optional[str]) -> Optional[str]:
    """
    Return the first " / "-separated segment of ``original_club`` that looks
    like a school, or None.

    A segment "looks like a school" if (lower-cased) it contains any of the
    school tokens (College / Grammar / School / Academy / Institute /
    Collegiate). The first such segment, stripped of surrounding whitespace, is
    returned. None / empty / whitespace-only / non-string input returns None.
    """
    if not original_club or not isinstance(original_club, str):
        return None
    for segment in original_club.split(" / "):
        seg = segment.strip()
        low = seg.lower()
        if any(token in low for token in _SCHOOL_TOKENS):
            return seg
    return None


def _matches_any(low: str, keys: List[str]) -> bool:
    return any(key in low for key in keys)


def classify_school(school_name: Optional[str]) -> str:
    """
    Map a school name to an affiliation bucket.

    Returns one of: APS, GPS_QLD, GPS_WA, GPS_NSW, GPS_SA, other_private,
    state, unknown. Matching is case-insensitive substring matching.

    Precedence (first hit wins) and why:
      1. state-school indicators ("High School", "Secondary College",
         "State High") -- catch public schools before any private key can
         match a coincidental substring.
      2. GPS_WA -- "Wesley College (WA)" must beat the APS "Wesley College".
      3. GPS_QLD / GPS_NSW / GPS_SA -- interstate elite schools.
      4. APS -- Victorian elite schools.
      5. other_private -- non-GPS Catholic/independent schools.
      6. unknown -- no key matched.
    """
    if not school_name or not isinstance(school_name, str):
        return "unknown"
    low = school_name.lower()

    if _matches_any(low, list(_STATE_TOKENS)):
        return "state"
    if _matches_any(low, _GPS_WA):
        return "GPS_WA"
    if _matches_any(low, _GPS_QLD):
        return "GPS_QLD"
    if _matches_any(low, _GPS_NSW):
        return "GPS_NSW"
    if _matches_any(low, _GPS_SA):
        return "GPS_SA"
    if _matches_any(low, _APS):
        return "APS"
    if _matches_any(low, _OTHER_PRIVATE):
        return "other_private"
    return "unknown"


def build_school_affiliation(enrichment_csv: str, output_csv: str) -> pd.DataFrame:
    """
    Read ``enrichment_csv``, derive extracted_school + school_type for every
    row, write ``output_csv`` with OUTPUT_COLUMNS, and return the DataFrame.

    Rows whose pathway holds no school get a null extracted_school and a
    "unknown" school_type (classify_school(None) -> "unknown").
    """
    df = pd.read_csv(enrichment_csv)
    df["extracted_school"] = df["original_club"].apply(extract_school)
    df["school_type"] = df["extracted_school"].apply(classify_school)
    out = df[OUTPUT_COLUMNS]
    out.to_csv(output_csv, index=False)
    return out


def _main(argv: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify AFL draftees by elite-school affiliation"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    df = build_school_affiliation(args.input, args.output)
    print(f"[schools] {len(df)} rows -> {args.output}")
    print(df["school_type"].value_counts())
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
