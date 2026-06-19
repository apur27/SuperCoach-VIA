"""
AFL squad-builder name/draft resolution layer.

This module resolves the players in a club's selected-squad (by jersey number)
back to their draft pathway by matching display names against the draft-history
CSVs. The matching is deliberately layered -- exact namekey, curated alias, then
a guarded loose (first-initial + surname) fallback -- so the common spelling and
nickname drifts ("Matt"/"Matthew", "Cam"/"Cameron") resolve without admitting
false positives ("Jeremy"/"Jarrod", "Sam"/"Sid").

THE COLLISION FIX (``pick_best``)
---------------------------------
Different players can share a namekey across clubs and eras, e.g.

  baileywilliams -> Bailey Williams / Western Bulldogs 2015 (pick 48)
                    Bailey Williams / West Coast       2018 (pick 35)
  willhayes      -> Will Hayes / Western Bulldogs 2018 (pick 78)
                    Will Hayes / Collingwood       2024 (pick 56)
  sambutler      -> Sam Butler / West Coast Eagles 2003 (pick 20)
                    Sam Butler / Hawthorn          2021 (pick 23)

The previous resolver always returned the *earliest* draft entry, so looking up
the West Coast Bailey Williams returned the Bulldogs record. ``pick_best`` takes
the club being looked up (``cteam``) and prefers the record whose canonical club
matches it, falling back to "earliest" only when no club matches.

Pure helpers (``namekey`` ... ``lookup_draft``) take no I/O and are unit-tested
directly; ``load_draft_index`` is the only file boundary.
"""

import csv
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Curated alias: perf-display namekey -> draft-file namekey. These are the
# nickname/Dutch-prefix cases that neither exact nor loose matching can bridge.
ALIAS = {
    "camrayner": "cameronrayner",
    "jordangoey": "jordandegoey",
    "jacobrooyen": "jacobvanrooyen",
}

# Canonical team key (collapse spelling variants; never collapse Port Adelaide
# into Adelaide).
TEAM_CANON = {
    "sydney": "sydney", "sydney swans": "sydney",
    "brisbane lions": "brisbane", "brisbane": "brisbane",
    "greater western sydney": "gws", "gws": "gws", "gws giants": "gws",
    "fremantle": "fremantle",
    "collingwood": "collingwood",
    "geelong": "geelong", "geelong cats": "geelong",
    "gold coast": "goldcoast", "gold coast suns": "goldcoast",
    "port adelaide": "portadelaide",
    "hawthorn": "hawthorn",
    "adelaide": "adelaide", "adelaide crows": "adelaide",
    "western bulldogs": "westernbulldogs", "footscray": "westernbulldogs",
    "carlton": "carlton",
    "melbourne": "melbourne",
    "st kilda": "stkilda",
    "essendon": "essendon",
    "north melbourne": "northmelbourne", "kangaroos": "northmelbourne",
    "richmond": "richmond",
    "west coast": "westcoast", "west coast eagles": "westcoast",
}


# ---------------------------------------------------------------------------
# Pure normalisation / matching helpers
# ---------------------------------------------------------------------------

def namekey(s: str) -> str:
    """Strip every non-alpha character and lowercase (``Bailey Williams`` ->
    ``baileywilliams``)."""
    return re.sub(r"[^a-z]", "", s.lower())


def loosekey(s: str) -> str:
    """First-initial + full surname (``Matthew Crouch`` -> ``mcrouch``).

    Falls back to :func:`namekey` for single-token names and returns ``""`` for
    empty input.
    """
    parts = [p for p in re.split(r"\s+", s.strip()) if p]
    if not parts:
        return ""
    first = re.sub(r"[^a-z]", "", parts[0].lower())
    last = re.sub(r"[^a-z]", "", parts[-1].lower())
    return (first[:1] + last) if first and last else namekey(s)


def parse_jersey(x) -> Optional[int]:
    """Parse a jersey number from a cell that may be an int, a float-string
    (``"7.0"``), empty, or non-numeric. Returns ``None`` when not a number."""
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def canon_team(t: str) -> str:
    """Canonical club key. Known spelling variants collapse via
    :data:`TEAM_CANON`; unknown clubs fall back to their alpha-only form. Port
    Adelaide is never collapsed into Adelaide."""
    return TEAM_CANON.get(t.strip().lower(), re.sub(r"[^a-z]", "", t.lower()))


def _words(s: str) -> List[str]:
    return [re.sub(r"[^a-z]", "", w.lower()) for w in re.split(r"\s+", s.strip()) if w]


def firstname_compat(a: str, b: str) -> bool:
    """True when two first names are prefix- or near-prefix-compatible.

    Accepts exact (``matt``/``matt``), prefix (``matt``/``matthew``) and a
    common-prefix of length >= 3 (``lachie``/``lachlan``, ``harry``/``harrison``).
    Rejects unrelated names (``jack``/``sam``, ``sam``/``sid``).
    """
    if not a or not b:
        return False
    if a == b or a.startswith(b) or b.startswith(a):
        return True
    # common-prefix length >= 3 catches Lachie/Lachlan, Harry/Harrison, etc.
    n = 0
    for x, y in zip(a, b):
        if x == y:
            n += 1
        else:
            break
    return n >= 3


def loose_ok(display: str, cand_name: str) -> bool:
    """A loose (first-initial + surname) match is only safe when the surname
    matches exactly AND the first names are prefix/near-prefix compatible.

    Rejects Jeremy/Jarrod, Sam/Sid, Charlie/Clay, Jack/Jess while keeping
    Matt/Matthew, Lachie/Lachlan.
    """
    dw, cw = _words(display), _words(cand_name)
    if not dw or not cw:
        return False
    if dw[-1] != cw[-1]:
        return False
    return firstname_compat(dw[0], cw[0])


def _year(r: dict) -> int:
    try:
        return int(r.get("year", 0) or 0)
    except (ValueError, TypeError):
        return 0


def pick_best(recs: List[dict], cteam: Optional[str] = None,
              club_col: str = "club") -> dict:
    """Choose the right record among same-namekey collisions.

    When ``cteam`` is given, prefer records whose canonical club matches it
    (resolving the cross-club same-name collision); among those (or, if none
    match, among all records) return the earliest by year -- the original
    pathway into the league.
    """
    pool = recs
    if cteam is not None:
        matched = [r for r in recs if canon_team(r.get(club_col, "")) == cteam]
        if matched:
            pool = matched
    return sorted(pool, key=_year)[0]


def lookup_draft(display: str, nk: str, by_nk: dict, by_lk: dict,
                 name_col: str = "player_name", cteam: Optional[str] = None,
                 club_col: str = "club") -> Tuple[Optional[dict], str]:
    """Resolve a display name to its draft record.

    Returns ``(record, how)`` where ``how`` is one of ``exact``/``alias``/
    ``loose``/``none``. Collisions are resolved against ``cteam`` via
    :func:`pick_best`.
    """
    if nk in by_nk:
        return (pick_best(by_nk[nk], cteam, club_col), "exact")
    al = ALIAS.get(nk)
    if al and al in by_nk:
        return (pick_best(by_nk[al], cteam, club_col), "alias")
    lk = loosekey(display)
    cands = [r for r in by_lk.get(lk, []) if loose_ok(display, r.get(name_col, ""))]
    if cands:
        return (pick_best(cands, cteam, club_col), "loose")
    return (None, "none")


# ---------------------------------------------------------------------------
# File boundary
# ---------------------------------------------------------------------------

def load_draft_index(path: str, name_col: str) -> Tuple[dict, dict]:
    """Read a draft CSV into ``(by_nk, by_lk)`` dicts keyed by :func:`namekey`
    and :func:`loosekey` of the ``name_col`` column. Each value is the list of
    rows sharing that key (so same-name collisions are preserved for
    :func:`pick_best` to resolve)."""
    by_nk: Dict[str, list] = defaultdict(list)
    by_lk: Dict[str, list] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            nm = row.get(name_col, "")
            by_nk[namekey(nm)].append(row)
            by_lk[loosekey(nm)].append(row)
    return by_nk, by_lk


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Resolve a player display name to its AFL draft record."
    )
    parser.add_argument("--draft-csv", required=True,
                        help="Path to a draft-history CSV (year,...,player_name,...)")
    parser.add_argument("--name", required=True, help="Player display name to resolve")
    parser.add_argument("--team", default=None,
                        help="Club being looked up (collision tie-breaker)")
    parser.add_argument("--name-col", default="player_name")
    parser.add_argument("--club-col", default="club")
    args = parser.parse_args(argv)

    by_nk, by_lk = load_draft_index(args.draft_csv, args.name_col)
    cteam = canon_team(args.team) if args.team else None
    rec, how = lookup_draft(args.name, namekey(args.name), by_nk, by_lk,
                            name_col=args.name_col, cteam=cteam,
                            club_col=args.club_col)
    if rec is None:
        print(f"{args.name}: no match")
        return 1
    print(f"{args.name}: [{how}] {rec.get(args.name_col)} "
          f"{rec.get('year')} #{rec.get('pick')} ({rec.get(args.club_col)})")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
