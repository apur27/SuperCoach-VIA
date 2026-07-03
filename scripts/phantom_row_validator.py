"""Phantom-row validator: detect silently dropped (or duplicated) player game rows.

Motivation
----------
A drawn Grand Final and its replay share (team, year, round, opponent). The old
player-file dedup keyed on exactly those four columns and collapsed the two into
one, deleting a real game's stat line (Steele Sidebottom lost his 2010 drawn GF).
No gate caught it. This validator is that gate.

Two independent layers
----------------------
(a) ``check_counter_gaps`` -- DETERMINISTIC, zero false positives. afltables'
    ``games_played`` column is a per-player running career game counter that ticks
    only for games actually played. For a complete file it must equal {1..max}
    with no gaps and no duplicates. A missing counter == a deleted game row; a
    duplicated counter == a doubled row. Legitimately missed rounds do NOT create
    gaps (the counter simply doesn't advance for a game not played), so this layer
    never false-positives on injured/rested players.

(b) ``find_drawn_finals`` + ``check_drawn_final_consistency`` -- the counter-gap
    blind spot. If a drawn-final row was dropped AND the counter renumbered, layer
    (a) sees a contiguous counter and stays silent. This layer is scoped to the
    rare years whose ``matches_<year>.csv`` contains a drawn final + replay (two
    rows, same round label, same two teams, different dates, earlier one a draw on
    total points). For a finalist club's player it flags:
      * WARNING  -- >2 rows for the drawn round, or a repeated result (real dup);
      * REVIEW   -- exactly one row (draw-only or replay-only): undecidable from
                    the file alone, so it is surfaced for an external afltables
                    check rather than hard-failed (a squad genuinely changes
                    between the draw and the replay, so a single row is not by
                    itself a bug).

CLI: ``python scripts/phantom_row_validator.py`` scans every active player file
and prints the counter gaps and finals reviews it finds.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config  # noqa: E402

# matches_<year>.csv round labels -> the finals codes used in player files.
_FINALS_ROUND_CODES = {
    "Qualifying Final": "QF",
    "Elimination Final": "EF",
    "Semi Final": "SF",
    "Preliminary Final": "PF",
    "Grand Final": "GF",
}


def _counter_series(series: pd.Series) -> pd.Series:
    """games_played -> clean numeric, stripping ↑/↓ debut/milestone arrows."""
    cleaned = (
        series.astype(str)
        .str.replace("↑", "", regex=False)
        .str.replace("↓", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


# --------------------------------------------------------------------------
# Layer (a): deterministic counter-gap check
# --------------------------------------------------------------------------

def check_counter_gaps(player_csv_path: str) -> Dict[str, Any]:
    """Return the games_played counter integrity of one player file.

    {player, counter_max, row_count, missing:[...], duplicated:[...], ok:bool}
    ``ok`` is True iff the counter set is exactly {1..counter_max} with no dups.
    Unreadable/empty files return ok=True with a note (never raises).
    """
    player = os.path.basename(player_csv_path).replace("_performance_details.csv", "")
    result: Dict[str, Any] = {
        "player": player,
        "counter_max": None,
        "row_count": 0,
        "missing": [],
        "duplicated": [],
        "ok": True,
    }
    try:
        df = pd.read_csv(player_csv_path, low_memory=False)
    except Exception as e:  # unreadable file -- report, do not crash a scan
        result["note"] = f"unreadable: {e}"
        return result

    result["row_count"] = len(df)
    if "games_played" not in df.columns or df.empty:
        result["note"] = "no games_played column or empty file"
        return result

    counters = _counter_series(df["games_played"]).dropna().astype(int)
    if counters.empty:
        result["note"] = "no numeric counters"
        return result

    counter_max = int(counters.max())
    counts = counters.value_counts()
    expected = set(range(1, counter_max + 1))
    have = set(counters.tolist())
    result["counter_max"] = counter_max
    result["missing"] = sorted(expected - have)
    result["duplicated"] = sorted(int(c) for c, n in counts.items() if n > 1)
    result["ok"] = not result["missing"] and not result["duplicated"]
    return result


def gaps_in_season(player_csv_path: str, season: int) -> List[int]:
    """Missing counters attributable to `season`.

    A deleted interior row leaves a gap `m` in the games_played counter. The
    game that would have carried `m` sits immediately before the next surviving
    counter, so we attribute `m` to the season of that next surviving counter.
    At a season boundary this errs toward the later season -- the safe direction
    for a current-season abort gate. Returns [] for a clean file.
    """
    gaps = check_counter_gaps(player_csv_path)
    missing = gaps["missing"]
    if not missing:
        return []
    try:
        df = pd.read_csv(player_csv_path, low_memory=False)
    except Exception:
        return []
    if "games_played" not in df.columns or "year" not in df.columns:
        return []
    counters = _counter_series(df["games_played"])
    years = pd.to_numeric(df["year"], errors="coerce")
    counter_year = {
        int(c): int(y)
        for c, y in zip(counters, years)
        if pd.notna(c) and pd.notna(y)
    }
    surviving = sorted(counter_year)
    out: List[int] = []
    for m in missing:
        nxt = next((c for c in surviving if c > m), None)
        if nxt is not None and counter_year[nxt] == season:
            out.append(m)
    return out


# --------------------------------------------------------------------------
# Layer (b): drawn-final cross-check
# --------------------------------------------------------------------------

def _total_points(row: pd.Series, side: int) -> Optional[float]:
    g = pd.to_numeric(row.get(f"team_{side}_final_goals"), errors="coerce")
    b = pd.to_numeric(row.get(f"team_{side}_final_behinds"), errors="coerce")
    if pd.isna(g) or pd.isna(b):
        return None
    return float(g) * 6 + float(b)


def find_drawn_finals(matches_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Identify drawn finals in a matches_<year>.csv frame.

    A drawn final is a round label that appears on exactly two rows, between the
    same two teams, on different dates, where the EARLIER game is a draw on total
    points. Returns one descriptor per drawn final:
      {round_label, round_code, teams: frozenset, year, draw_date, replay_date}
    """
    out: List[Dict[str, Any]] = []
    if "round_num" not in matches_df.columns:
        return out
    for label, grp in matches_df.groupby("round_num"):
        code = _FINALS_ROUND_CODES.get(str(label).strip())
        if code is None or len(grp) != 2:
            continue
        teamsets = [
            frozenset({str(r["team_1_team_name"]).strip(), str(r["team_2_team_name"]).strip()})
            for _, r in grp.iterrows()
        ]
        if teamsets[0] != teamsets[1]:
            continue
        grp = grp.sort_values("date")
        first = grp.iloc[0]
        t1, t2 = _total_points(first, 1), _total_points(first, 2)
        if t1 is None or t2 is None or t1 != t2:
            continue  # earlier game not a draw -> not a drawn final
        year = None
        if "year" in grp.columns:
            year = pd.to_numeric(grp.iloc[0]["year"], errors="coerce")
            year = int(year) if pd.notna(year) else None
        out.append(
            {
                "round_label": str(label).strip(),
                "round_code": code,
                "teams": teamsets[0],
                "year": year,
                "draw_date": str(first["date"]),
                "replay_date": str(grp.iloc[1]["date"]),
            }
        )
    return out


def check_drawn_final_consistency(
    player_df: pd.DataFrame, year: int, drawn_finals: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Flag drawn-final row inconsistencies for one player, scoped to `year`.

    Only players from a finalist club are considered. For the drawn round:
      * >2 rows or a repeated result -> WARNING (a real duplication);
      * exactly 1 row               -> REVIEW (draw-only vs replay-only vs a
                                        dropped+renumbered counterpart cannot be
                                        decided from the file; needs afltables).
    Two rows with distinct results (D + W/L) is the clean "played both" case.
    """
    issues: List[Dict[str, Any]] = []
    if player_df.empty or "team" not in player_df.columns:
        return issues
    yr = pd.to_numeric(player_df["year"], errors="coerce")
    scoped = player_df[yr == year]
    if scoped.empty:
        return issues
    player_teams = {str(t).strip() for t in scoped["team"].dropna()}

    for final in drawn_finals:
        if final.get("year") not in (None, year):
            continue
        if not (player_teams & final["teams"]):
            continue
        code = final["round_code"]
        rows = scoped[scoped["round"].astype(str).str.strip() == code]
        n = len(rows)
        if n == 0:
            continue
        results = [str(r).strip() for r in rows["result"]]
        if n > 2 or len(set(results)) < len(results):
            issues.append(
                {
                    "severity": "WARNING",
                    "type": "DUPLICATE_FINAL",
                    "round_code": code,
                    "year": year,
                    "rows": n,
                    "results": results,
                }
            )
        elif n == 1:
            issues.append(
                {
                    "severity": "REVIEW",
                    "type": "SINGLE_FINAL_ROW",
                    "round_code": code,
                    "year": year,
                    "rows": n,
                    "results": results,
                }
            )
    return issues


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def load_drawn_finals(matches_dir: str) -> Dict[int, List[Dict[str, Any]]]:
    """Map year -> drawn-final descriptors across every matches_<year>.csv."""
    by_year: Dict[int, List[Dict[str, Any]]] = {}
    for path in glob.glob(os.path.join(matches_dir, "matches_*.csv")):
        try:
            mdf = pd.read_csv(path, dtype=str)
        except Exception:
            continue
        for final in find_drawn_finals(mdf):
            yr = final.get("year")
            if yr is not None:
                by_year.setdefault(yr, []).append(final)
    return by_year


def validate_player_file(
    player_csv_path: str,
    drawn_finals_by_year: Optional[Dict[int, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Run both layers on one player file.

    Returns {player, counter: <layer a dict>, finals: [<layer b issues>]}.
    Layer (b) runs only when `drawn_finals_by_year` is supplied (the caller loads
    it once and reuses it across a scan).
    """
    counter = check_counter_gaps(player_csv_path)
    finals: List[Dict[str, Any]] = []
    if drawn_finals_by_year:
        try:
            df = pd.read_csv(player_csv_path, low_memory=False)
        except Exception:
            df = pd.DataFrame()
        if not df.empty and "year" in df.columns:
            years = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int).unique()
            for yr in years:
                if yr in drawn_finals_by_year:
                    finals.extend(
                        check_drawn_final_consistency(df, int(yr), drawn_finals_by_year[yr])
                    )
    return {"player": counter["player"], "counter": counter, "finals": finals}


def scan_all_players(player_dir: str, matches_dir: str) -> List[Dict[str, Any]]:
    """Scan every active player file; return only the ones with a finding."""
    drawn = load_drawn_finals(matches_dir)
    findings: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(player_dir, "*_performance_details.csv"))):
        res = validate_player_file(path, drawn)
        if not res["counter"]["ok"] or res["finals"]:
            findings.append(res)
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--player-dir", default=config.PLAYER_DATA_DIR,
        help="directory of *_performance_details.csv files",
    )
    parser.add_argument(
        "--matches-dir", default=config.MATCHES_DIR,
        help="directory of matches_<year>.csv files (for the drawn-final layer)",
    )
    parser.add_argument(
        "--file", default=None,
        help="validate a single player file instead of scanning all",
    )
    args = parser.parse_args()

    if args.file:
        drawn = load_drawn_finals(args.matches_dir)
        res = validate_player_file(args.file, drawn)
        c = res["counter"]
        print(f"{res['player']}: counter ok={c['ok']} missing={c['missing']} "
              f"duplicated={c['duplicated']} finals={res['finals']}")
        return 0 if c["ok"] and not any(f["severity"] == "WARNING" for f in res["finals"]) else 1

    findings = scan_all_players(args.player_dir, args.matches_dir)
    gap_hits = [f for f in findings if not f["counter"]["ok"]]
    review_hits = [f for f in findings if f["finals"]]
    print(f"Scanned player files. counter-gap hits: {len(gap_hits)}, "
          f"finals-review hits: {len(review_hits)}")
    for f in gap_hits:
        c = f["counter"]
        print(f"  [GAP] {f['player']}: missing={c['missing']} duplicated={c['duplicated']} "
              f"(rows={c['row_count']} counter_max={c['counter_max']})")
    for f in review_hits:
        for issue in f["finals"]:
            print(f"  [{issue['severity']}] {f['player']}: {issue['type']} "
                  f"{issue['round_code']} {issue['year']} results={issue['results']}")
    return 1 if gap_hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
