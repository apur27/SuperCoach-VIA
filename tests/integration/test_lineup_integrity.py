"""Integration tier — are the shipped lineup CSVs actually player names?

Incident (BL-03, logged in docs/experiment-log.md): a regression in the
afltables lineup parser (`_extract_player_names`) read the wrong table column
and wrote the stats table's jersey-number column into `players` instead of the
name column. The output looked structurally valid — same schema, same row
count, semicolon-delimited — so nothing failed. It shipped from 2025 R3 through
2026, corrupting ~700 of 34,017 rows across 18 team files before anyone read a
`players` cell.

The parser was fixed (`_find_player_column` + `LineupParseError`) and the rows
were backfilled. This check is the tripwire that stops the corruption returning
silently: the unit tier proves the parser is correct on a fixture page, but only
a read of the real `data/lineups/` tree proves the parser's correct output is
what actually landed on disk.

Marked `integration`, so it runs in the weekly harness (Phase 0b) and is
excluded from the pre-commit tier.
"""
import glob
import os
import re
import sys

import pandas as pd
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

pytestmark = pytest.mark.integration

LINEUPS_DIR = os.path.join(_REPO_ROOT, "data", "lineups")

# A token is junk if it is a bare jersey number (optionally carrying an
# afltables sub-on/sub-off marker) or one of the stats-table footer labels that
# the broken parser swept up alongside the numbers.
_JERSEY_RE = re.compile(r"\s*\d+\s*[↑↓]?\s*\Z")
_FOOTER_TOKENS = {"Rushed", "Totals", "Opposition", ""}


def _is_garbage(players_field):
    """True when a `players` cell holds jersey numbers/footer labels, not names.

    Mirrors the detector used for the BL-03 backfill. Majority-vote rather than
    any-match, because a legitimate lineup can contain an odd unparsed token
    without the whole row being the wrong column.
    """
    tokens = str(players_field).split(";")
    junk = sum(
        1
        for t in tokens
        if _JERSEY_RE.fullmatch(t) or t.strip() in _FOOTER_TOKENS
    )
    return junk > len(tokens) / 2


def _lineup_files():
    files = sorted(glob.glob(os.path.join(LINEUPS_DIR, "team_lineups_*.csv")))
    assert files, f"no team_lineups_*.csv found under {LINEUPS_DIR}"
    return files


def test_no_jersey_number_rows_in_lineups():
    """Zero rows may hold the jersey-number column in `players` (BL-03)."""
    offenders = []
    total_rows = 0

    for path in _lineup_files():
        df = pd.read_csv(path)
        total_rows += len(df)
        if "players" not in df.columns:
            pytest.fail(f"{os.path.basename(path)} has no 'players' column")
        bad = df[df["players"].map(_is_garbage)]
        for _, row in bad.iterrows():
            offenders.append(
                f"  {os.path.basename(path)}: {row['year']} R{row['round_num']} "
                f"{row['team_name']} -> {str(row['players'])[:60]!r}"
            )

    assert not offenders, (
        f"BL-03 lineup corruption has returned: {len(offenders)} of {total_rows} "
        f"rows in data/lineups/ hold jersey numbers or stats-table footer labels "
        f"in the 'players' column instead of player names.\n"
        f"This is the afltables parser reading the wrong table column "
        f"(see _find_player_column in scrapers/game_scraper.py and the BL-03 "
        f"entry in docs/experiment-log.md). Delete the affected rows and "
        f"re-scrape the affected seasons; a plain delta run will NOT repair "
        f"them, because the scraper dedups on (year, date, round_num, "
        f"team_name) and skips keys already on disk.\n"
        + "\n".join(offenders[:25])
        + (f"\n  ... and {len(offenders) - 25} more" if len(offenders) > 25 else "")
    )


def test_lineup_rows_carry_plausible_player_names():
    """Every row's `players` cell must contain at least one 'Firstname Surname'.

    Weaker but independent of the jersey-number shape: catches a future parser
    that writes some *other* wrong column (venue, score, umpire) which would
    slip past the numeric detector above.
    """
    name_re = re.compile(r"[A-Za-z][A-Za-z'\-\.]*\s+[A-Za-z][A-Za-z'\-\.]+")
    offenders = []

    for path in _lineup_files():
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            tokens = [t.strip() for t in str(row["players"]).split(";") if t.strip()]
            if not any(name_re.fullmatch(t) for t in tokens):
                offenders.append(
                    f"  {os.path.basename(path)}: {row['year']} R{row['round_num']} "
                    f"{row['team_name']} -> {str(row['players'])[:60]!r}"
                )

    assert not offenders, (
        f"{len(offenders)} lineup rows contain no recognisable "
        f"'Firstname Surname' token — the parser is writing a non-name column "
        f"into 'players' (BL-03 class defect).\n" + "\n".join(offenders[:25])
    )


# ------------------------------------------------------------------ matches


def test_no_duplicate_fixtures_in_matches():
    """One row per fixture per season, whatever the venue string or round label.

    Incident (BL-03 follow-up): re-scraping 2025 appended the Gather Round games
    a second time because afltables publishes 'Barossa Park' where the file held
    'Barossa Oval', and the dedup key contained venue. The same class of miss
    relabelled the Cyclone Alfred fixture between R1 and R4.

    A fixture is identified by (date, unordered team pair) — venue and round
    label are both mutable, the two teams playing are not. `audit_match_rounds`
    cannot cover this: a duplicate makes a round look complete, never short.
    """
    offenders = []
    for path in sorted(glob.glob(os.path.join(_REPO_ROOT, "data", "matches", "matches_*.csv"))):
        df = pd.read_csv(path)
        cols = {"date", "team_1_team_name", "team_2_team_name"}
        if not cols.issubset(df.columns):
            continue
        fixture = df.apply(
            lambda r: (
                str(r["date"]),
                tuple(sorted([str(r["team_1_team_name"]), str(r["team_2_team_name"])])),
            ),
            axis=1,
        )
        dup_mask = fixture.duplicated(keep=False)
        for _, row in df[dup_mask].iterrows():
            offenders.append(
                f"  {os.path.basename(path)}: {row['date']} R{row['round_num']} "
                f"{row['team_1_team_name']} v {row['team_2_team_name']} "
                f"@ {row['venue']}"
            )

    assert not offenders, (
        f"{len(offenders)} duplicate fixture rows in data/matches/ — the same "
        f"game is stored more than once under differing venue spellings or "
        f"round labels, so season aggregates double-count it.\n"
        + "\n".join(offenders[:20])
    )
