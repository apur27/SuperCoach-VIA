"""
AFL national draft history scraper.

Source: Wikipedia (``https://en.wikipedia.org/wiki/{year}_AFL_draft``).
afltables.com has no draft pages (404), so Wikipedia is the ground truth.

The national draft table is identified by HEADER NAME, not table index, because
the wikitable order varies year to year. We require a wikitable whose header row
contains all of "Round", "Pick" and "Player". Column meaning is also read by
header name (not position) because the column order itself drifts across years:

  2004 headers: Round, Pick, Player, Recruited from, Club
  2022 headers: Round, Pick, Player, Club, Recruited from, Notes, Club, League

Parsing rules (from the task brief):
  - "Priority" in the Round column -> round = 0.
  - Rows whose Pick cell is not an integer are sub-headers / separators -> skip.
  - Footnote markers ("[1]", "[2]") are stripped from player names.

Output: ``data/drafts/afl_draft_history.csv`` with columns
  year, round, pick, club, player_name, recruited_from
deduplicated on (year, pick): re-running a year replaces that year's rows in
place rather than appending duplicates.

Style mirrors ``scrapers/game_scraper.py``: a requests-based fetch with an
explicit User-Agent, a 0.5s inter-request delay, and delta-friendly CSV writes.
"""

import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

WIKI_URL = "https://en.wikipedia.org/wiki/{year}_AFL_draft"
USER_AGENT = (
    "SuperCoach-VIA draft scraper (https://github.com/; research/non-commercial)"
)
REQUEST_DELAY_S = 0.5
DEFAULT_OUT_PATH = "data/drafts/afl_draft_history.csv"
OUTPUT_COLUMNS = ["year", "round", "pick", "club", "player_name", "recruited_from"]

_FOOTNOTE_RE = re.compile(r"\[[^\]]*\]")


# ---------------------------------------------------------------------------
# Pure parsing helpers (no network -- directly unit-tested)
# ---------------------------------------------------------------------------

def clean_player_name(raw: str) -> str:
    """Strip footnote markers ("[1]") and collapse whitespace in a name."""
    no_notes = _FOOTNOTE_RE.sub("", raw)
    return " ".join(no_notes.split()).strip()


def player_key(raw: str) -> str:
    """
    Normalise a (possibly footnoted) name to ``surname_firstname`` lowercase.

    The last whitespace-token is treated as the surname and everything before
    it as the first-name run, matching the repo's existing player-file naming
    convention (``<lastname>_<firstname>_...``).
    """
    name = clean_player_name(raw).lower()
    parts = name.split()
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    surname = parts[-1]
    first = "_".join(parts[:-1])
    return f"{surname}_{first}"


def _header_names(table: BeautifulSoup) -> List[str]:
    """Header labels of a wikitable, taken from its first row's <th> cells."""
    first_row = table.find("tr")
    if first_row is None:
        return []
    return [th.get_text(" ", strip=True) for th in first_row.find_all("th")]


def _expand_rows(table: BeautifulSoup) -> List[List[str]]:
    """
    Render an HTML table into a dense grid of cell texts that honours rowspan
    and colspan, so every logical row has a value in every column position.

    This is essential for modern draft pages (2018+) where the Round number is
    written once per round via ``rowspan`` and the later picks of the round omit
    the Round <td>. Without expansion those rows' cells shift left, the Pick
    column lands on a name, and the whole round after pick 1 is silently dropped
    (the "2-6 picks" data-loss bug). A rowspanned cell's text is repeated down
    into each row it covers, keeping every column aligned.
    """
    grid: List[List[str]] = []
    # pending[col] = (text, remaining_rows) for a cell still spanning downward.
    pending: Dict[int, "tuple[str, int]"] = {}

    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        row: List[str] = []
        col = 0
        ci = 0
        # Build the row left-to-right, filling carried-over rowspan cells first.
        while ci < len(cells) or any(rem > 0 for _, rem in pending.values()):
            if col in pending and pending[col][1] > 0:
                text, rem = pending[col]
                row.append(text)
                pending[col] = (text, rem - 1) if rem - 1 > 0 else ("", 0)
                col += 1
                continue
            if ci >= len(cells):
                break
            cell = cells[ci]
            ci += 1
            text = cell.get_text(" ", strip=True)
            try:
                colspan = int(cell.get("colspan", 1))
            except (TypeError, ValueError):
                colspan = 1
            try:
                rowspan = int(cell.get("rowspan", 1))
            except (TypeError, ValueError):
                rowspan = 1
            for _ in range(max(1, colspan)):
                row.append(text)
                if rowspan > 1:
                    pending[col] = (text, rowspan - 1)
                col += 1
        grid.append(row)
    return grid


# Header labels that denote the round-number column. "Round" is the modern
# label; "Rd." / "Rd" is the 2019 abbreviation. Single-round-era tables (1990s)
# omit this column entirely -- see _ROUND_HEADERS use in parse_draft_table.
_ROUND_HEADERS = {"round", "rd.", "rd"}

# Header labels for the DRAFTING club. This column is named differently by era:
# "Club" (modern), "Recruited by" (1993), "Recruited to" (2000), "Drafted to"
# (2012). It must never be confused with "Recruited from" (the pathway club).
# Order matters: the first header that matches wins, so "Club" is preferred.
_CLUB_HEADERS = ["club", "recruited to", "drafted to", "recruited by"]


# Header labels for the pick-number column. "Pick" is standard; "#" is the
# 1990/1991 variant.
_PICK_HEADERS = {"pick", "#"}


def _pick_index(headers: List[str]) -> Optional[int]:
    """Index of the pick-number column (handles the 1990/1991 '#' header)."""
    for i, h in enumerate(headers):
        if h.lower() in _PICK_HEADERS:
            return i
    return None


def _is_draft_table(table: BeautifulSoup) -> bool:
    """A draft table has a Pick column and a Player header (Round is optional)."""
    headers = _header_names(table)
    lower = {h.lower() for h in headers}
    return _pick_index(headers) is not None and "player" in lower


def find_national_draft_table(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    """
    Return the table holding the NATIONAL draft, identified by the section
    heading immediately preceding it containing "national draft".

    Detection is by heading text, not table index, because modern pages (2018+)
    carry several Pick/Player tables -- mid-season rookie draft, national draft,
    pre-season draft, rookie draft -- in that order, so "first match" grabs the
    wrong one and mixes draft types. The table itself only needs Pick + Player
    headers; the Round column is optional (the 1990s single-round national draft
    has no Round column).

    Returns None when the page has no "national draft" section (e.g. pages that
    only document a rookie/mid-season draft), so the caller skips the year rather
    than silently scraping a different draft type.
    """
    candidates = soup.find_all("table", class_="wikitable")
    if not candidates:
        candidates = soup.find_all("table")
    for table in candidates:
        if not _is_draft_table(table):
            continue
        heading = table.find_previous(["h2", "h3", "h4"])
        heading_text = heading.get_text(" ", strip=True).lower() if heading else ""
        if "national draft" in heading_text:
            return table
    return None


def _column_index(headers: List[str], name: str) -> Optional[int]:
    """First index whose header equals ``name`` (case-insensitive)."""
    target = name.lower()
    for i, h in enumerate(headers):
        if h.lower() == target:
            return i
    return None


def parse_draft_table(table: BeautifulSoup, year: int) -> List[Dict[str, Any]]:
    """
    Parse a national-draft wikitable into row dicts.

    Columns are located by header name (Round / Pick / Player / Club /
    Recruited from), so column-order drift across years does not matter. Rows
    whose Pick cell is not an integer are skipped (sub-headers / separators).
    "Priority" in the Round column maps to round=0.
    """
    headers = _header_names(table)
    if not headers:
        return []

    # Round column may be "Round", "Rd." or "Rd"; it is absent in the 1990s
    # single-round national draft, in which case every pick defaults to round 1.
    i_round = None
    for name in _ROUND_HEADERS:
        i_round = next(
            (i for i, h in enumerate(headers) if h.lower() == name), None
        )
        if i_round is not None:
            break
    i_pick = _pick_index(headers)
    i_player = _column_index(headers, "Player")
    i_recruited = _column_index(headers, "Recruited from")
    # Drafting club: try each era alias in preference order.
    i_club = None
    for name in _CLUB_HEADERS:
        i_club = next((i for i, h in enumerate(headers) if h.lower() == name), None)
        if i_club is not None:
            break
    if i_pick is None or i_player is None:
        return []

    rows: List[Dict[str, Any]] = []
    for texts in _expand_rows(table):
        # Skip the header row and any row too short to hold the Pick column.
        if len(texts) <= i_pick:
            continue

        pick_raw = texts[i_pick]
        if not pick_raw.isdigit():
            continue  # sub-header / separator / blank row
        pick = int(pick_raw)

        # Round: "Priority" -> 0; numeric -> int; anything else -> mark round NA
        # rather than guess (defensive against "Pre-season"/"Rookie" rows). When
        # the table has no Round column (1990s single-round draft) default to 1.
        if i_round is None:
            round_val: Optional[int] = 1
        else:
            round_val = None
            if len(texts) > i_round:
                round_raw = texts[i_round].strip()
                if round_raw.lower().startswith("priority"):
                    round_val = 0
                elif round_raw.isdigit():
                    round_val = int(round_raw)

        player_name = clean_player_name(texts[i_player]) if len(texts) > i_player else ""
        if not player_name:
            continue

        club = ""
        if i_club is not None and len(texts) > i_club:
            club = clean_player_name(texts[i_club])

        recruited_from = ""
        if i_recruited is not None and len(texts) > i_recruited:
            recruited_from = clean_player_name(texts[i_recruited])

        rows.append({
            "year": int(year),
            "round": round_val,
            "pick": pick,
            "club": club,
            "player_name": player_name,
            "recruited_from": recruited_from,
        })
    return rows


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class DraftScraper:
    def __init__(self, out_path: str = DEFAULT_OUT_PATH):
        self.out_path = out_path

    def _fetch_soup(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch a Wikipedia page and return parsed soup. Returns None on 404 or
        any request error (so a missing-year page is skipped, not fatal). This
        is the single network boundary -- tests patch it.
        """
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            if resp.status_code == 404:
                print(f"[draft] {url} -> 404; skipping")
                return None
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            print(f"[draft] error fetching {url}: {e}; skipping")
            return None

    def scrape_year(self, year: int) -> int:
        """
        Scrape one year's national draft and merge into the output CSV,
        deduplicating on (year, pick). Returns the number of rows scraped for
        the year (0 if the page or table was missing).
        """
        url = WIKI_URL.format(year=year)
        soup = self._fetch_soup(url)
        if soup is None:
            return 0

        table = find_national_draft_table(soup)
        if table is None:
            print(f"[draft] {year}: no national draft table found; skipping")
            return 0

        rows = parse_draft_table(table, year)
        if not rows:
            print(f"[draft] {year}: table found but no parseable pick rows; skipping")
            return 0

        new_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        self._merge_and_write(new_df, year)
        print(f"[draft] {year}: {len(new_df)} picks scraped")
        return len(new_df)

    def _merge_and_write(self, new_df: pd.DataFrame, year: int) -> None:
        """
        Replace ``year``'s rows in the on-disk CSV with ``new_df`` (delta mode):
        drop any existing rows for that year, append the fresh ones, then drop
        duplicate (year, pick) keeping the latest. This makes re-runs idempotent.
        """
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)

        if os.path.exists(self.out_path):
            existing = pd.read_csv(self.out_path)
            existing = existing[existing["year"] != year]
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df

        combined = combined.drop_duplicates(subset=["year", "pick"], keep="last")
        combined = combined.sort_values(["year", "pick"]).reset_index(drop=True)
        # Write round/pick as nullable integers so a handful of missing rounds
        # don't coerce the whole column to "1.0" floats in the CSV.
        combined["round"] = combined["round"].astype("Int64")
        combined["pick"] = combined["pick"].astype("Int64")
        combined.to_csv(self.out_path, index=False)

    def scrape_all(self, start: int = 1986, end: int = 2025) -> int:
        """
        Scrape every year in [start, end], honouring the inter-request delay.
        Returns the total number of rows in the output CSV after the run.
        """
        total = 0
        for year in range(start, end + 1):
            total += self.scrape_year(year)
            time.sleep(REQUEST_DELAY_S)
        if os.path.exists(self.out_path):
            n = len(pd.read_csv(self.out_path))
            print(f"[draft] scrape_all complete: {n} total rows on disk")
            return n
        return total


def _main(argv: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="AFL national draft scraper (Wikipedia)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--year", type=int, help="Scrape a single year")
    group.add_argument("--all", action="store_true", help="Scrape 1986-2025")
    parser.add_argument("--start", type=int, default=1986)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    args = parser.parse_args(argv)

    scraper = DraftScraper(out_path=args.out)
    if args.all:
        scraper.scrape_all(start=args.start, end=args.end)
    else:
        scraper.scrape_year(args.year)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
