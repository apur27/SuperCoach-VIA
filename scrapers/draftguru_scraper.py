"""
DraftGuru draft-pathway enrichment scraper.

Source: DraftGuru (``https://www.draftguru.com.au/years/{year}``). Each year's
page carries a single static-HTML table covering every draft type (National,
Rookie, Preseason, ...). We keep only the National-draft rows and extract the
"Original Club" pathway field (School / TAC-Cup-club lineage) plus Grade, Games
and Goals, to enrich ``data/drafts/afl_draft_history.csv`` on (year, pick).

Page columns (confirmed in the brief, in order):
  Pick, Draft, #, Club, Signing, Player, Age, Height, Weight,
  Original Club, Grade, Games, Goals, Coaches, Brownlow, Awards

Columns are located by HEADER NAME, not position, so a column-order change on
DraftGuru's side surfaces as missing data (caught in the per-year report) rather
than silently shifting every field.

Output: ``data/drafts/draftguru_enrichment.csv`` with columns
  year, pick, player_name, original_club, grade, games, goals
deduplicated on (year, pick): re-running a year replaces that year's rows.

Style mirrors ``scrapers/draft_scraper.py``: a requests-based fetch behind a
single ``_fetch_soup`` network boundary (the only thing tests patch), an
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

YEAR_URL = "https://www.draftguru.com.au/years/{year}"
USER_AGENT = (
    "SuperCoach-VIA draftguru scraper (https://github.com/; research/non-commercial)"
)
REQUEST_DELAY_S = 0.5
DEFAULT_OUT_PATH = "data/drafts/draftguru_enrichment.csv"
OUTPUT_COLUMNS = [
    "year", "pick", "player_name", "original_club", "grade", "games", "goals",
]

_FOOTNOTE_RE = re.compile(r"\[[^\]]*\]")
# Empty-ish cell markers for Grade and numeric fields: blank, dash, en/em dash.
_EMPTY_CELLS = {"", "-", "–", "—", "n/a", "na"}
# Unicode whitespace DraftGuru embeds in cells: non-breaking and zero-width
# spaces. These survive a naive .strip()/.split() (zero-width isn't whitespace),
# so strip them explicitly before normalising.
_ZERO_WIDTH = "​‌‍﻿"
_ZW_RE = re.compile(f"[{_ZERO_WIDTH}]")


# ---------------------------------------------------------------------------
# Pure parsing helpers (no network -- directly unit-tested)
# ---------------------------------------------------------------------------

def _normalise_ws(raw: str) -> str:
    """Drop zero-width chars and collapse all unicode whitespace (incl. nbsp)."""
    no_zw = _ZW_RE.sub("", raw or "")
    return " ".join(no_zw.split()).strip()


def clean_player_name(raw: str) -> str:
    """Strip footnote markers ("[1]") and collapse whitespace in a name."""
    return _normalise_ws(_FOOTNOTE_RE.sub("", raw or ""))


def _to_int(raw: str) -> int:
    """
    Coerce a cell to int; blank/dash/non-numeric -> 0.

    Takes only the FIRST integer token. DraftGuru's Games cell is formatted
    "102 (83)" (total games, finals in parentheses); naively stripping all
    non-digits would glue the two numbers into "10283", so we extract the
    leading run of digits and discard any parenthetical secondary figure.
    """
    text = _normalise_ws(raw)
    if text.lower() in _EMPTY_CELLS:
        return 0
    m = re.search(r"\d+", text.replace(",", ""))
    return int(m.group()) if m else 0


def _clean_grade(raw: str) -> str:
    """Grade verbatim, but blank/dash cells become an empty string."""
    text = (raw or "").strip()
    return "" if text.lower() in _EMPTY_CELLS else text


def _header_names(table: BeautifulSoup) -> List[str]:
    """Header labels of a table, taken from its first <tr> containing <th>s."""
    for tr in table.find_all("tr"):
        ths = tr.find_all("th")
        if ths:
            return [th.get_text(" ", strip=True) for th in ths]
    return []


def _column_index(headers: List[str], name: str) -> Optional[int]:
    """First index whose header equals ``name`` (case-insensitive)."""
    target = name.lower()
    for i, h in enumerate(headers):
        if h.lower() == target:
            return i
    return None


def _hash_index(headers: List[str]) -> Optional[int]:
    """
    Index of the pick-number "#" column. On the live page the header renders as
    "# ↧" (a sort-arrow suffix), so match on the leading "#" rather than equality.
    """
    for i, h in enumerate(headers):
        if h.strip().startswith("#"):
            return i
    return None


def _find_table(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    """Return the first table that carries both a Player and Draft header."""
    for table in soup.find_all("table"):
        headers = _header_names(table)
        lower = {h.lower() for h in headers}
        if "player" in lower and "draft" in lower:
            return table
    return None


def parse_year_table(soup: BeautifulSoup, year: int) -> List[Dict[str, Any]]:
    """
    Parse one DraftGuru year page into National-draft row dicts.

    Columns are located by header name (Draft / # / Player / Original Club /
    Grade / Games / Goals). Only rows whose Draft column equals "National" are
    kept. The pick number is read from the "#" column (pick-within-draft-type):
    the "Pick" column is an overall order-label that is often blank or
    "Priority (Club)" and is NOT a usable number. Returns [] when no matching
    table is found (page structure changed or the year is absent) so the caller
    skips the year rather than crashing.
    """
    table = _find_table(soup)
    if table is None:
        return []

    headers = _header_names(table)
    i_draft = _column_index(headers, "Draft")
    i_num = _hash_index(headers)          # the real pick-within-draft number
    i_player = _column_index(headers, "Player")
    i_original = _column_index(headers, "Original Club")
    i_grade = _column_index(headers, "Grade")
    i_games = _column_index(headers, "Games")
    i_goals = _column_index(headers, "Goals")
    if i_num is None or i_draft is None or i_player is None:
        return []

    rows: List[Dict[str, Any]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if not cells:
            continue  # header / separator row
        texts = [c.get_text(" ", strip=True) for c in cells]
        if len(texts) <= max(i_num, i_draft, i_player):
            continue

        draft_type = texts[i_draft].strip().lower()
        if draft_type != "national":
            continue

        pick_raw = re.sub(r"[^0-9]", "", texts[i_num])
        if not pick_raw:
            continue  # no numeric pick -> sub-header / malformed
        pick = int(pick_raw)

        player_name = clean_player_name(texts[i_player])
        if not player_name:
            continue

        def _cell(idx: Optional[int]) -> str:
            return texts[idx] if idx is not None and len(texts) > idx else ""

        rows.append({
            "year": int(year),
            "pick": pick,
            "player_name": player_name,
            "original_club": _normalise_ws(_cell(i_original)),
            "grade": _clean_grade(_cell(i_grade)),
            "games": _to_int(_cell(i_games)),
            "goals": _to_int(_cell(i_goals)),
        })
    return rows


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class DraftGuruScraper:
    def __init__(self, out_path: str = DEFAULT_OUT_PATH):
        self.out_path = out_path

    def _fetch_soup(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch a DraftGuru page and return parsed soup. Returns None on 404 or
        any request error (so a missing-year page is skipped, not fatal). This
        is the single network boundary -- tests patch it.
        """
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            if resp.status_code == 404:
                print(f"[draftguru] {url} -> 404; skipping")
                return None
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            print(f"[draftguru] error fetching {url}: {e}; skipping")
            return None

    def scrape_year(self, year: int) -> int:
        """
        Scrape one year's National-draft rows and merge into the output CSV,
        deduplicating on (year, pick). Returns the number of rows scraped for
        the year (0 if the page or table was missing).
        """
        url = YEAR_URL.format(year=year)
        soup = self._fetch_soup(url)
        if soup is None:
            return 0

        rows = parse_year_table(soup, year)
        if not rows:
            print(f"[draftguru] {year}: no National draft rows found; skipping")
            return 0

        new_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        self._merge_and_write(new_df, year)
        print(f"[draftguru] {year}: {len(new_df)} National picks scraped")
        return len(new_df)

    def _merge_and_write(self, new_df: pd.DataFrame, year: int) -> None:
        """
        Replace ``year``'s rows in the on-disk CSV with ``new_df`` (delta mode):
        drop existing rows for that year, append the fresh ones, then drop
        duplicate (year, pick) keeping the latest. Makes re-runs idempotent.
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
        combined["year"] = combined["year"].astype("Int64")
        combined["pick"] = combined["pick"].astype("Int64")
        combined["games"] = combined["games"].astype("Int64")
        combined["goals"] = combined["goals"].astype("Int64")
        combined.to_csv(self.out_path, index=False)

    def scrape_all(self, start: int = 2004, end: int = 2025) -> pd.DataFrame:
        """
        Scrape every year in [start, end], honouring the inter-request delay.
        Returns the full output DataFrame after the run.
        """
        for year in range(start, end + 1):
            self.scrape_year(year)
            time.sleep(REQUEST_DELAY_S)
        if os.path.exists(self.out_path):
            df = pd.read_csv(self.out_path)
            print(f"[draftguru] scrape_all complete: {len(df)} total rows on disk")
            return df
        return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _main(argv: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="DraftGuru pathway enrichment scraper")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--year", type=int, help="Scrape a single year")
    group.add_argument("--all", action="store_true", help="Scrape 2004-2025")
    parser.add_argument("--start", type=int, default=2004)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    args = parser.parse_args(argv)

    scraper = DraftGuruScraper(out_path=args.out)
    if args.all:
        scraper.scrape_all(start=args.start, end=args.end)
    else:
        scraper.scrape_year(args.year)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
