"""
AFL rookie draft history scraper (end-of-season and mid-season).

Source: Wikipedia (``https://en.wikipedia.org/wiki/{year}_AFL_draft``).
The same page that carries the national draft also carries the rookie draft
(one or two sections depending on the year):

  - End-of-season rookie draft: heading contains "rookie draft" but NOT
    "mid-season" (e.g. "2024 rookie draft", "2023 rookie draft").
  - Mid-season rookie draft: heading contains "mid-season" and "rookie"
    (introduced 2018; e.g. "2022 mid-season rookie draft").

Table detection is by heading text, not table index. Column meaning is read
by header name so column-order drift across years is handled automatically.

Parsing rules:
  - Rows whose Player cell is "Passed" (club passed their pick) are skipped.
  - Rows whose Pick cell is non-numeric (sub-headers) are skipped.
  - Footnote markers ("[1]", "[2]") are stripped from player names.

Output: ``data/drafts/afl_rookie_draft_history.csv`` with columns:
  year, draft_type, pick, club, player_name, recruited_from

  draft_type is "end_season" or "mid_season".

Deduplicated on (year, draft_type, pick): re-running a year replaces rows.

Style mirrors ``scrapers/draft_scraper.py``.
"""

import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

WIKI_URL = "https://en.wikipedia.org/wiki/{year}_AFL_draft"
USER_AGENT = (
    "SuperCoach-VIA rookie-draft scraper (https://github.com/; research/non-commercial)"
)
REQUEST_DELAY_S = 0.5
DEFAULT_OUT_PATH = "data/drafts/afl_rookie_draft_history.csv"
OUTPUT_COLUMNS = ["year", "draft_type", "pick", "club", "player_name", "recruited_from"]

_FOOTNOTE_RE = re.compile(r"\[[^\]]*\]")


# ---------------------------------------------------------------------------
# Pure parsing helpers
# ---------------------------------------------------------------------------

def clean_player_name(raw: str) -> str:
    """Strip footnote markers and collapse whitespace."""
    return " ".join(_FOOTNOTE_RE.sub("", raw).split()).strip()


def _header_names(table: BeautifulSoup) -> List[str]:
    first_row = table.find("tr")
    if first_row is None:
        return []
    return [th.get_text(" ", strip=True) for th in first_row.find_all("th")]


def _expand_rows(table: BeautifulSoup) -> List[List[str]]:
    """Render HTML table into a dense grid honouring rowspan/colspan."""
    grid: List[List[str]] = []
    pending: Dict[int, Tuple[str, int]] = {}

    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        row: List[str] = []
        col = 0
        ci = 0
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


def _is_rookie_table(table: BeautifulSoup) -> bool:
    """A rookie-draft table has Pick and Player headers (Round is absent)."""
    headers = {h.lower() for h in _header_names(table)}
    return ("pick" in headers or "#" in headers) and "player" in headers


def _classify_heading(heading_text: str) -> Optional[str]:
    """
    Classify a heading as a rookie draft type.

    Returns "mid_season", "end_season", or None (not a rookie draft heading).
    """
    t = heading_text.lower()
    if "rookie" not in t:
        return None
    if "national" in t:
        return None  # national draft heading, not a rookie draft
    if "mid-season" in t or "mid season" in t:
        return "mid_season"
    return "end_season"


def find_rookie_draft_tables(
    soup: BeautifulSoup,
) -> List[Tuple[BeautifulSoup, str]]:
    """
    Return all (table, draft_type) pairs where the table is under a heading
    that identifies a rookie draft section (end-of-season or mid-season).

    draft_type is "end_season" or "mid_season".
    """
    results = []
    candidates = soup.find_all("table", class_="wikitable")
    if not candidates:
        candidates = soup.find_all("table")

    for table in candidates:
        if not _is_rookie_table(table):
            continue
        heading_tag = table.find_previous(["h2", "h3", "h4"])
        if heading_tag is None:
            continue
        heading_text = heading_tag.get_text(" ", strip=True)
        draft_type = _classify_heading(heading_text)
        if draft_type is None:
            continue
        results.append((table, draft_type))

    return results


def _column_index(headers: List[str], *names: str) -> Optional[int]:
    """First index whose header matches any of the given names (case-insensitive)."""
    target = {n.lower() for n in names}
    for i, h in enumerate(headers):
        if h.lower() in target:
            return i
    return None


def parse_rookie_table(
    table: BeautifulSoup,
    year: int,
    draft_type: str,
) -> List[Dict[str, Any]]:
    """
    Parse a rookie-draft wikitable into row dicts.

    Skips rows where:
      - Pick is non-numeric (sub-headers)
      - Player name is "Passed" (club passed their selection)
    """
    headers = _header_names(table)
    if not headers:
        return []

    i_pick = _column_index(headers, "Pick", "#")
    i_player = _column_index(headers, "Player")
    i_club = _column_index(headers, "Club", "Recruited by", "Recruited to", "Drafted to")
    i_recruited = _column_index(headers, "Recruited from")

    if i_pick is None or i_player is None:
        return []

    rows: List[Dict[str, Any]] = []
    for texts in _expand_rows(table):
        if len(texts) <= i_pick:
            continue

        pick_raw = texts[i_pick].strip()
        if not pick_raw.isdigit():
            continue
        pick = int(pick_raw)

        if i_player >= len(texts):
            continue
        player_name = clean_player_name(texts[i_player])
        if not player_name or player_name.lower() == "passed":
            continue

        club = ""
        if i_club is not None and i_club < len(texts):
            club = clean_player_name(texts[i_club])
            if club in ("—", "-", ""):
                club = ""

        recruited_from = ""
        if i_recruited is not None and i_recruited < len(texts):
            recruited_from = clean_player_name(texts[i_recruited])
            if recruited_from in ("—", "-"):
                recruited_from = ""

        rows.append({
            "year": int(year),
            "draft_type": draft_type,
            "pick": pick,
            "club": club,
            "player_name": player_name,
            "recruited_from": recruited_from,
        })

    return rows


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class RookieDraftScraper:
    def __init__(self, out_path: str = DEFAULT_OUT_PATH):
        self.out_path = out_path

    def _fetch_soup(self, url: str) -> Optional[BeautifulSoup]:
        """Single network boundary — tests patch this."""
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            if resp.status_code == 404:
                print(f"[rookie] {url} -> 404; skipping")
                return None
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            print(f"[rookie] error fetching {url}: {e}; skipping")
            return None

    def scrape_year(self, year: int) -> int:
        """
        Scrape one year's rookie draft(s) and merge into the output CSV.
        Returns total rows scraped across all rookie draft types for the year.
        """
        url = WIKI_URL.format(year=year)
        soup = self._fetch_soup(url)
        if soup is None:
            return 0

        table_pairs = find_rookie_draft_tables(soup)
        if not table_pairs:
            print(f"[rookie] {year}: no rookie draft tables found")
            return 0

        all_rows: List[Dict[str, Any]] = []
        for table, draft_type in table_pairs:
            rows = parse_rookie_table(table, year, draft_type)
            if rows:
                print(f"[rookie] {year} {draft_type}: {len(rows)} picks")
                all_rows.extend(rows)

        if not all_rows:
            return 0

        new_df = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)
        self._merge_and_write(new_df, year)
        return len(all_rows)

    def _merge_and_write(self, new_df: pd.DataFrame, year: int) -> None:
        """Replace year's rows in the CSV (delta mode, idempotent)."""
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)

        if os.path.exists(self.out_path):
            existing = pd.read_csv(self.out_path)
            existing = existing[existing["year"] != year]
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df

        combined = combined.drop_duplicates(
            subset=["year", "draft_type", "pick"], keep="last"
        )
        combined = combined.sort_values(
            ["year", "draft_type", "pick"]
        ).reset_index(drop=True)
        combined["pick"] = combined["pick"].astype("Int64")
        combined.to_csv(self.out_path, index=False)

    def scrape_all(self, start: int = 2004, end: int = 2025) -> int:
        """
        Scrape every year in [start, end]. Returns total rows in the output CSV.
        Mid-season rookie draft started in 2018; earlier years have end-season only.
        """
        total = 0
        for year in range(start, end + 1):
            total += self.scrape_year(year)
            time.sleep(REQUEST_DELAY_S)
        if os.path.exists(self.out_path):
            n = len(pd.read_csv(self.out_path))
            print(f"[rookie] scrape_all complete: {n} total rows on disk")
            return n
        return total


def _main(argv: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="AFL rookie draft scraper (Wikipedia, end-of-season + mid-season)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--year", type=int, help="Scrape a single year")
    group.add_argument("--all", action="store_true", help="Scrape 2004-2025")
    parser.add_argument("--start", type=int, default=2004)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    args = parser.parse_args(argv)

    scraper = RookieDraftScraper(out_path=args.out)
    if args.all:
        scraper.scrape_all(start=args.start, end=args.end)
    else:
        scraper.scrape_year(args.year)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
