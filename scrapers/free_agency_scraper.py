"""
AFL 2026 free-agency / off-contract list scraper.

Two reputable, server-rendered sources (both verified by hand on 2026-06-19):

  PRIMARY  (official FA status): AFL.com.au 2026 free agents list
           https://www.afl.com.au/news/1484077/2026afl-free-agentslist
  SECONDARY (broader pool):      ZeroHanger off-contract 2026
           https://www.zerohanger.com/afl/players/off-contract-2026/

Both sites are fronted by bot mitigation (Cloudflare / Akamai) and frequently
return 403 to a plain ``requests`` call. The scraper therefore attempts a live
fetch and, on a block, FALLS BACK to a persisted fixture of the verified content
under ``scrapers/fixtures/`` -- logging which path it used. Either way the job is
the same: normalise source rows into ``data/contracts/afl_2026_contracts.csv``.

Contract facts come from the SOURCES, not the repo, and are tagged downstream as
``[contract source: ...]`` -- never ``[data]``. The repo is used ONLY to enrich
verifiable per-player stats (age, games) where a player_data file exists.

Output columns: player_name, club, position, age, contract_end, games, notes
  - contract_end = 2026 for every row (all are off-contract at end of 2026).
  - FA status lives in ``notes`` (e.g. "Restricted FA (AFL.com.au)").
  - position is always blank: the repo player_data has no position field.

Style mirrors ``scrapers/draft_scraper.py`` (requests + explicit User-Agent +
pure, directly-unit-tested parsing helpers).
"""

import glob
import logging
import os
import re
import sys
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

AFL_FA_URL = "https://www.afl.com.au/news/1484077/2026afl-free-agentslist"
ZEROHANGER_URL = "https://www.zerohanger.com/afl/players/off-contract-2026/"
USER_AGENT = (
    "SuperCoach-VIA free-agency scraper "
    "(https://github.com/; research/non-commercial)"
)
REQUEST_TIMEOUT_S = 20

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_AFL_FIXTURE = os.path.join(_REPO_ROOT, "scrapers", "fixtures", "afl_fa_2026.html")
DEFAULT_ZH_FIXTURE = os.path.join(
    _REPO_ROOT, "scrapers", "fixtures", "zerohanger_offcontract_2026.html")
DEFAULT_PLAYER_DIR = os.path.join(_REPO_ROOT, "data", "player_data")
DEFAULT_OUT_PATH = os.path.join(_REPO_ROOT, "data", "contracts", "afl_2026_contracts.csv")

OUTPUT_COLUMNS = ["player_name", "club", "position", "age", "contract_end", "games", "notes"]
CONTRACT_END_YEAR = 2026

# Statuses that signal a Cloudflare / bot block rather than a genuine page.
_BLOCK_STATUSES = {403, 429, 503}


class FetchBlocked(Exception):
    """Raised when a live fetch is bot-blocked (403/429/503) -- triggers fixture fallback."""


# ---------------------------------------------------------------------------
# Pure parsing helpers (no network -- directly unit-tested)
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    return " ".join((text or "").split()).strip()


def parse_afl_fa_list(html: str) -> List[Dict[str, str]]:
    """Parse the AFL.com.au FA list fixture into ``[{player_name, club, status}]``.

    Structure (per club): a ``div.club-section`` carrying the club name, holding
    rows of ``td.player`` + ``td.status``. A missing status cell yields ``""``
    rather than crashing. Blank player names are skipped.
    """
    if not html or not html.strip():
        return []
    soup = BeautifulSoup(html, "html.parser")
    rows: List[Dict[str, str]] = []
    for section in soup.select("div.club-section"):
        club = _clean(section.get("data-club", ""))
        if not club:
            head = section.find(class_="club-name")
            club = _clean(head.get_text()) if head else ""
        for tr in section.select("tr"):
            name_cell = tr.find("td", class_="player")
            if name_cell is None:
                continue
            name = _clean(name_cell.get_text())
            if not name:
                continue
            status_cell = tr.find("td", class_="status")
            status = _clean(status_cell.get_text()) if status_cell is not None else ""
            rows.append({"player_name": name, "club": club, "status": status})
    return rows


def parse_zerohanger(html: str) -> List[Dict[str, str]]:
    """Parse the ZeroHanger off-contract fixture into ``[{player_name, club}]``.

    Structure (per club): a ``div.club`` carrying the club name, holding ``li``
    elements, one player name each. Blank names are skipped.
    """
    if not html or not html.strip():
        return []
    soup = BeautifulSoup(html, "html.parser")
    rows: List[Dict[str, str]] = []
    for block in soup.select("div.club"):
        club = _clean(block.get("data-club", ""))
        if not club:
            head = block.find(["h2", "h3", "h4"])
            club = _clean(head.get_text()) if head else ""
        for li in block.select("li"):
            name = _clean(li.get_text())
            if name:
                rows.append({"player_name": name, "club": club})
    return rows


def fa_status_note(status: str) -> str:
    """Map an AFL FA status string to the canonical ``notes`` value."""
    s = (status or "").strip().lower()
    if s.startswith("restrict"):
        return "Restricted FA (AFL.com.au)"
    if s.startswith("unrestrict"):
        return "Unrestricted FA (AFL.com.au)"
    return "Free agent (AFL.com.au)"


def _name_key(name: str) -> str:
    """Normalise a player name for dedup matching (case/whitespace/apostrophe-insensitive)."""
    return re.sub(r"[^a-z]", "", (name or "").lower())


def merge_rows(afl_rows: List[Dict], zh_rows: List[Dict]) -> List[Dict]:
    """Union AFL FA rows with ZeroHanger rows, deduped by player name.

    AFL FA rows are authoritative: a ZeroHanger row for a player already on the
    FA list is dropped (the FA status note is kept).
    """
    seen = {_name_key(r["player_name"]) for r in afl_rows}
    merged = list(afl_rows)
    for r in zh_rows:
        if _name_key(r["player_name"]) in seen:
            continue
        seen.add(_name_key(r["player_name"]))
        merged.append(r)
    return merged


def compute_age(born_date_str: Optional[str], as_of: date) -> Optional[int]:
    """Age in whole years as of ``as_of`` from a ``DD-MM-YYYY`` born date.

    Returns ``None`` on missing/unparseable input.
    """
    if not born_date_str:
        return None
    try:
        born = datetime.strptime(str(born_date_str).strip(), "%d-%m-%Y").date()
    except ValueError:
        return None
    age = as_of.year - born.year - ((as_of.month, as_of.day) < (born.month, born.day))
    return age


# ---------------------------------------------------------------------------
# Repo enrichment (reads data/player_data; injectable dir for testing)
# ---------------------------------------------------------------------------

# Bidirectional full-name <-> nickname pairs seen in repo file names. Player CSV
# files frequently store the nickname ("mitch", "ollie") while the official FA
# list uses the full first name ("Mitchell", "Oliver"). Matching on first-name
# alone otherwise produces false "stats not in repo" misses.
_NICKNAME_PAIRS = [
    ("mitchell", "mitch"), ("samuel", "sam"), ("nicholas", "nic"), ("nicholas", "nick"),
    ("zachary", "zac"), ("zachary", "zach"), ("lachlan", "lachie"), ("lachlan", "lachy"),
    ("thomas", "tom"), ("oliver", "ollie"), ("matthew", "matt"), ("jarrad", "jarrod"),
    ("william", "will"), ("william", "billy"), ("benjamin", "ben"), ("joshua", "josh"),
    ("jacob", "jake"), ("nathan", "nath"), ("daniel", "dan"), ("michael", "mick"),
    ("christopher", "chris"), ("harrison", "harry"), ("maxwell", "max"),
    ("edward", "ed"), ("anthony", "tony"), ("gregory", "greg"), ("patrick", "pat"),
    ("jonathon", "jon"), ("timothy", "tim"), ("kieren", "kieran"),
]


def _first_name_variants(first: str) -> set:
    f = re.sub(r"[^a-z]", "", (first or "").lower())
    variants = {f}
    for full, short in _NICKNAME_PAIRS:
        if f == full:
            variants.add(short)
        if f == short:
            variants.add(full)
    return variants


def _last_name_norm(tokens: List[str]) -> str:
    """Normalise the surname portion: lowercase, drop apostrophes, keep hyphens."""
    last = " ".join(tokens).lower().replace("'", "")
    return last.replace(" ", "")


def _candidate_personal_paths(name: str, player_data_dir: str) -> List[str]:
    """All personal_details paths whose surname matches and whose first-name token
    is in the name's nickname-equivalence set. May be 0, 1, or many (ambiguous)."""
    parts = (name or "").strip().split()
    if len(parts) < 2:
        return []
    first_variants = _first_name_variants(parts[0])
    last = _last_name_norm(parts[1:])
    hits = glob.glob(os.path.join(player_data_dir, f"{last}_*_personal_details.csv"))
    matched = []
    for path in hits:
        fname = os.path.basename(path)
        toks = fname.split("_")
        if len(toks) < 3:
            continue
        # toks[0] = surname (may contain hyphen, never underscore), toks[1] = first
        if toks[0] == last and toks[1] in first_variants:
            matched.append(path)
    return sorted(matched)


def enrich_with_repo_stats(rows: List[Dict], player_data_dir: str,
                           as_of_date: date) -> List[Dict]:
    """Fill ``position`` (always blank), ``age``, ``games``, ``contract_end`` per row.

    Matching is surname + nickname-aware first name. Where no file matches, age/games
    are left blank and the note is suffixed ``" | stats not in repo"``. Where MULTIPLE
    files match (genuine same-name players), the matcher refuses to guess: blank stats
    and a ``" | ambiguous repo match (N files)"`` note.
    """
    out: List[Dict] = []
    for r in rows:
        row = {
            "player_name": r["player_name"],
            "club": r.get("club", ""),
            "position": "",  # repo player_data has no position field
            "age": "",
            "contract_end": CONTRACT_END_YEAR,
            "games": "",
            "notes": r.get("notes", ""),
        }
        cands = _candidate_personal_paths(r["player_name"], player_data_dir)
        if not cands:
            note = row["notes"]
            row["notes"] = (note + " | stats not in repo") if note else "stats not in repo"
            out.append(row)
            continue
        if len(cands) > 1:
            note = row["notes"]
            suffix = f"ambiguous repo match ({len(cands)} files)"
            row["notes"] = (note + " | " + suffix) if note else suffix
            out.append(row)
            continue
        personal = cands[0]
        try:
            pdf = pd.read_csv(personal)
            born = str(pdf.iloc[0]["born_date"]) if "born_date" in pdf.columns else None
            age = compute_age(born, as_of_date)
            if age is not None:
                row["age"] = age
            perf = personal.replace("_personal_details.csv", "_performance_details.csv")
            if os.path.exists(perf):
                gdf = pd.read_csv(perf)
                if "games_played" in gdf.columns and len(gdf):
                    row["games"] = int(pd.to_numeric(gdf["games_played"],
                                                     errors="coerce").max())
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("enrichment failed for %s: %s", r["player_name"], exc)
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Scraper (network boundary + orchestration)
# ---------------------------------------------------------------------------

class FreeAgencyScraper:
    def __init__(self, afl_fixture: str = DEFAULT_AFL_FIXTURE,
                 zh_fixture: str = DEFAULT_ZH_FIXTURE,
                 player_data_dir: str = DEFAULT_PLAYER_DIR):
        self.afl_fixture = afl_fixture
        self.zh_fixture = zh_fixture
        self.player_data_dir = player_data_dir

    def _fetch_html(self, url: str) -> str:
        """Network boundary -- the ONLY method that touches the network.

        Raises ``FetchBlocked`` on a bot-mitigation status so the caller can fall
        back to a fixture. Patched out in all unit tests.
        """
        resp = requests.get(url, headers={"User-Agent": USER_AGENT},
                            timeout=REQUEST_TIMEOUT_S)
        if resp.status_code in _BLOCK_STATUSES:
            raise FetchBlocked(f"{resp.status_code} for {url}")
        resp.raise_for_status()
        return resp.text

    def fetch_or_fixture(self, url: str, fixture_path: str) -> Tuple[str, str]:
        """Try a live fetch; on block/error fall back to the fixture.

        Returns ``(html, source)`` where source is ``"live"`` or ``"fixture"``.
        """
        try:
            html = self._fetch_html(url)
            logger.info("fetched live: %s", url)
            return html, "live"
        except (FetchBlocked, requests.RequestException) as exc:
            logger.warning("live fetch blocked/failed for %s (%s); "
                           "falling back to fixture %s", url, exc, fixture_path)
            with open(fixture_path, "r", encoding="utf-8") as fh:
                return fh.read(), "fixture"

    def _source_rows(self, url: str, fixture_path: str, parser) -> Tuple[List[Dict], str]:
        """Fetch+parse a source; if a live 200 parses to 0 rows (JS/Cloudflare shell),
        fall back to the fixture and re-parse. Returns ``(parsed_rows, source)``."""
        html, source = self.fetch_or_fixture(url, fixture_path)
        parsed = parser(html)
        if not parsed and source == "live":
            logger.warning("live fetch for %s parsed 0 rows (likely a JS/Cloudflare "
                           "shell); falling back to fixture %s", url, fixture_path)
            with open(fixture_path, "r", encoding="utf-8") as fh:
                parsed = parser(fh.read())
            source = "fixture"
        return parsed, source

    def scrape(self, as_of_date: Optional[date] = None) -> List[Dict]:
        """Full pipeline: fetch (or fixture) both sources, merge, enrich. Returns rows."""
        if as_of_date is None:
            as_of_date = date.today()

        afl_parsed, afl_src = self._source_rows(
            AFL_FA_URL, self.afl_fixture, parse_afl_fa_list)
        zh_parsed, zh_src = self._source_rows(
            ZEROHANGER_URL, self.zh_fixture, parse_zerohanger)
        logger.info("source paths -> AFL: %s, ZeroHanger: %s", afl_src, zh_src)

        afl_rows = [
            {"player_name": r["player_name"], "club": r["club"],
             "notes": fa_status_note(r["status"])}
            for r in afl_parsed
        ]
        zh_rows = [
            {"player_name": r["player_name"], "club": r["club"],
             "notes": "Off-contract (ZeroHanger)"}
            for r in zh_parsed
        ]
        merged = merge_rows(afl_rows, zh_rows)
        return enrich_with_repo_stats(merged, self.player_data_dir, as_of_date)

    def to_csv(self, rows: List[Dict], out_path: str) -> int:
        """Write rows to ``out_path`` with the canonical column order. Returns row count."""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        df = df.sort_values(["club", "player_name"]).reset_index(drop=True)
        df.to_csv(out_path, index=False)
        return len(df)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    scraper = FreeAgencyScraper()
    rows = scraper.scrape(as_of_date=date(2026, 6, 19))
    n = scraper.to_csv(rows, DEFAULT_OUT_PATH)
    logger.info("wrote %d rows to %s", n, DEFAULT_OUT_PATH)
    print(f"wrote {n} rows to {DEFAULT_OUT_PATH}")


if __name__ == "__main__":
    main()
