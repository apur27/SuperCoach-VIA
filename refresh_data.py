"""
Targeted, bounded refresh of AFL data for the SuperCoach-VIA pipeline.

Strategy:
  1. Run MatchScraper in delta mode (it picks up from the latest year on disk).
  2. For players, walk team all-time pages to discover the current roster. Then
     re-scrape only players who are either:
       (a) currently active per local data (last game >= ACTIVE_SINCE), OR
       (b) discovered on a team page but missing from local data (new debuts).
     This avoids re-fetching ~12,500 retired players whose career stats are stable.
  3. Logs how many player files were touched, with row-count deltas where possible.

Run from repo root:
    source /home/abhi/sourceCode/python/coding/.venv/bin/activate
    python refresh_data.py
"""
from __future__ import annotations

import concurrent.futures
import glob
import logging
import os
import sys
from datetime import datetime

import pandas as pd

from scrapers.game_scraper import MatchScraper, audit_match_rounds
from scrapers.player_scraper import PlayerScraper, get_soup

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)

ACTIVE_SINCE = config.ACTIVE_SINCE  # players with any game on/after this are refresh candidates
DATA_DIR = config.PLAYER_DATA_DIR


def find_active_player_ids(data_dir: str) -> set[str]:
    """Return player_id (filename stem) for players with last_game >= ACTIVE_SINCE."""
    perf_files = sorted(glob.glob(os.path.join(data_dir, "*_performance_details.csv")))
    active: set[str] = set()
    for f in perf_files:
        try:
            df = pd.read_csv(f, usecols=["date"], low_memory=False)
            if df.empty:
                continue
            mx = df["date"].max()
            if isinstance(mx, str) and mx >= ACTIVE_SINCE:
                pid = os.path.basename(f).replace("_performance_details.csv", "")
                active.add(pid)
        except Exception as e:
            logging.warning("Could not read %s: %s", f, e)
    return active


def player_link_to_id_hint(href: str) -> str:
    """Produce a hint string from a /players/L/First_Last.html href (lower-case stem)."""
    base = os.path.basename(href).replace(".html", "")
    return base.lower()


def candidate_links(scraper: PlayerScraper, active_ids: set[str]) -> list[str]:
    """Return player links that are either matched by an active id OR look new.

    Active match is loose — we rely on the player_scraper's own retirement skip + delta
    logic for correctness. Here we just decide which pages to fetch at all.
    """
    all_links = scraper._get_player_links()
    logging.info("Discovered %d player links from team pages", len(all_links))

    # Build a quick lookup: lowercase last_first hints from active player_ids.
    # active_ids look like "bontempelli_marcus_24111995"; the link stem is "Marcus_Bontempelli"
    active_name_hints: set[str] = set()
    for pid in active_ids:
        parts = pid.split("_")
        if len(parts) >= 2:
            last, first = parts[0], parts[1]
            active_name_hints.add(f"{first}_{last}")  # match link convention "First_Last"

    # Existing player files set (for new-debut detection)
    existing_perf = {
        os.path.basename(p).replace("_performance_details.csv", "")
        for p in glob.glob(os.path.join(DATA_DIR, "*_performance_details.csv"))
    }
    existing_name_hints = set()
    for pid in existing_perf:
        parts = pid.split("_")
        if len(parts) >= 2:
            existing_name_hints.add(f"{parts[1]}_{parts[0]}")

    selected: list[str] = []
    for href in all_links:
        stem = os.path.basename(href).replace(".html", "")  # "Marcus_Bontempelli" or "Marcus_Bontempelli0"
        # strip trailing digits (AFLTables disambiguator like "Jack_Dyer0")
        stem_clean = stem.rstrip("0123456789")
        hint = stem_clean.lower()
        if hint in active_name_hints:
            selected.append(href)
        elif hint not in existing_name_hints:
            # potentially a new debut not previously scraped
            selected.append(href)
    logging.info(
        "Selected %d candidate links (active or new) out of %d",
        len(selected),
        len(all_links),
    )
    return selected


def refresh_players() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    active_ids = find_active_player_ids(DATA_DIR)
    logging.info("Active players (last game >= %s): %d", ACTIVE_SINCE, len(active_ids))

    scraper = PlayerScraper()
    targets = candidate_links(scraper, active_ids)

    if not targets:
        logging.warning("No player targets selected; nothing to do.")
        return

    # Snapshot row counts before
    before_rows: dict[str, int] = {}
    for pid in active_ids:
        path = os.path.join(DATA_DIR, f"{pid}_performance_details.csv")
        if os.path.exists(path):
            try:
                before_rows[pid] = sum(1 for _ in open(path)) - 1
            except Exception:
                pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(scraper._process_player, link, DATA_DIR) for link in targets]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                fut.result()
            except Exception as e:
                logging.error("worker error: %s", e)
            if i % 50 == 0:
                logging.info("Processed %d / %d player pages", i, len(targets))

    # Diff row counts
    grew = 0
    new = 0
    after_rows = 0
    for pid in active_ids:
        path = os.path.join(DATA_DIR, f"{pid}_performance_details.csv")
        if os.path.exists(path):
            try:
                cur = sum(1 for _ in open(path)) - 1
                after_rows += cur
                if pid in before_rows:
                    if cur > before_rows[pid]:
                        grew += 1
                else:
                    new += 1
            except Exception:
                pass
    logging.info(
        "Refresh summary: %d active files grew, %d new files appeared, total active rows now %d",
        grew,
        new,
        after_rows,
    )


def refresh_matches() -> None:
    scraper = MatchScraper()
    scraper.scrape_all_matches(
        match_folder_path=config.MATCHES_DIR,
        lineup_folder_path=config.LINEUPS_DIR,
    )
    # Post-write self-check: audit the current season's match file for rounds
    # that were silently truncated (the "R10 2026" bug). Warnings only -- this
    # never aborts the refresh, it just surfaces probable gaps in the log.
    current_year = datetime.now().year
    current_file = os.path.join(config.MATCHES_DIR, f"matches_{current_year}.csv")
    if os.path.exists(current_file):
        issues = audit_match_rounds(current_file)
        gaps = [i for i in issues if i["severity"] == "WARNING"]
        if gaps:
            logging.warning(
                "Match audit found %d probable scraper gap(s) in %s: rounds %s",
                len(gaps),
                os.path.basename(current_file),
                ", ".join(str(i["round_num"]) for i in gaps),
            )
        else:
            logging.info("Match audit clean for %s", os.path.basename(current_file))


def main() -> None:
    started = datetime.now()
    logging.info("=== Refresh started at %s ===", started.isoformat(timespec="seconds"))
    logging.info("--- Match scraper (delta) ---")
    refresh_matches()
    logging.info("--- Player scraper (targeted) ---")
    refresh_players()
    logging.info(
        "=== Refresh complete in %s ===",
        datetime.now() - started,
    )


if __name__ == "__main__":
    main()
