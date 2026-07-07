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
import time
from datetime import datetime

import pandas as pd

from scrapers.game_scraper import (
    MatchScraper,
    audit_match_rounds,
    audit_player_career_totals,
)
from scrapers.player_scraper import PlayerScraper, get_soup
from scripts.phantom_row_validator import check_counter_gaps, gaps_in_season

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
    grown_paths: list[str] = []
    for pid in active_ids:
        path = os.path.join(DATA_DIR, f"{pid}_performance_details.csv")
        if os.path.exists(path):
            try:
                cur = sum(1 for _ in open(path)) - 1
                after_rows += cur
                if pid in before_rows:
                    if cur > before_rows[pid]:
                        grew += 1
                        grown_paths.append(path)
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

    # Self-check: reconcile each updated player's career totals against the
    # afltables profile page, so a silently dropped/double-counted career stat
    # is surfaced this run. Only the files that actually grew are audited (not
    # all ~13k), honouring the 0.5s inter-request delay. Warnings only -- never
    # aborts the refresh, same contract as audit_match_rounds().
    if grown_paths:
        logging.info("Auditing career totals for %d updated player file(s)", len(grown_paths))
        total_player_warnings = 0
        for path in grown_paths:
            try:
                issues = audit_player_career_totals(path)
            except Exception as e:
                logging.error("player audit error for %s: %s", path, e)
                continue
            for issue in issues:
                if issue["severity"] == "WARNING":
                    total_player_warnings += 1
                    logging.warning(
                        "Player audit: %s %s csv=%g vs afltables=%g (delta %g)",
                        issue["player"],
                        issue["stat"],
                        issue["csv_val"],
                        issue["source_val"],
                        issue["delta"],
                    )
            time.sleep(0.5)
        if total_player_warnings:
            logging.warning(
                "Player audit found %d career-total reconciliation WARNING(s)",
                total_player_warnings,
            )
        else:
            logging.info("Player audit clean across %d updated file(s)", len(grown_paths))

        # Phantom-row gate: verify each updated player's games_played counter is
        # a contiguous {1..max} with no gaps/dups. A gap == a silently dropped
        # game row (the drawn-GF dedup collapse that lost Sidebottom's 2010 draw).
        # HARD ABORT if a gap lands in the current season -- that is a live data
        # loss we must not ship. Historical gaps are WARNING-only (they predate
        # the fix and are backfilled separately).
        current_season = datetime.now().year
        historical_gap_files = 0
        current_gap_files = []
        for path in grown_paths:
            try:
                gaps = check_counter_gaps(path)
            except Exception as e:
                logging.error("phantom-row check error for %s: %s", path, e)
                continue
            if gaps["ok"]:
                continue
            season_gaps = gaps_in_season(path, current_season)
            if season_gaps:
                current_gap_files.append(path)
                logging.error(
                    "Phantom-row gate: %s has a %d-season counter gap %s "
                    "(missing=%s duplicated=%s) -- a current-season game row was dropped",
                    gaps["player"], current_season, season_gaps,
                    gaps["missing"], gaps["duplicated"],
                )
            else:
                historical_gap_files += 1
                logging.warning(
                    "Phantom-row check: %s has historical counter gap(s) "
                    "missing=%s duplicated=%s (pre-existing, needs backfill)",
                    gaps["player"], gaps["missing"], gaps["duplicated"],
                )
        if current_gap_files:
            raise RuntimeError(
                f"Phantom-row gate ABORT: {len(current_gap_files)} updated player "
                f"file(s) lost a {current_season}-season game row: "
                + ", ".join(os.path.basename(p) for p in current_gap_files)
            )
        if historical_gap_files:
            logging.warning(
                "Phantom-row check found %d file(s) with pre-existing historical gaps",
                historical_gap_files,
            )
        else:
            logging.info("Phantom-row check clean across %d updated file(s)", len(grown_paths))


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
    # Single-entry-point discipline (F04): refresh_data.py is an internal phase of
    # the weekly cycle (invoked by refresh_and_rank.sh, itself invoked by
    # scripts/weekly_refresh.sh). Running it directly does a partial, ungated scrape.
    if os.environ.get("WEEKLY_REFRESH_PARENT") != "1" and "--allow-direct" not in sys.argv:
        sys.stderr.write(
            "refresh_data.py is an internal phase of the weekly cycle, not an entry point.\n"
            "  Run:  bash scripts/weekly_refresh.sh            (full gated cycle)\n"
            "  Or, for a deliberate partial scrape:  python refresh_data.py --allow-direct\n"
        )
        sys.exit(1)
    main()
