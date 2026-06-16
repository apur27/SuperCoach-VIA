import os
import re
import sys
import hashlib
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
from bs4 import BeautifulSoup
import requests

def get_soup(url: str) -> BeautifulSoup:
    """
    Gets a BeautifulSoup object from the given URL.
    
    Args:
        url (str): The URL to get the soup from.
        
    Returns:
        BeautifulSoup: The BeautifulSoup object.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        return BeautifulSoup("", 'html.parser')

# Canonical AFL/VFL club names as they appear on afltables.com season pages and
# in our scraped matches_<year>.csv. Used to identify team-name cells in the
# fixture HTML and to ignore ladder/footnote rows. Includes current clubs plus
# historical names so the fixture parser works across eras.
KNOWN_TEAM_NAMES = {
    "Adelaide", "Brisbane Lions", "Carlton", "Collingwood", "Essendon",
    "Fremantle", "Geelong", "Gold Coast", "Greater Western Sydney", "Hawthorn",
    "Melbourne", "North Melbourne", "Port Adelaide", "Richmond", "St Kilda",
    "Sydney", "West Coast", "Western Bulldogs",
    # Historical / former names that appear on older season pages
    "Brisbane Bears", "Fitzroy", "Footscray", "Kangaroos", "South Melbourne",
    "University",
}

# afltables season-results page. This is the SAME site the match scraper already
# hits (self.base_url), and it lists the full fixture: completed rounds carry
# scores, and *upcoming* rounds list the scheduled matchups with no scores. That
# makes it the ground-truth schedule -- no extra dependency, no second source.
FIXTURE_BASE_URL = "https://afltables.com/afl/seas/"

# Per-process cache of parsed season pages, so auditing every round in a file
# fetches the season page once instead of once per round.
_SEASON_SOUP_CACHE: Dict[int, BeautifulSoup] = {}


def _get_season_soup(year: int) -> BeautifulSoup:
    """Return the parsed afltables season page for `year`, cached per process."""
    if year not in _SEASON_SOUP_CACHE:
        _SEASON_SOUP_CACHE[year] = get_soup(f"{FIXTURE_BASE_URL}{year}.html")
    return _SEASON_SOUP_CACHE[year]


def fetch_round_fixture(year: int, round_num: int) -> Optional[set]:
    """
    Fetch the scheduled matchups for a single home-and-away round from the
    afltables season-results page, which lists the full fixture (played and
    upcoming) for the year.

    Returns a set of frozenset({team_a, team_b}) pairs -- one per scheduled
    match -- so it can be compared directly against scraped matchups regardless
    of home/away ordering. Returns None if the round heading or its match table
    cannot be located (e.g. the page layout changed, or the round does not
    exist), so the caller can distinguish "no ground truth available" from
    "zero matches scheduled".

    Parsing model (verified against the 2026 season page):
      - Each round is introduced by a `<b>Round N ...</b>` heading; the asterisk
        footnote variant ("Round 1* see notes") is handled by a bounded match.
      - The round's matches live in the very next sibling `<table>`. Within it,
        each match is two consecutive rows whose first cell is a known team name;
        we read the team-name cells in document order and pair them sequentially.
      - Bye rows ("<team> | Bye") and the trailing "Rd N Ladder" block are
        skipped because they are not 3+-cell team rows.
    """
    soup = _get_season_soup(year)

    heading = None
    pat = re.compile(rf"^\s*Round\s+{round_num}(?!\d)")
    for b in soup.find_all("b"):
        if pat.match(b.get_text(strip=True)):
            heading = b
            break
    if heading is None:
        return None

    # Walk up from the heading <b> to its enclosing table, then take the next
    # sibling table -- that is the block of matches for this round.
    heading_table = heading
    while heading_table is not None and heading_table.name != "table":
        heading_table = heading_table.parent
    if heading_table is None:
        return None

    match_table = heading_table.find_next_sibling()
    while match_table is not None and match_table.name != "table":
        match_table = match_table.find_next_sibling()
    if match_table is None:
        return None

    team_sequence: List[str] = []
    for row in match_table.find_all("tr"):
        cells = row.find_all("td")
        if not cells:
            continue
        # Bye row is exactly [team, "Bye"] -- not a scheduled match.
        if len(cells) == 2 and cells[1].get_text(strip=True) == "Bye":
            continue
        name = cells[0].get_text(" ", strip=True)
        if name in KNOWN_TEAM_NAMES and len(cells) >= 3:
            team_sequence.append(name)

    pairs = set()
    for i in range(0, len(team_sequence) - 1, 2):
        pairs.add(frozenset((team_sequence[i], team_sequence[i + 1])))
    return pairs


def audit_match_rounds(file_path: str) -> List[Dict[str, Any]]:
    """
    Audits a matches_<year>.csv against the public AFL fixture, round by round.

    Catches the "R10 2026" class of bug, where the match-summary scraper silently
    wrote only 1 of 9 rows for a round and the gap went undetected for weeks.

    This check is exact, not probabilistic. For each integer home-and-away round
    present in the file, it fetches the scheduled matchups from the afltables
    season page (`fetch_round_fixture`) and compares them, by team pair, to what
    was scraped. A round is flagged WARNING if any *scheduled* matchup is absent
    from the scraped file -- and the warning names the exact missing matchups.
    There is no modal/threshold heuristic and no MAX_BYE_MATCH_DROP: byes are
    handled because the fixture itself omits resting teams.

    Returns a list of issue dicts (empty when clean). Each dict has keys:
      year, round_num, n_matches, expected, teams_present, severity, missing.
      - n_matches: matches scraped for the round
      - expected: matches scheduled in the fixture for the round
      - missing:  list of "TeamA v TeamB" strings for scheduled-but-unscraped games
      - severity: "WARNING" when missing is non-empty, else "INFO"
    Rounds whose fixture cannot be fetched are skipped (no false positives when
    the schedule is unavailable).
    """
    issues: List[Dict[str, Any]] = []
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[match-audit] could not read {file_path}: {e}")
        return issues

    required = {"round_num", "team_1_team_name", "team_2_team_name"}
    if not required.issubset(df.columns):
        print(f"[match-audit] {file_path} missing expected columns; skipping audit")
        return issues

    # Restrict to integer home-and-away rounds; finals rounds are stored as
    # non-numeric strings and are not covered by the season-page round headings.
    round_as_int = pd.to_numeric(df["round_num"], errors="coerce")
    ha = df[round_as_int.notna()].copy()
    ha["round_num"] = round_as_int[round_as_int.notna()].astype(int)
    if ha.empty:
        return issues

    year = int(ha["year"].iloc[0]) if "year" in ha.columns and not ha["year"].isna().all() else None
    if year is None:
        print(f"[match-audit] {file_path} has no usable year column; skipping audit")
        return issues

    for rnd in sorted(ha["round_num"].unique()):
        rnd = int(rnd)
        group = ha[ha["round_num"] == rnd]
        scraped_pairs = {
            frozenset((r.team_1_team_name, r.team_2_team_name))
            for r in group.itertuples()
        }
        n_matches = len(scraped_pairs)
        teams_present = len(set(group["team_1_team_name"]) | set(group["team_2_team_name"]))

        fixture = fetch_round_fixture(year, rnd)
        if fixture is None:
            print(
                f"[match-audit][info] {year} round {rnd}: {n_matches} matches scraped; "
                f"fixture unavailable -- cannot verify"
            )
            continue

        expected = len(fixture)
        missing_pairs = fixture - scraped_pairs
        missing = [" v ".join(sorted(p)) for p in missing_pairs]

        severity = "WARNING" if missing else "INFO"
        issues.append({
            "year": year,
            "round_num": rnd,
            "n_matches": n_matches,
            "expected": expected,
            "teams_present": teams_present,
            "severity": severity,
            "missing": missing,
        })

        if missing:
            print(
                f"[match-audit][WARNING] {year} round {rnd}: {n_matches}/{expected} "
                f"scheduled matches scraped <-- MISSING {len(missing)}: "
                f"{'; '.join(sorted(missing))}"
            )
        else:
            print(
                f"[match-audit][info] {year} round {rnd}: {n_matches}/{expected} "
                f"scheduled matches present (complete)"
            )
    return issues


# afltables player profile pages. Each player has a profile at
#   https://afltables.com/afl/stats/players/{initial}/{First}_{Last}.html
# where {initial} is the FIRST letter of the player's FIRST name (empirically:
# Scott Pendlebury lives at .../players/S/Scott_Pendlebury.html, and the
# last-name variant .../players/P/... 404s). The profile carries a stats table
# whose final "Totals" row holds career aggregates (GM games, DI disposals,
# GL goals, TK tackles, CL clearances, etc.) -- the ground truth we reconcile
# the locally-scraped per-game CSV against.
PLAYER_PROFILE_BASE_URL = "https://afltables.com/afl/stats/players/"

# Per-process cache of (initial, First_Last) -> Totals row (pd.Series) or None,
# mirroring _SEASON_SOUP_CACHE so re-auditing the same player in one run does
# not refetch. Value is None when the page 404s or has no parseable Totals row.
_PLAYER_TOTALS_CACHE: Dict[str, Optional["pd.Series"]] = {}

# Earliest season each stat is recorded league-wide on afltables. A stat is only
# reconciled when the player's career started on/after its era floor, because a
# pre-era career legitimately has zeros/NaNs in our CSV that the afltables Totals
# row also omits -- comparing them would manufacture phantom deltas.
# (Tackles from 1987; clearances from 1998 -- see Scientist memory
# data_stat_coverage_eras.md.)
_STAT_ERA_FLOOR = {
    "tackles": 1987,
    "clearances": 1998,
}


def _player_url_from_csv_path(player_csv_path: str) -> Optional[str]:
    """
    Derive the afltables profile URL from a performance-details CSV filename.

    Filenames are `<lastname>_<firstname>_<DDMMYYYY>_performance_details.csv`,
    so `pendlebury_scott_07011988_performance_details.csv` -> first=Scott,
    last=Pendlebury, initial=S (first letter of the FIRST name), giving
    `https://afltables.com/afl/stats/players/S/Scott_Pendlebury.html`.

    Returns None if the filename does not match the expected pattern.
    """
    stem = os.path.basename(player_csv_path).replace("_performance_details.csv", "")
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    last, first = parts[0], parts[1]
    if not first or not last:
        return None
    first_cap = first.capitalize()
    last_cap = last.capitalize()
    initial = first_cap[0].upper()
    return f"{PLAYER_PROFILE_BASE_URL}{initial}/{first_cap}_{last_cap}.html"


def _get_player_totals(player_url: str) -> Optional["pd.Series"]:
    """
    Fetch the afltables profile at `player_url` and return its career "Totals"
    row as a pandas Series (indexed by afltables column codes: GM, DI, GL, ...),
    cached per process. Returns None if the page 404s, can't be fetched, or has
    no table with a parseable Totals row.

    afltables 404s pandas' default urllib User-Agent, so we fetch with requests
    (same client the rest of this module uses) and hand the HTML to read_html.
    The first stats table on the page is the season-by-season block whose last
    row is "Totals"; we take the first table that contains such a row.
    """
    if player_url in _PLAYER_TOTALS_CACHE:
        return _PLAYER_TOTALS_CACHE[player_url]

    totals: Optional["pd.Series"] = None
    try:
        resp = requests.get(player_url, timeout=30)
        if resp.status_code == 404:
            print(f"[player-audit][info] {player_url} not found (404); skipping")
            _PLAYER_TOTALS_CACHE[player_url] = None
            return None
        resp.raise_for_status()
        import io
        tables = pd.read_html(io.StringIO(resp.text), flavor="lxml")
        for t in tables:
            if t.shape[1] == 0:
                continue
            first_col = t.iloc[:, 0].astype(str)
            mask = first_col.str.contains("Total", case=False, na=False)
            if mask.any():
                totals = t[mask].iloc[0]
                break
    except Exception as e:
        print(f"[player-audit][info] could not parse {player_url}: {e}; skipping")
        totals = None

    _PLAYER_TOTALS_CACHE[player_url] = totals
    return totals


def audit_player_career_totals(player_csv_path: str) -> List[Dict[str, Any]]:
    """
    Reconcile a player's locally-scraped career totals against the afltables
    profile page, stat by stat. Catches the class of bug where a per-game CSV
    silently dropped or double-counted games/disposals/goals over a career.

    For each reconcilable stat it compares the afltables career "Totals" value
    (ground truth) to the aggregate of our CSV: max(games_played) for games,
    column sum for counting stats. A stat is flagged WARNING when the absolute
    delta is > 0.

    Era-aware: tackles are only recorded league-wide from 1987 and clearances
    from 1998, so those stats are reconciled only when the player's career
    started on/after the relevant floor (career start = min(year) in the CSV).
    Games, disposals and goals are reconciled for every era.

    Returns a list of issue dicts (empty when clean or unverifiable). Each dict:
      {severity, player, stat, csv_val, source_val, delta}
      - severity: always "WARNING" (an issue is only emitted on a non-zero delta)
      - player:   the CSV filename stem
      - stat:     our CSV column name (games_played, disposals, ...)
      - csv_val / source_val / delta: numeric

    Never raises on a missing page or parse failure -- it logs INFO and returns
    an empty list, mirroring audit_match_rounds()'s warnings-only contract.
    """
    issues: List[Dict[str, Any]] = []
    player = os.path.basename(player_csv_path).replace("_performance_details.csv", "")

    try:
        df = pd.read_csv(player_csv_path)
    except Exception as e:
        print(f"[player-audit] could not read {player_csv_path}: {e}")
        return issues

    if "year" not in df.columns or df["year"].isna().all():
        print(f"[player-audit] {player}: no usable year column; skipping")
        return issues
    career_start = int(pd.to_numeric(df["year"], errors="coerce").min())
    if career_start < 1897:
        print(f"[player-audit] {player}: implausible career start {career_start}; skipping")
        return issues

    player_url = _player_url_from_csv_path(player_csv_path)
    if player_url is None:
        print(f"[player-audit] {player}: cannot derive profile URL; skipping")
        return issues

    totals = _get_player_totals(player_url)
    if totals is None:
        return issues

    # (afltables Totals column, our CSV column, aggregator). Games uses max of
    # the cumulative games_played counter (object dtype); counting stats sum.
    checks = [
        ("GM", "games_played", "max"),
        ("DI", "disposals", "sum"),
        ("GL", "goals", "sum"),
        ("TK", "tackles", "sum"),
        ("CL", "clearances", "sum"),
    ]

    for src_col, csv_col, how in checks:
        # Era gate: skip stats whose recording era postdates this career start.
        floor = _STAT_ERA_FLOOR.get(csv_col)
        if floor is not None and career_start < floor:
            continue
        if src_col not in totals.index or csv_col not in df.columns:
            continue

        source_val = pd.to_numeric(totals[src_col], errors="coerce")
        if pd.isna(source_val):
            continue

        col = pd.to_numeric(df[csv_col], errors="coerce")
        if how == "max":
            csv_val = col.max()
        else:
            # Blank counting stats are real zeros in a played game, not missing
            # data (Scientist memory blank_counting_stat_means.md), so sum over
            # fill-zero rather than dropna-then-sum.
            csv_val = col.fillna(0).sum()
        if pd.isna(csv_val):
            continue

        csv_val = float(csv_val)
        source_val = float(source_val)
        delta = abs(csv_val - source_val)
        if delta > 0:
            issues.append({
                "severity": "WARNING",
                "player": player,
                "stat": csv_col,
                "csv_val": csv_val,
                "source_val": source_val,
                "delta": delta,
            })
            print(
                f"[player-audit][WARNING] {player}: {csv_col} csv={csv_val:g} "
                f"vs afltables={source_val:g} (delta {delta:g})"
            )

    if not issues:
        print(f"[player-audit][info] {player}: career totals reconcile (clean)")
    return issues


class MatchScraper:
    def __init__(self):
        """
        Initialize the MatchScraper.
        """
        self.team_lineups: Dict[str, List[Dict[str, Union[str, List[str]]]]] = {}
        self.processed_lineup_keys = set()  # Track processed lineup entries
        self.base_url = "https://afltables.com/afl/seas/"  # Base URL for AFL tables

    def scrape_all_matches(self, match_folder_path: str = "./data/matches", lineup_folder_path: str = "./data/lineups") -> None:
        """
        Scrapes match details from the last processed date to the current date.

        Args:
            match_folder_path (str): The path to the folder where the match CSV files will be saved.
            lineup_folder_path (str): The path to the folder where the team lineup CSV files will be saved.

        Returns:
            None
        """
        # Ensure directories exist
        os.makedirs(match_folder_path, exist_ok=True)
        os.makedirs(lineup_folder_path, exist_ok=True)

        # Check if data already exists
        match_files = [f for f in os.listdir(match_folder_path) if f.endswith('.csv')]
        lineup_files = [f for f in os.listdir(lineup_folder_path) if f.endswith('.csv')]
        
        if match_files or lineup_files:
            print(f"Data already exists in {match_folder_path} ({len(match_files)} files) and/or {lineup_folder_path} ({len(lineup_files)} files).")
            if sys.stdin is None or not sys.stdin.isatty():
                print("Non-interactive run detected; proceeding with delta update.")
            else:
                proceed = input("Do you want to proceed with updating the data? (y/n): ").lower()
                if proceed != 'y':
                    print("Scraping aborted by user.")
                    return

        # Load existing team lineup data to avoid duplicates
        self._load_existing_lineup_keys(lineup_folder_path)
        
        # Get the current year and the last processed year/date
        current_year = datetime.now().year
        last_processed_year, last_processed_date = self._get_last_processed_info(match_folder_path)
        
        # Default to 2011 if no previous data exists
        start_year = last_processed_year if last_processed_year else 2011
        
        print(f"Starting delta update from {start_year} to {current_year}")
        print(f"Last processed date: {last_processed_date}")
        
        # Process each year
        for year in range(start_year, current_year + 1):
            self._process_year(year, match_folder_path, last_processed_date)
        
        # Process team lineups
        self._process_team_lineups(lineup_folder_path)

    def _get_last_processed_info(self, match_folder_path: str) -> Tuple[Optional[int], Optional[datetime]]:
        """
        Gets the last processed year and date from existing match files.
        
        Args:
            match_folder_path (str): The path to the folder where match CSV files are saved.
            
        Returns:
            Tuple[Optional[int], Optional[datetime]]: The last processed year and date.
        """
        try:
            if not os.path.exists(match_folder_path):
                return None, None
                
            csv_files = [f for f in os.listdir(match_folder_path) if f.startswith('matches_') and f.endswith('.csv')]
            
            if not csv_files:
                return None, None
            
            # Get the latest year from file names
            years = [int(f.split('_')[-1].split('.')[0]) for f in csv_files]
            latest_year = max(years) if years else None
            
            # Get the latest date from the file for the latest year
            latest_date = None
            if latest_year:
                latest_file = os.path.join(match_folder_path, f"matches_{latest_year}.csv")
                try:
                    df = pd.read_csv(latest_file)
                    if not df.empty and 'date' in df.columns:
                        latest_date_str = df['date'].max()
                        try:
                            latest_date = datetime.strptime(latest_date_str, "%Y-%m-%d %H:%M")
                        except ValueError:
                            pass
                except Exception as e:
                    print(f"Error reading latest file: {e}")
            
            return latest_year, latest_date
        except Exception as e:
            print(f"Error determining last processed info: {e}")
            return None, None

    def _load_existing_lineup_keys(self, lineup_folder_path: str) -> None:
        """
        Loads existing lineup keys to avoid duplicates.
        
        Args:
            lineup_folder_path (str): The path to the folder where team lineup CSV files are saved.
        """
        try:
            if not os.path.exists(lineup_folder_path):
                return
                
            csv_files = [f for f in os.listdir(lineup_folder_path) if f.startswith('team_lineups_') and f.endswith('.csv')]
            
            for file in csv_files:
                file_path = os.path.join(lineup_folder_path, file)
                try:
                    df = pd.read_csv(file_path)
                    for _, row in df.iterrows():
                        key = self._generate_lineup_key(row['year'], row['date'], row['round_num'], row['team_name'])
                        self.processed_lineup_keys.add(key)
                except Exception as e:
                    print(f"Error loading existing lineup data from {file}: {e}")
        except Exception as e:
            print(f"Error loading existing lineup keys: {e}")

    def _find_game_links(self, soup: BeautifulSoup) -> List[str]:
        """
        Finds game links in the given soup.
        
        Args:
            soup (BeautifulSoup): The BeautifulSoup object.
            
        Returns:
            List[str]: The list of game links.
        """
        links = []
        base_url = "https://afltables.com/afl/"  # Correct base URL for AFLTables
        a_tags = soup.find_all('a', href=lambda href: href and 'stats/games/' in href)
        for a in a_tags:
            href = a['href']
            # Handle relative URLs starting with "../"
            if href.startswith('../'):
                full_link = base_url + href[3:]  # Remove "../" and append to base URL
            elif href.startswith('/'):
                full_link = base_url[:-1] + href  # Handle absolute paths from root
            else:
                full_link = base_url + href  # Handle relative paths without "../"
            links.append(full_link)
            print(f"Generated link: {full_link}")  # Debug print to verify URLs
        return links

    def _process_year(self, year: int, folder_path: str, last_processed_date: Optional[datetime] = None) -> None:
        """
        Processes a single year by scraping all match details and writing them to a CSV file.

        Args:
            year (int): The year to scrape.
            folder_path (str): The path to the folder where the CSV files will be saved.
            last_processed_date (Optional[datetime]): The last processed date.

        Returns:
            None
        """
        print(f"Processing year {year}...")
        year_soup: BeautifulSoup = get_soup(self.base_url + str(year) + '.html')
        game_links: List[str] = self._find_game_links(year_soup)
        match_data: List[Optional[Dict[str, Any]]] = []
        
        # Track processed match keys to avoid duplicates
        processed_match_keys = set()
        
        # Load existing match data if file exists
        file_path = os.path.join(folder_path, f"matches_{year}.csv")
        if os.path.exists(file_path):
            try:
                existing_df = pd.read_csv(file_path)
                for _, row in existing_df.iterrows():
                    if 'date' in row and 'round_num' in row:
                        key = f"{row['date']}_{row['round_num']}"
                        processed_match_keys.add(key)
            except Exception as e:
                print(f"Error loading existing match data for {year}: {e}")
        
        # Process each game link
        for link in game_links:
            match_info = self._extract_match_summary_table_data(link)
            if match_info:
                # Skip matches before the last processed date
                if last_processed_date and 'date' in match_info:
                    try:
                        match_date = datetime.strptime(match_info['date'], "%Y-%m-%d %H:%M")
                        if match_date <= last_processed_date:
                            continue
                    except ValueError:
                        pass
                
                # Check for duplicates using a unique key
                match_key = f"{match_info['date']}_{match_info['round_num']}"
                if match_key not in processed_match_keys:
                    match_data.append(match_info)
                    processed_match_keys.add(match_key)
        
        # Save match data
        if match_data:
            if os.path.exists(file_path):
                # Append to existing file
                existing_df = pd.read_csv(file_path)
                new_df = pd.DataFrame(match_data)
                combined_df = pd.concat([existing_df, new_df])
                # Remove duplicates based on date and round_num
                combined_df = combined_df.drop_duplicates(subset=['date', 'round_num'], keep='last')
                combined_df.to_csv(file_path, index=False)
            else:
                # Create new file
                df = pd.DataFrame(match_data)
                df.to_csv(file_path, index=False)
            
            print(f"Saved {len(match_data)} new matches for year {year}")

        # Self-check: audit per-round match counts so a silently-truncated round
        # (the "R10 2026" bug) is surfaced on every run instead of weeks later.
        # Runs even when no new matches were written this pass, so re-runs still
        # validate the on-disk file. Warnings only -- never aborts the pipeline.
        if os.path.exists(file_path):
            audit_match_rounds(file_path)

    def _process_team_lineups(self, lineup_folder_path: str) -> None:
        """
        Processes team lineups, merging with existing data.
        
        Args:
            lineup_folder_path (str): The path to the folder where team lineup CSV files are saved.
        """
        for team, value in self.team_lineups.items():
            # Skip invalid team names
            if not self._is_valid_team_name(team):
                print(f"Skipping invalid team name: {team}")
                continue
                
            team_filename = team.replace(' ', '_').lower()
            file_path = os.path.join(lineup_folder_path, f"team_lineups_{team_filename}.csv")
            
            # Deduplicate lineup data
            deduplicated_data = []
            for entry in value:
                key = self._generate_lineup_key(entry['year'], entry['date'], entry['round_num'], entry['team_name'])
                if key not in self.processed_lineup_keys:
                    deduplicated_data.append(entry)
                    self.processed_lineup_keys.add(key)
            
            # If no new data after deduplication, continue to next team
            if not deduplicated_data:
                continue
            
            # Convert player lists to strings
            new_df = pd.DataFrame(deduplicated_data)
            new_df['players'] = new_df['players'].apply(lambda x: ';'.join(x) if isinstance(x, list) else x)
            
            # Merge with existing data if file exists
            if os.path.exists(file_path):
                try:
                    existing_df = pd.read_csv(file_path)
                    combined_df = pd.concat([existing_df, new_df])
                    combined_df = combined_df.drop_duplicates(subset=['year', 'date', 'round_num', 'team_name'], keep='last')
                    combined_df.to_csv(file_path, index=False, encoding='utf-8')
                except Exception as e:
                    print(f"Error merging lineup data for {team}: {e}")
                    new_df.to_csv(file_path, index=False, encoding='utf-8')
            else:
                new_df.to_csv(file_path, index=False, encoding='utf-8')
            print(f"Saved lineup data for {team} to {file_path}")

    def _is_valid_team_name(self, team_name: str) -> bool:
        """
        Checks if a team name is valid.
        
        Args:
            team_name (str): The team name to check.
            
        Returns:
            bool: True if the team name is valid, False otherwise.
        """
        # Check if the team name contains digits and periods (like an IP address)
        if re.match(r'^\d+\.\d+\.\d+', team_name):
            return False
        
        # Check if the team name is too short
        if len(team_name) < 3:
            return False
        
        # Check if the team name is mostly digits
        if sum(c.isdigit() for c in team_name) > len(team_name) / 2:
            return False
        
        # Check for known AFL team names (including historical teams)
        known_teams = [
            "adelaide", "brisbane", "carlton", "collingwood", "essendon", 
            "fremantle", "geelong", "gold coast", "greater western sydney", 
            "hawthorn", "melbourne", "north melbourne", "port adelaide", 
            "richmond", "st kilda", "sydney", "west coast", "western bulldogs",
            # Historical teams
            "fitzroy", "south melbourne", "brisbane bears", "footscray",
            "kangaroos", "north melbourne kangaroos", "university", 
            "st. kilda", "gws giants"
        ]
        
        team_lower = team_name.lower()
        return any(team in team_lower for team in known_teams)

    def _generate_lineup_key(self, year: str, date: str, round_num: str, team_name: str) -> str:
        """
        Generates a unique key for a lineup entry.
        
        Args:
            year (str): The year.
            date (str): The date.
            round_num (str): The round number.
            team_name (str): The team name.
            
        Returns:
            str: A unique key for the lineup entry.
        """
        key_str = f"{year}_{date}_{round_num}_{team_name}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _extract_match_summary_table_data(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Extracts match summary table data from the given URL.

        Args:
            url (str): The URL of the match summary page.

        Returns:
            Optional[Dict[str, Any]]: The extracted match data.
        """
        try:
            soup: BeautifulSoup = get_soup(url)
            tables = soup.find_all('table')
            
            if not tables:
                return None
                
            td_elements = tables[0].find_all('td')
            data_list: List[str] = [elem.text.strip() for elem in td_elements]
            
            if len(data_list) < 13:  # Basic validation
                return None

            data: Dict[str, Any] = self._extract_match_details(data_list)
            
            team_detail_tables: List[BeautifulSoup] = soup.find_all('table', class_='sortable')
            if len(team_detail_tables) < 2:
                return data
                
            team_lineup: Dict[str, List[str]] = {}
            teams: List[Dict[str, Union[str, int]]] = []
            team_data: List[str] = data_list[3:13]
            
            table_num: int = 0
            for i in range(0, len(team_data), 5):
                if i + 4 >= len(team_data):
                    break
                    
                team: Dict[str, Union[str, int, List[str]]] = {}
                team['team_name'] = team_data[i]
                
                if table_num < len(team_detail_tables):
                    team_lineup[team['team_name']] = self._extract_player_names(team_detail_tables[table_num])
                
                team['q1_goals'], team['q1_behinds'] = self._parse_score(team_data[i+1])
                team['q2_goals'], team['q2_behinds'] = self._parse_score(team_data[i+2])
                team['q3_goals'], team['q3_behinds'] = self._parse_score(team_data[i+3])
                team['final_goals'], team['final_behinds'] = self._parse_score(team_data[i+4])
                
                teams.append(team)
                table_num += 1

            data.update({f'team_{i+1}_{k}': v for i, team in enumerate(teams) for k, v in team.items()})
            self._add_team_lineups(data, team_lineup)
            
            return data
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return None

    def _parse_score(self, score_str: str) -> Tuple[int, int]:
        """
        Parses a score string into goals and behinds.
        
        Args:
            score_str (str): The score string to parse.
            
        Returns:
            Tuple[int, int]: The goals and behinds.
        """
        parts = score_str.split('.')
        if len(parts) >= 2:
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                return 0, 0
        return 0, 0

    def _extract_match_details(self, data_list: List[str]) -> Dict[str, Any]:
        """
        Extracts match details from the given data list.

        Args:
            data_list (List[str]): The list of data elements.

        Returns:
            Dict[str, Any]: The extracted match details.
        """
        pattern: str = r"Round: (.+) Venue: (.+) Date: (\w+, \d+-\w+-\d{4} \d{1,2}:\d{2} (?:AM|PM))(?: \(\d{1,2}:\d{2} (?:AM|PM)\))?(?: Attendance: (\d+))?"       

        data: Dict[str, Any] = {}

        match: Optional[re.Match] = re.search(pattern, data_list[1])
        if match:
            data['round_num'] = match.group(1)
            data['venue'] = match.group(2)
            try:
                date_str = match.group(3).split('(')[0].strip()
                data['date'] = datetime.strptime(date_str, "%a, %d-%b-%Y %I:%M %p").strftime("%Y-%m-%d %H:%M")
                data['year'] = data['date'][:4]
            except ValueError:
                # Fallback for date parsing issues
                data['date'] = match.group(3)
                data['year'] = match.group(3).split('-')[-1][:4]
                
            data['attendance'] = match.group(4) if match.group(4) else "N/A"
        else:
            print(f"No match found in: {data_list[1]}")
            data['round_num'] = "Unknown"
            data['venue'] = "Unknown"
            data['date'] = "Unknown"
            data['year'] = "Unknown"
            data['attendance'] = "N/A"

        return data

    def _add_team_lineups(self, data: Dict[str, Any], team_lineup: Dict[str, List[str]]) -> None:
        """
        Adds team lineups to the data.

        Args:
            data (Dict[str, Any]): The match data.
            team_lineup (Dict[str, List[str]]): The team lineup data.

        Returns:
            None
        """
        for team, players in team_lineup.items():
            if not self._is_valid_team_name(team):
                continue
            lineup_entry = {
                'year': data['year'],
                'date': data['date'],
                'round_num': data['round_num'],
                'team_name': team,
                'players': players
            }
            if team not in self.team_lineups:
                self.team_lineups[team] = []
            self.team_lineups[team].append(lineup_entry)

    def _extract_player_names(self, table: BeautifulSoup) -> List[str]:
        """
        Extracts player names from a team detail table.

        Args:
            table (BeautifulSoup): The BeautifulSoup table object containing player data.

        Returns:
            List[str]: A list of player names.
        """
        try:
            player_names = []
            rows = table.find_all('tr')[1:]  # Skip header row
            for row in rows:
                cells = row.find_all('td')
                if cells and len(cells) > 0:
                    player_name = cells[0].text.strip()  # Assuming first column is player name
                    if player_name:
                        player_names.append(player_name)
            return player_names
        except Exception as e:
            print(f"Error extracting player names: {e}")
            return []

if __name__ == "__main__":
    # Standalone audit mode: `python game_scraper.py --audit [matches_dir]`
    # Audits every matches_<year>.csv (or just the given files) without scraping.
    if len(sys.argv) > 1 and sys.argv[1] == "--audit":
        import glob
        targets = sys.argv[2:]
        if not targets:
            targets = sorted(glob.glob(os.path.join("./data/matches", "matches_*.csv")))
        elif len(targets) == 1 and os.path.isdir(targets[0]):
            targets = sorted(glob.glob(os.path.join(targets[0], "matches_*.csv")))
        total_warnings = 0
        for path in targets:
            issues = audit_match_rounds(path)
            total_warnings += sum(1 for i in issues if i["severity"] == "WARNING")
        if total_warnings:
            print(f"[match-audit] DONE: {total_warnings} probable-gap WARNING(s) across {len(targets)} file(s)")
        else:
            print(f"[match-audit] DONE: no probable gaps across {len(targets)} file(s)")
        sys.exit(0)

    # Standalone player-totals audit: `python game_scraper.py --audit-players [glob]`
    # Reconciles each player CSV's career totals against its afltables profile
    # without scraping. Honours the existing 0.5s inter-request delay.
    if len(sys.argv) > 1 and sys.argv[1] == "--audit-players":
        import glob
        import time
        pattern = sys.argv[2] if len(sys.argv) > 2 else "data/player_data/*performance*.csv"
        targets = sorted(glob.glob(pattern))
        total_warnings = 0
        for path in targets:
            issues = audit_player_career_totals(path)
            total_warnings += sum(1 for i in issues if i["severity"] == "WARNING")
            time.sleep(0.5)
        if total_warnings:
            print(f"[player-audit] DONE: {total_warnings} reconciliation WARNING(s) across {len(targets)} file(s)")
        else:
            print(f"[player-audit] DONE: no reconciliation issues across {len(targets)} file(s)")
        sys.exit(0)

    scraper = MatchScraper()
    scraper.scrape_all_matches()  # Uses default paths "./data/matches" and "./data/lineups"