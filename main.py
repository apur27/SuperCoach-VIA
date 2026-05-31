from scrapers.game_scraper import MatchScraper
from scrapers.player_scraper import PlayerScraper
from datetime import datetime

import config

def main() -> None:
    match_scraper: MatchScraper = MatchScraper()
    player_scraper: PlayerScraper = PlayerScraper()

    match_scraper.scrape_all_matches(
        match_folder_path=config.MATCHES_DIR,
        lineup_folder_path=config.LINEUPS_DIR
    )
    player_scraper.scrape_all_players(
        folder_path=config.PLAYER_DATA_DIR
    )

if __name__ == "__main__":
    main()