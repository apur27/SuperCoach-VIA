from game_scraper import MatchScraper
from player_scraper import PlayerScraper
from datetime import datetime

def main() -> None:
    folder_path = "data/"

    match_scraper: MatchScraper = MatchScraper()
    player_scraper: PlayerScraper = PlayerScraper()

    match_scraper.scrape_all_matches(
        match_folder_path=folder_path + "matches",
        lineup_folder_path=folder_path + "lineups"
    )
    player_scraper.scrape_all_players(
        folder_path=folder_path + "player_data"
    )

if __name__ == "__main__":
    main()