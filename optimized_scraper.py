import asyncio
import aiohttp
import os
import re
import hashlib
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import concurrent.futures
from pathlib import Path
import time
import pickle

class OptimizedScraper:
    """Optimized scraper with async requests, caching, and improved performance."""
    
    def __init__(self, cache_dir: str = "cache", use_cache: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.use_cache = use_cache
        self.session = None
        self.base_url = "https://afltables.com/afl/"
        
        # Rate limiting
        self.request_delay = 0.1  # Reduced from 0.5s
        self.max_concurrent_requests = 20  # Increased from 10
        
        # Team URLs for player scraping
        self.team_urls = [
            'https://afltables.com/afl/stats/alltime/adelaide.html',
            'https://afltables.com/afl/stats/alltime/brisbanel.html',
            'https://afltables.com/afl/stats/alltime/brisbaneb.html',
            'https://afltables.com/afl/stats/alltime/carlton.html',
            'https://afltables.com/afl/stats/alltime/collingwood.html',
            'https://afltables.com/afl/stats/alltime/essendon.html',
            'https://afltables.com/afl/stats/alltime/fitzroy.html',
            'https://afltables.com/afl/stats/alltime/fremantle.html',
            'https://afltables.com/afl/stats/alltime/geelong.html',
            'https://afltables.com/afl/stats/alltime/goldcoast.html',
            'https://afltables.com/afl/stats/alltime/gws.html',
            'https://afltables.com/afl/stats/alltime/hawthorn.html',
            'https://afltables.com/afl/stats/alltime/melbourne.html',
            'https://afltables.com/afl/stats/alltime/kangaroos.html',
            'https://afltables.com/afl/stats/alltime/padelaide.html',
            'https://afltables.com/afl/stats/alltime/richmond.html',
            'https://afltables.com/afl/stats/alltime/stkilda.html',
            'https://afltables.com/afl/stats/alltime/swans.html',
            'https://afltables.com/afl/stats/alltime/westcoast.html',
            'https://afltables.com/afl/stats/alltime/bullldogs.html',
            'https://afltables.com/afl/stats/alltime/university.html'
        ]

    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for URL."""
        return hashlib.md5(url.encode()).hexdigest()[:16] + ".pkl"

    def _load_from_cache(self, cache_key: str):
        """Load cached response."""
        if not self.use_cache:
            return None
        cache_file = self.cache_dir / cache_key
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                # Check if cache is less than 1 day old
                if time.time() - cached_data['timestamp'] < 86400:
                    return cached_data['content']
            except:
                pass
        return None

    def _save_to_cache(self, content: str, cache_key: str):
        """Save response to cache."""
        if not self.use_cache:
            return
        cache_file = self.cache_dir / cache_key
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'content': content,
                    'timestamp': time.time()
                }, f)
        except:
            pass

    async def fetch_url(self, session: aiohttp.ClientSession, url: str) -> str:
        """Fetch URL with caching and error handling."""
        cache_key = self._get_cache_key(url)
        
        # Try cache first
        cached_content = self._load_from_cache(cache_key)
        if cached_content:
            return cached_content

        try:
            await asyncio.sleep(self.request_delay)  # Rate limiting
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    content = await response.text()
                    self._save_to_cache(content, cache_key)
                    return content
                else:
                    print(f"HTTP {response.status} for {url}")
                    return ""
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return ""

    def parse_html(self, html_content: str) -> BeautifulSoup:
        """Parse HTML content with BeautifulSoup."""
        return BeautifulSoup(html_content, 'html.parser')

    async def scrape_player_links(self) -> List[str]:
        """Scrape all player links from team pages asynchronously."""
        print("🔍 Scraping player links...")
        
        # Check cache for player links
        cache_key = "all_player_links.pkl"
        cache_file = self.cache_dir / cache_key
        
        if self.use_cache and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                if time.time() - cached_data['timestamp'] < 86400:  # Cache for 1 day
                    print(f"📋 Loaded {len(cached_data['links'])} player links from cache")
                    return cached_data['links']
            except:
                pass

        player_links = set()
        
        async with aiohttp.ClientSession() as session:
            # Fetch all team pages concurrently
            tasks = [self.fetch_url(session, url) for url in self.team_urls]
            team_pages = await asyncio.gather(*tasks)
            
            for html_content in team_pages:
                if html_content:
                    soup = self.parse_html(html_content)
                    links = soup.select('a[href*="/players/"]')
                    
                    for link in links:
                        href = link['href']
                        if '/players/' in href:
                            normalized_href = href.replace('../', '')
                            player_links.add(normalized_href)

        player_links_list = list(player_links)
        
        # Cache the results
        if self.use_cache:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'links': player_links_list,
                        'timestamp': time.time()
                    }, f)
            except:
                pass
        
        print(f"🎯 Found {len(player_links_list)} unique player links")
        return player_links_list

    async def scrape_game_links(self, year: int) -> List[str]:
        """Scrape game links for a specific year."""
        year_url = f"{self.base_url}seas/{year}.html"
        
        async with aiohttp.ClientSession() as session:
            html_content = await self.fetch_url(session, year_url)
            
        if not html_content:
            return []
            
        soup = self.parse_html(html_content)
        game_links = []
        
        # Find game links
        for link in soup.find_all('a', href=lambda href: href and 'stats/games/' in href):
            href = link['href']
            if href.startswith('../'):
                full_link = self.base_url + href[3:]
            elif href.startswith('/'):
                full_link = self.base_url[:-1] + href
            else:
                full_link = self.base_url + href
            game_links.append(full_link)
            
        return game_links

    def extract_player_data(self, html_content: str, player_link: str) -> Dict:
        """Extract player personal and performance data from HTML."""
        soup = self.parse_html(html_content)
        
        # Extract personal details
        h1_tag = soup.find('h1')
        if not h1_tag:
            return {}
            
        full_name = h1_tag.text.strip().split()
        
        # Extract birth date
        born_tag = soup.find('b', string='Born:')
        born_date_str = born_tag.next_sibling.strip().rstrip(' (') if born_tag else '01-Jan-1900'
        
        try:
            born_date = datetime.strptime(born_date_str, "%d-%b-%Y")
        except ValueError:
            born_date = datetime(1900, 1, 1)

        # Extract performance data from tables
        performance_data = []
        tables = soup.find_all('table')
        
        for table in tables:
            th_colspan_28 = table.find('th', colspan="28")
            if not th_colspan_28:
                continue

            header_text = th_colspan_28.text.strip()
            try:
                team, year = header_text.split(' - ')
                year_int = int(year)
            except ValueError:
                continue

            tbody = table.find('tbody')
            if not tbody:
                continue

            for row in tbody.find_all('tr'):
                cells = [td.text.strip() for td in row.find_all('td')]
                if len(cells) >= 26:
                    round_str = cells[2]
                    round_num = int(re.sub(r'\D', '', round_str)) if re.sub(r'\D', '', round_str) else 1
                    game_date = datetime(year_int, 3, 1) + timedelta(weeks=round_num - 1)
                    
                    row_data = {
                        'team': team,
                        'year': year_int,
                        'round': round_str,
                        'date': game_date.strftime("%Y-%m-%d"),
                        'disposals': cells[10] if len(cells) > 10 else '',
                        'kicks': cells[7] if len(cells) > 7 else '',
                        'handballs': cells[9] if len(cells) > 9 else '',
                        'goals': cells[11] if len(cells) > 11 else '',
                        'behinds': cells[12] if len(cells) > 12 else '',
                    }
                    performance_data.append(row_data)

        return {
            'personal': {
                'first_name': full_name[0] if full_name else 'Unknown',
                'last_name': full_name[-1] if len(full_name) > 1 else 'Unknown',
                'born_date': born_date.strftime('%d-%m-%Y'),
            },
            'performance': performance_data
        }

    async def scrape_all_players_optimized(self, output_dir: str):
        """Scrape all players with optimized async processing."""
        print("🚀 Starting optimized player scraping...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get all player links
        player_links = await self.scrape_player_links()
        
        if not player_links:
            print("❌ No player links found")
            return

        print(f"📥 Processing {len(player_links)} players...")
        
        # Process players in batches to avoid overwhelming the server
        batch_size = self.max_concurrent_requests
        processed_count = 0
        
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(player_links), batch_size):
                batch = player_links[i:i + batch_size]
                
                # Create tasks for the batch
                tasks = []
                for player_link in batch:
                    full_url = f"{self.base_url}stats/{player_link}"
                    task = self.fetch_url(session, full_url)
                    tasks.append((task, player_link))
                
                # Execute batch
                for task, player_link in tasks:
                    html_content = await task
                    if html_content:
                        try:
                            player_data = self.extract_player_data(html_content, player_link)
                            if player_data and player_data.get('performance'):
                                self.save_player_data(player_data, output_path)
                                processed_count += 1
                        except Exception as e:
                            print(f"Error processing {player_link}: {e}")
                
                print(f"📊 Processed {processed_count}/{len(player_links)} players")

        print(f"✅ Completed scraping {processed_count} players")

    def save_player_data(self, player_data: Dict, output_path: Path):
        """Save player data to CSV files."""
        personal = player_data['personal']
        performance = player_data['performance']
        
        if not performance:
            return
            
        # Generate filename
        born_date = datetime.strptime(personal['born_date'], '%d-%m-%Y')
        filename_base = f"{personal['last_name']}_{personal['first_name']}_{born_date.strftime('%d%m%Y')}".lower()
        
        # Save personal details
        personal_file = output_path / f"{filename_base}_personal_details.csv"
        personal_df = pd.DataFrame([personal])
        personal_df.to_csv(personal_file, index=False)
        
        # Save performance details
        performance_file = output_path / f"{filename_base}_performance_details.csv"
        performance_df = pd.DataFrame(performance)
        
        # Merge with existing data if file exists
        if performance_file.exists():
            try:
                existing_df = pd.read_csv(performance_file)
                combined_df = pd.concat([existing_df, performance_df])
                combined_df = combined_df.drop_duplicates(subset=['team', 'year', 'round'], keep='last')
                combined_df.to_csv(performance_file, index=False)
            except:
                performance_df.to_csv(performance_file, index=False)
        else:
            performance_df.to_csv(performance_file, index=False)

    async def scrape_matches_optimized(self, start_year: int, end_year: int, output_dir: str):
        """Scrape match data with optimized processing."""
        print(f"🏈 Starting optimized match scraping ({start_year}-{end_year})...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for year in range(start_year, end_year + 1):
            print(f"📅 Processing year {year}...")
            
            # Get game links for the year
            game_links = await self.scrape_game_links(year)
            
            if not game_links:
                print(f"⚠️ No games found for {year}")
                continue
            
            # Process games in batches
            match_data = []
            batch_size = self.max_concurrent_requests
            
            async with aiohttp.ClientSession() as session:
                for i in range(0, len(game_links), batch_size):
                    batch = game_links[i:i + batch_size]
                    
                    # Fetch all games in batch
                    tasks = [self.fetch_url(session, link) for link in batch]
                    html_contents = await asyncio.gather(*tasks)
                    
                    # Process each game
                    for html_content, link in zip(html_contents, batch):
                        if html_content:
                            match_info = self.extract_match_data(html_content)
                            if match_info:
                                match_data.append(match_info)
            
            # Save year's data
            if match_data:
                file_path = output_path / f"matches_{year}.csv"
                df = pd.DataFrame(match_data)
                df.to_csv(file_path, index=False)
                print(f"💾 Saved {len(match_data)} matches for {year}")

    def extract_match_data(self, html_content: str) -> Optional[Dict]:
        """Extract match data from HTML content."""
        soup = self.parse_html(html_content)
        tables = soup.find_all('table')
        
        if not tables:
            return None
            
        td_elements = tables[0].find_all('td')
        data_list = [elem.text.strip() for elem in td_elements]
        
        if len(data_list) < 13:
            return None

        # Extract match details using regex
        pattern = r"Round: (.+) Venue: (.+) Date: (\w+, \d+-\w+-\d{4} \d{1,2}:\d{2} (?:AM|PM))(?: \(\d{1,2}:\d{2} (?:AM|PM)\))?(?: Attendance: (\d+))?"
        
        match_data = {}
        match = re.search(pattern, data_list[1])
        
        if match:
            match_data['round_num'] = match.group(1)
            match_data['venue'] = match.group(2)
            try:
                date_str = match.group(3).split('(')[0].strip()
                match_data['date'] = datetime.strptime(date_str, "%a, %d-%b-%Y %I:%M %p").strftime("%Y-%m-%d %H:%M")
                match_data['year'] = match_data['date'][:4]
            except ValueError:
                match_data['date'] = match.group(3)
                match_data['year'] = match.group(3).split('-')[-1][:4]
            match_data['attendance'] = match.group(4) if match.group(4) else "N/A"
        
        return match_data

    def run_full_optimization(self, start_year: int = 2020, end_year: int = None):
        """Run complete optimized scraping pipeline."""
        if end_year is None:
            end_year = datetime.now().year
            
        print("🚀 Starting Full Optimization Pipeline...")
        
        # Create output directories
        os.makedirs("data/matches", exist_ok=True)
        os.makedirs("data/player_data", exist_ok=True)
        
        # Run async operations
        asyncio.run(self._run_async_pipeline(start_year, end_year))
        
        print("✅ Optimization pipeline completed!")

    async def _run_async_pipeline(self, start_year: int, end_year: int):
        """Run the async scraping pipeline."""
        # Run both match and player scraping concurrently
        tasks = [
            self.scrape_matches_optimized(start_year, end_year, "data/matches"),
            self.scrape_all_players_optimized("data/player_data")
        ]
        
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    scraper = OptimizedScraper(use_cache=True)
    scraper.run_full_optimization(start_year=2023, end_year=2025)