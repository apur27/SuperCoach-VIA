import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

# Set up logging for diagnostics
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

# Define eras with year ranges and available statistics
ERAS = {
    'pre_1965': (1897, 1964, ['goals', 'behinds']),
    '1965_1990': (1965, 1990, ['goals', 'behinds', 'kicks', 'handballs']),
    '1990_2010': (1991, 2010, ['goals', 'behinds', 'kicks', 'handballs', 'marks', 'disposals']),
    'post_2010': (2011, 2025, ['goals', 'behinds', 'kicks', 'handballs', 'marks', 'disposals', 'tackles', 'one_percenters', 'clearances', 'contested_possessions', 'contested_marks', 'goal_assist'])
}

def initialize_df_lib():
    """Initialize DataFrame library (cuDF or pandas) based on GPU availability."""
    try:
        import cudf
        cudf.Series([1])  # Lightweight GPU initialization test
        logging.info("GPU acceleration enabled via CuDF")
        return cudf, True
    except (ImportError, RuntimeError, OSError, AttributeError) as e:
        logging.info(f"GPU unavailable: {e}. Falling back to CPU with pandas.")
        return pd, False

df_lib, USE_GPU = initialize_df_lib()

def get_era(year: int) -> Tuple[str, List[str]]:
    """Determine the era and available statistics for a given year."""
    for era_name, (start, end, stats) in ERAS.items():
        if start <= year <= end:
            return era_name, stats
    return 'unknown', []

def safe_sum(df, col: str, df_lib) -> float:
    """Safely sum a column after converting to numeric, optimized for GPU/CPU."""
    series = df[col]
    if USE_GPU:
        try:
            series = series.astype('float64')
        except Exception as e:
            logging.error(f"cuDF conversion failed for {col}: {e}")
            series = series.applymap(lambda x: float(x) if x else 0)
        return series.sum()
    else:
        return pd.to_numeric(series, errors='coerce').fillna(0).sum()

def parse_dates(df, df_lib, col='date'):
    """Parse dates using the appropriate library (cudf or pandas)."""
    if USE_GPU:
        try:
            df[col] = df_lib.to_datetime(df[col])
        except Exception as e:
            logging.warning(f"cuDF date parsing failed: {e}. Using pandas fallback.")
            dates_cpu = pd.to_datetime(df[col].to_pandas(), errors='coerce')
            df[col] = df_lib.Series(dates_cpu)
    else:
        df[col] = df_lib.to_datetime(df[col], errors='coerce')
    return df.dropna(subset=[col])

def process_player_file(filepath: str, year: int, weights: dict, df_lib) -> Optional[Tuple[str, int, Dict[str, int], int]]:
    """Process a single player's performance file for a given year."""
    try:
        # Read only the 'date' column first to check for the target year
        if USE_GPU:
            df_dates = df_lib.read_csv(filepath, usecols=['date'])
        else:
            df_dates = df_lib.read_csv(filepath, usecols=['date'], low_memory=False)
        
        df_dates = parse_dates(df_dates, df_lib, 'date')
        df_dates['year'] = df_dates['date'].dt.year
        
        # Skip if the target year isn't present
        if year not in (df_dates['year'].values_host if USE_GPU else df_dates['year'].values):
            logging.debug(f"No games found for {filepath} in {year}")
            return None
        
        # Load the full DataFrame only if the year is relevant
        if USE_GPU:
            df = df_lib.read_csv(filepath)
        else:
            df = df_lib.read_csv(filepath, low_memory=False)
        
        df = parse_dates(df, df_lib, 'date')
        df['year'] = df['date'].dt.year
        df = df[df['year'] == year]
        
        if df.empty:
            logging.debug(f"No games found for {filepath} in {year} after full load")
            return None
        
        player_name = "_".join(os.path.basename(filepath).split("_")[:-2])
        era, era_stats = get_era(year)
        available_stats = [stat for stat in era_stats if stat in df.columns]
        if not available_stats:
            logging.warning(f"No era-specific stats ({era_stats}) found in {filepath}")
            return None
        
        score = 0
        totals = {}
        for stat in available_stats:
            total = safe_sum(df, stat, df_lib)
            totals[stat] = int(total)
            score += total * weights.get(stat, 0)
        
        games_played = len(df)
        return (player_name, int(score), totals, games_played)
    
    except Exception as e:
        logging.error(f"Error processing {filepath}: {e}")
        return None

def calculate_percentile_ranks(scores: List[int]) -> np.ndarray:
    """Efficiently calculate percentile ranks using pandas."""
    return pd.Series(scores).rank(pct=True).to_numpy() * 100

def create_ranking_dataframe(players_data: List[Tuple[str, int, float, int]]) -> pd.DataFrame:
    """Create a standardized DataFrame for rankings."""
    return pd.DataFrame(players_data, columns=['player', 'score', 'percentile_rank', 'games_played'])

def generate_yearly_top_100(data_dir: str, year: int, weights: dict, output_dir: str, df_lib) -> List[Tuple[str, int, Dict[str, int], int, float]]:
    """Generate the top 100 players for a specific year."""
    player_scores = []
    for filename in os.listdir(data_dir):
        if filename.endswith("_performance_details.csv"):
            filepath = os.path.join(data_dir, filename)
            result = process_player_file(filepath, year, weights, df_lib)
            if result:
                player_scores.append(result)
    
    if not player_scores:
        logging.info(f"No player data found for year {year}")
        return []
    
    scores = [score for _, score, _, _ in player_scores]
    percentile_ranks = calculate_percentile_ranks(scores)
    
    sorted_players = sorted(
        zip(player_scores, percentile_ranks),
        key=lambda x: (x[0][1], x[0][3]),  # Sort by score, then games played
        reverse=True
    )
    
    top_100 = sorted_players[:100]
    
    df = create_ranking_dataframe([(p[0], p[1], pr, p[3]) for p, pr in top_100])
    df.to_csv(os.path.join(output_dir, 'yearly', f'year_{year}.csv'), index=False)
    return [(p[0], p[1], p[2], p[3], pr) for p, pr in top_100]

def compile_all_time_top_100(yearly_top_100: Dict[int, List[Tuple[str, int, Dict[str, int], int, float]]], output_dir: str) -> None:
    """Compile the all-time top 100 players based on yearly rankings."""
    player_data = {}
    for year, top_list in yearly_top_100.items():
        for player, score, totals, games, percentile in top_list:
            if player not in player_data:
                player_data[player] = {'percentile_ranks': [], 'seasons': 0}
            player_data[player]['percentile_ranks'].append(percentile)
            player_data[player]['seasons'] += 1
    
    all_time_scores = []
    for player, data in player_data.items():
        if data['seasons'] > 0:
            avg_percentile = np.mean(data['percentile_ranks'])
            seasons_in_top_100 = data['seasons']
            all_time_score = avg_percentile * seasons_in_top_100  # Balances peak performance and longevity
            all_time_scores.append((player, all_time_score))
    
    if not all_time_scores:
        logging.info("No all-time top 100 players found")
        return
    
    sorted_all_time = sorted(all_time_scores, key=lambda x: x[1], reverse=True)[:100]
    df = pd.DataFrame(sorted_all_time, columns=['player', 'all_time_score'])
    df.to_csv(os.path.join(output_dir, 'all_time_top_100.csv'), index=False)

def main():
    """Main function to generate top 100 rankings across all time frames."""
    data_dir = "./data/player_data/"
    output_dir = "./data/top100/"
    os.makedirs(os.path.join(output_dir, 'yearly'), exist_ok=True)
    
    weights = {
        'goals': 60.0,
        'behinds': 1.0,
        'disposals': 12.0,
        'goal_assist': 3.0,
        'contested_marks': 6.0,
        'contested_possessions': 4.5,
        'marks': 2.0,
        'kicks': 4.0,
        'tackles': 2.0,
        'one_percenters': 2.0,
        'clearances': 4.0
    }
    
    yearly_top_100 = {}
    for year in range(1897, 2026):
        logging.info(f"Processing yearly rankings for {year}")
        top_100 = generate_yearly_top_100(data_dir, year, weights, output_dir, df_lib)
        if top_100:
            yearly_top_100[year] = top_100
    
    compile_all_time_top_100(yearly_top_100, output_dir)

if __name__ == "__main__":
    main()