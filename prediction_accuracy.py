import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime
import csv

# Define directories and file paths
DATA_DIR = Path("./data/player_data")
PREDICTION_DIR = Path("./data/prediction")
PREDICTION_FILE = PREDICTION_DIR / "next_round_18_prediction_20250630_2013.csv"

# Extract round number from prediction file name
match = re.search(r'next_round_(\d+)_prediction_\d{8}_\d{4}\.csv', PREDICTION_FILE.name)
ROUND_NUM = int(match.group(1)) if match else 16  # Default to 16 if not found

def find_player_files(player_name, data_dir):
    """
    Finds all potential performance details CSV files for a given player.
    Handles variations in name formatting.

    Args:
        player_name (str): Player name in "Last First" format (e.g., "Murphy Reid").
        data_dir (Path): Directory containing player data files.

    Returns:
        list: List of Path objects for matching player files.
    """
    parts = player_name.split()
    if len(parts) < 2:
        print(f"Warning: Invalid player name format: {player_name}")
        return []
    last, first = parts[0].lower().strip(), parts[1].lower().strip()
    pattern = f"{last}_{first}_*_performance_details.csv"
    files = list(data_dir.glob(pattern))
    if not files:
        print(f"Warning: No performance files found for {player_name}")
    return files

def detect_delimiter(file_path):
    """
    Detects the delimiter used in the CSV file by inspecting the first line.

    Args:
        file_path (Path): Path to the CSV file.

    Returns:
        str: Detected delimiter (',' or '\t').
    """
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        if '\t' in first_line:
            return '\t'
        elif ',' in first_line:
            return ','
        else:
            print(f"Warning: Could not determine delimiter for {file_path}, defaulting to ','")
            return ','  # Default to comma if unclear

def get_actual_disposals(file_path, year, round_num, expected_team):
    """
    Retrieves actual disposals from a player's performance CSV for a given year and round,
    ensuring the team matches.

    Args:
        file_path (Path): Path to the player's performance CSV.
        year (int): Year to filter (e.g., 2025).
        round_num (int): Round number to filter (e.g., 16).
        expected_team (str): Expected team name.

    Returns:
        float or np.nan: Actual disposals or NaN if data is missing or team doesn't match.
    """
    try:
        # Detect delimiter and read CSV
        delimiter = detect_delimiter(file_path)
        df = pd.read_csv(file_path, sep=delimiter, na_values=['NA', 'N/A', '', 'nan'])
        
        # Verify required columns
        required_cols = ['team', 'year', 'round', 'disposals']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: Missing required columns in {file_path}. Found: {df.columns.tolist()}")
            return np.nan
        
        # Convert columns to appropriate types
        df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int16', errors='ignore')
        df['round'] = pd.to_numeric(df['round'], errors='coerce')
        df['disposals'] = pd.to_numeric(df['disposals'], errors='coerce')
        
        # Drop rows with NaN in critical columns
        df = df.dropna(subset=['year', 'round', 'disposals', 'team'])
        
        # Filter for the specific year, round, and team
        game_row = df[(df['year'] == year) & 
                      (df['round'] == round_num) & 
                      (df['team'].str.lower() == expected_team.lower())]
        
        if game_row.empty:
            print(f"No data found for {expected_team}, year {year}, round {round_num} in {file_path}")
            return np.nan
        
        return game_row['disposals'].iloc[0]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.nan

def main():
    """
    Main function to assess prediction accuracy and save results.
    """
    # Check if directories and files exist
    if not PREDICTION_DIR.exists():
        print(f"Error: Prediction directory {PREDICTION_DIR} does not exist.")
        return
    if not PREDICTION_FILE.exists():
        print(f"Error: Prediction file {PREDICTION_FILE} does not exist.")
        return

    # Load prediction CSV
    try:
        prediction_df = pd.read_csv(PREDICTION_FILE)
        print(f"Loaded prediction CSV: {PREDICTION_FILE}")
    except Exception as e:
        print(f"Error reading {PREDICTION_FILE}: {e}")
        return

    # Verify required columns in prediction CSV
    required_cols = ['player', 'team', 'predicted_disposals']
    if not all(col in prediction_df.columns for col in required_cols):
        print(f"Error: Prediction CSV missing required columns. Found: {prediction_df.columns.tolist()}")
        return

    # Initialize result columns
    prediction_df['actual_disposals'] = np.nan
    prediction_df['difference'] = np.nan

    # Process each player
    for idx, row in prediction_df.iterrows():
        player = row['player']
        team = row['team']
        predicted = row['predicted_disposals']
        
        # Find player files
        file_paths = find_player_files(player, DATA_DIR)
        if not file_paths:
            continue
        
        # Check each file for matching data
        for file_path in file_paths:
            actual = get_actual_disposals(file_path, 2025, ROUND_NUM, team)
            if not pd.isna(actual):
                prediction_df.at[idx, 'actual_disposals'] = actual
                prediction_df.at[idx, 'difference'] = predicted - actual
                break
        else:
            print(f"Warning: No matching performance data for {player} in {team} for Round {ROUND_NUM}, 2025")

    # Calculate Mean Absolute Error (MAE)
    valid_rows = prediction_df.dropna(subset=['actual_disposals'])
    if not valid_rows.empty:
        mae = valid_rows['difference'].abs().mean()
        print(f"Mean Absolute Error: {mae:.2f}")
    else:
        print("No actual data available to calculate MAE.")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = PREDICTION_DIR / f"prediction_vs_actual_round_{ROUND_NUM}_2025_{timestamp}.csv"
    prediction_df[['player', 'team', 'predicted_disposals', 
                   'actual_disposals', 'difference']].to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")

    # Print summary
    num_with_data = valid_rows.shape[0]
    num_total = prediction_df.shape[0]
    print(f"Players with actual data: {num_with_data}/{num_total}")

if __name__ == "__main__":
    main()