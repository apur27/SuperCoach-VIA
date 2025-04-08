import csv
import os
from statistics import median

def read_player_data(player_id):
    """
    Read the player's data from their personal details CSV file in data/player_data.
    Returns a dictionary with the player's info or None if the file is missing or empty.
    """
    file_path = f"data/player_data/{player_id}_personal_details.csv"
    print(f"Looking for personal details file: {file_path}")
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return None
    try:
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            row = next(reader)
            required_columns = ['first_name', 'last_name', 'born_date', 'debut_date', 'height', 'weight']
            if not all(col in row for col in required_columns):
                print(f"Missing columns in {file_path}")
                return None
            return row
    except StopIteration:
        print(f"File is empty: {file_path}")
        return None

def read_performance_data(player_id):
    """
    Read the player's performance data from their performance CSV file.
    Returns a dictionary with teams in chronological order, total games, total goals,
    total disposals, average disposals per game, total Brownlow votes, games with 20+ disposals,
    games with 3+ goals, number of seasons, peak disposals, peak goals, median disposals,
    impact score, and consistency score. Returns a default dictionary if the file is missing or empty.
    """
    file_path = f"data/player_data/{player_id}_performance_details.csv"
    print(f"Looking for performance file: {file_path}")
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return {
            'teams': [],
            'total_games': 0,
            'total_goals': 0,
            'total_disposals': 0,
            'avg_disposals': 0.0,
            'total_brownlow_votes': 0,
            'games_20_plus_disposals': 0,
            'games_3_plus_goals': 0,
            'num_seasons': 0,
            'peak_disposals': 0,
            'peak_goals': 0,
            'median_disposals': 0,
            'impact_score': 0,
            'consistency_score': 0
        }
    try:
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            teams_order = []          # List to maintain chronological order of teams
            seen_teams = set()        # Set to track teams already encountered
            unique_years = set()      # Set to track unique seasons
            disposals_per_game = []   # List to track disposals per game
            goals_per_game = []       # List to track goals per game
            total_games = 0
            total_goals = 0
            total_disposals = 0
            total_brownlow_votes = 0
            games_20_plus_disposals = 0
            games_3_plus_goals = 0
            
            for row in reader:
                team = row['team']
                year = row['year']
                unique_years.add(year)
                if team not in seen_teams:
                    teams_order.append(team)
                    seen_teams.add(team)
                total_games += 1
                kicks = int(row['kicks']) if row['kicks'] else 0
                handballs = int(row['handballs']) if row['handballs'] else 0
                disposals = kicks + handballs
                disposals_per_game.append(disposals)
                total_disposals += disposals
                goals = int(row['goals']) if row['goals'] else 0
                goals_per_game.append(goals)
                total_goals += goals
                brownlow_votes = int(row['brownlow_votes']) if row['brownlow_votes'] else 0
                total_brownlow_votes += brownlow_votes
                if disposals >= 20:
                    games_20_plus_disposals += 1
                if goals >= 3:
                    games_3_plus_goals += 1
            
            num_seasons = len(unique_years)
            peak_disposals = max(disposals_per_game) if disposals_per_game else 0
            peak_goals = max(goals_per_game) if goals_per_game else 0
            median_disposals = median(disposals_per_game) if disposals_per_game else 0
            impact_score = (total_brownlow_votes / total_games) * 100 if total_games > 0 else 0
            consistency_score = (games_20_plus_disposals / total_games) * 100 if total_games > 0 else 0
            
            return {
                'teams': teams_order,
                'total_games': total_games,
                'total_goals': total_goals,
                'total_disposals': total_disposals,
                'avg_disposals': total_disposals / total_games if total_games > 0 else 0.0,
                'total_brownlow_votes': total_brownlow_votes,
                'games_20_plus_disposals': games_20_plus_disposals,
                'games_3_plus_goals': games_3_plus_goals,
                'num_seasons': num_seasons,
                'peak_disposals': peak_disposals,
                'peak_goals': peak_goals,
                'median_disposals': median_disposals,
                'impact_score': impact_score,
                'consistency_score': consistency_score
            }
    except StopIteration:
        print(f"File is empty: {file_path}")
        return {
            'teams': [],
            'total_games': 0,
            'total_goals': 0,
            'total_disposals': 0,
            'avg_disposals': 0.0,
            'total_brownlow_votes': 0,
            'games_20_plus_disposals': 0,
            'games_3_plus_goals': 0,
            'num_seasons': 0,
            'peak_disposals': 0,
            'peak_goals': 0,
            'median_disposals': 0,
            'impact_score': 0,
            'consistency_score': 0
        }

def main(input_csv, output_csv):
    """
    Process the input CSV containing player IDs, read player and performance data,
    and generate an output CSV with summaries for each player.
    """
    print(f"Current working directory: {os.getcwd()}")
    with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
        
        # Write the header to the output CSV
        writer.writerow(['Serial Number', 'Player Name', 'Footy Teams', 'Comment'])
        
        # Skip the header row in the input CSV
        next(reader)
        
        # Process each player
        for i, row in enumerate(reader, start=1):
            player_id = row[0].lower()  # Convert to lowercase to match file names if needed
            print(f"Processing player ID: {player_id}")
            
            # Read player data and performance data
            player_data = read_player_data(player_id)
            performance_data = read_performance_data(player_id)
            
            # Generate summary based on available data
            if player_data and performance_data['teams']:
                full_name = f"{player_data['first_name']} {player_data['last_name']}"
                teams_str = " - ".join(performance_data['teams'])
                summary = (
                    f"{full_name}, born on {player_data['born_date']}, "
                    f"debuted on {player_data['debut_date']}, "
                    f"height {player_data['height']} cm, weight {player_data['weight']} kg. "
                    f"A true legend of the game, {full_name} had an illustrious career spanning "
                    f"{performance_data['num_seasons']} seasons and {performance_data['total_games']} games. "
                    f"Playing for {teams_str}, he amassed an incredible {performance_data['total_disposals']} total disposals "
                    f"and {performance_data['total_goals']} goals. "
                    f"His impact on the game is reflected in his {performance_data['impact_score']:.1f} Impact Score "
                    f"and {performance_data['consistency_score']:.1f}% Consistency Score. "
                    f"Peak performances include {performance_data['peak_disposals']} disposals and "
                    f"{performance_data['peak_goals']} goals in a single game. "
                    f"His consistency is evident with {performance_data['games_20_plus_disposals']} games of 20+ disposals "
                    f"and {performance_data['games_3_plus_goals']} games with 3+ goals. "
                    f"A true champion, he earned {performance_data['total_brownlow_votes']} Brownlow votes "
                    f"throughout his career, cementing his status as one of the greats."
                )
            else:
                full_name = "Unknown"
                teams_str = "Unknown"
                summary = "No information available."
            
            # Write the row to the output CSV
            writer.writerow([i, full_name, teams_str, summary])

if __name__ == "__main__":
    main('data/top100/all_time_top_100.csv', 'all_time_top_100.csv')