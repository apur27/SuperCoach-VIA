import csv
import os

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
    Returns a dictionary with teams in chronological order of first appearance, total games, total goals,
    total disposals, average disposals per game, total Brownlow votes, games with 20+ disposals,
    and games with 3+ goals; returns a default dictionary if the file is missing or empty.
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
            'games_3_plus_goals': 0
        }
    try:
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            teams_order = []          # List to maintain chronological order of teams
            seen_teams = set()        # Set to track teams already encountered
            total_games = 0
            total_goals = 0
            total_disposals = 0
            total_brownlow_votes = 0
            games_20_plus_disposals = 0
            games_3_plus_goals = 0
            for row in reader:
                team = row['team']
                if team not in seen_teams:
                    teams_order.append(team)
                    seen_teams.add(team)
                total_games += 1
                # Calculate disposals from kicks and handballs
                kicks = int(row['kicks']) if row['kicks'] else 0
                handballs = int(row['handballs']) if row['handballs'] else 0
                disposals = kicks + handballs
                total_disposals += disposals
                # Goals and Brownlow votes
                goals = int(row['goals']) if row['goals'] else 0
                total_goals += goals
                brownlow_votes = int(row['brownlow_votes']) if row['brownlow_votes'] else 0
                total_brownlow_votes += brownlow_votes
                # Check for high-performance games
                if disposals >= 20:
                    games_20_plus_disposals += 1
                if goals >= 3:
                    games_3_plus_goals += 1
            # Calculate average disposals per game
            avg_disposals = total_disposals / total_games if total_games > 0 else 0.0
            return {
                'teams': teams_order,
                'total_games': total_games,
                'total_goals': total_goals,
                'total_disposals': total_disposals,
                'avg_disposals': avg_disposals,
                'total_brownlow_votes': total_brownlow_votes,
                'games_20_plus_disposals': games_20_plus_disposals,
                'games_3_plus_goals': games_3_plus_goals
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
            'games_3_plus_goals': 0
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
                summary = (f"{full_name}, born on {player_data['born_date']}, "
                           f"debuted on {player_data['debut_date']}, "
                           f"height {player_data['height']} cm, weight {player_data['weight']} kg. "
                           f"Played for {teams_str} across {performance_data['total_games']} games, "
                           f"scoring {performance_data['total_goals']} goals. "
                           f"Career highlights include {performance_data['total_disposals']} total disposals, "
                           f"averaging {performance_data['avg_disposals']:.1f} per game, "
                           f"{performance_data['total_brownlow_votes']} Brownlow votes, "
                           f"{performance_data['games_20_plus_disposals']} games with 20+ disposals, "
                           f"and {performance_data['games_3_plus_goals']} games with 3+ goals.")
            else:
                full_name = "Unknown"
                teams_str = "Unknown"
                summary = "No information available."
            
            # Write the row to the output CSV
            writer.writerow([i, full_name, teams_str, summary])

if __name__ == "__main__":
    main('data/top100/all_time_top_100.csv', 'all_time_top_100.csv')