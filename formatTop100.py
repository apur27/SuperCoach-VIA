import csv
import os
from statistics import median

def read_player_data(player_id):
    """
    Read the player's personal data from their CSV file in data/player_data.
    Returns a dictionary with the player's info or None if the file is missing or empty.
    """
    file_path = f"data/player_data/{player_id}_personal_details.csv"
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            row = next(reader)
            required_columns = ['first_name', 'last_name', 'born_date', 'debut_date', 'height', 'weight']
            if not all(col in row for col in required_columns):
                return None
            return row
    except StopIteration:
        return None

def read_performance_data(player_id):
    """
    Read the player's performance data from their CSV file.
    Returns a dictionary with performance metrics, setting values to 0 if data is missing.
    """
    file_path = f"data/player_data/{player_id}_performance_details.csv"
    if not os.path.exists(file_path):
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
            'consistency_score': 0,
            'brownlow_available': False
        }
    try:
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames
            brownlow_available = 'brownlow_votes' in fieldnames
            teams_order = []
            seen_teams = set()
            unique_years = set()
            disposals_per_game = []
            goals_per_game = []
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
                if brownlow_available:
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
            
            # Calculate Impact Score (requires total_games > 0)
            if total_games > 0:
                if brownlow_available and total_brownlow_votes > 0:
                    impact_score = (total_brownlow_votes / total_games) * 100
                else:
                    impact_score = total_disposals / total_games if total_disposals > 0 else 0
                consistency_score = (games_20_plus_disposals / total_games) * 100 if games_20_plus_disposals > 0 else 0
            else:
                impact_score = 0
                consistency_score = 0
            
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
                'consistency_score': consistency_score,
                'brownlow_available': brownlow_available
            }
    except StopIteration:
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
            'consistency_score': 0,
            'brownlow_available': False
        }

def main(input_csv, output_csv):
    """
    Process the input CSV with player IDs and generate an output CSV with summaries.
    """
    with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
        
        # Write header
        writer.writerow(['Serial Number', 'Player Name', 'Footy Teams', 'Comment'])
        next(reader)  # Skip header row
        
        # Process each player
        for i, row in enumerate(reader, start=1):
            player_id = row[0].lower()
            player_data = read_player_data(player_id)
            performance_data = read_performance_data(player_id)
            
            # Generate summary
            if player_data:
                full_name = f"{player_data['first_name']} {player_data['last_name']}"
                base_summary = (
                    f"{full_name}, born on {player_data['born_date']}, "
                    f"debuted on {player_data['debut_date']}, "
                    f"height {player_data['height']} cm, weight {player_data['weight']} kg"
                )
            else:
                full_name = "Unknown"
                base_summary = "Unknown player"
            
            if performance_data['teams']:
                teams_str = " - ".join(performance_data['teams'])
                summary = f"{base_summary}. A true legend of the game, {full_name} played for {teams_str}"
                
                # Add seasons and games if non-zero
                if performance_data['num_seasons'] > 0 and performance_data['total_games'] > 0:
                    summary += f" over {performance_data['num_seasons']} seasons and {performance_data['total_games']} games"
                
                summary += "."
                
                # Add total disposals and goals if non-zero
                if performance_data['total_disposals'] > 0 or performance_data['total_goals'] > 0:
                    summary += " He recorded"
                    if performance_data['total_disposals'] > 0:
                        summary += f" {performance_data['total_disposals']} total disposals"
                    if performance_data['total_goals'] > 0:
                        summary += f" and {performance_data['total_goals']} goals" if performance_data['total_disposals'] > 0 else f" {performance_data['total_goals']} goals"
                    summary += "."
                
                # Add impact score if non-zero
                if performance_data['impact_score'] > 0:
                    if performance_data['brownlow_available'] and performance_data['total_brownlow_votes'] > 0:
                        summary += f" His impact is shown by a {performance_data['impact_score']:.1f} Impact Score (based on Brownlow votes)."
                    else:
                        summary += f" His impact is shown by a {performance_data['impact_score']:.1f} Impact Score (based on disposals)."
                
                # Add consistency score if non-zero
                if performance_data['consistency_score'] > 0:
                    summary += f" He maintained a {performance_data['consistency_score']:.1f}% Consistency Score."
                
                # Add peak performances if non-zero
                if performance_data['peak_disposals'] > 0 or performance_data['peak_goals'] > 0:
                    summary += " Peak performances include"
                    if performance_data['peak_disposals'] > 0:
                        summary += f" {performance_data['peak_disposals']} disposals"
                    if performance_data['peak_goals'] > 0:
                        summary += f" and {performance_data['peak_goals']} goals" if performance_data['peak_disposals'] > 0 else f" {performance_data['peak_goals']} goals"
                    summary += " in a single game."
                
                # Add consistency metrics if non-zero
                if performance_data['games_20_plus_disposals'] > 0 or performance_data['games_3_plus_goals'] > 0:
                    summary += " His consistency shines with"
                    if performance_data['games_20_plus_disposals'] > 0:
                        summary += f" {performance_data['games_20_plus_disposals']} games of 20+ disposals"
                    if performance_data['games_3_plus_goals'] > 0:
                        summary += f" and {performance_data['games_3_plus_goals']} games with 3+ goals" if performance_data['games_20_plus_disposals'] > 0 else f" {performance_data['games_3_plus_goals']} games with 3+ goals"
                    summary += "."
                
                # Add Brownlow votes if available and non-zero
                if performance_data['brownlow_available'] and performance_data['total_brownlow_votes'] > 0:
                    summary += f" He earned {performance_data['total_brownlow_votes']} Brownlow votes, cementing his greatness."
                else:
                    summary += " His legacy as a great is undeniable."
            else:
                teams_str = "Unknown"
                summary = f"{base_summary}. No performance data available."
            
            writer.writerow([i, full_name, teams_str, summary])

if __name__ == "__main__":
    main('data/top100/all_time_top_100.csv', 'all_time_top_100.csv')