import csv
import os
from statistics import mean, stdev

def extract_debut_year(debut_date):
    try:
        parts = debut_date.split('-')
        if len(parts) == 3:
            if len(parts[0]) == 4:  # YYYY-MM-DD
                return int(parts[0])
            elif len(parts[2]) == 4:  # DD-MM-YYYY
                return int(parts[2])
        return None
    except (ValueError, IndexError):
        return None

def categorize_era(debut_year):
    if debut_year is None:
        return 'Unknown'
    elif debut_year < 1970:
        return 'pre-1970'
    elif 1970 <= debut_year <= 1990:
        return '1970-1990'
    else:
        start_year = (debut_year - 1991) // 5 * 5 + 1991
        end_year = start_year + 4
        return f"{start_year}-{end_year}"

def calculate_era_stats(player_data_folder):
    metrics = ['kicks', 'marks', 'handballs', 'disposals', 'goals', 'behinds', 'hit_outs', 'tackles', 
               'rebound_50s', 'inside_50s', 'clearances', 'clangers', 'free_kicks_for', 'free_kicks_against', 
               'brownlow_votes', 'contested_possessions', 'uncontested_possessions', 'contested_marks', 
               'marks_inside_50', 'one_percenters', 'bounces', 'goal_assist', 'percentage_of_game_played']
    
    era_stats = {}
    era_counts = {}
    processed_players = 0

    for file in os.listdir(player_data_folder):
        if file.endswith('_personal_details.csv'):
            player_id = file.replace('_personal_details.csv', '')
            personal_file = os.path.join(player_data_folder, f"{player_id}_personal_details.csv")
            performance_file = os.path.join(player_data_folder, f"{player_id}_performance_details.csv")

            if os.path.exists(personal_file) and os.path.exists(performance_file):
                with open(personal_file, 'r') as f:
                    reader = csv.DictReader(f)
                    try:
                        personal_data = next(reader)
                        if 'debut_date' not in personal_data:
                            print(f"Missing debut_date for player: {player_id}")
                            continue
                        debut_year = extract_debut_year(personal_data['debut_date'])
                        if debut_year:
                            era = categorize_era(debut_year)
                            if era not in era_stats:
                                era_stats[era] = {metric: [] for metric in metrics}
                                era_counts[era] = 0
                            era_counts[era] += 1
                            processed_players += 1
                            print(f"Processing {player_id} with debut year {debut_year} in era {era}")

                            with open(performance_file, 'r') as perf_f:
                                perf_reader = csv.DictReader(perf_f)
                                if not perf_reader.fieldnames:
                                    print(f"Performance file empty or malformed: {performance_file}")
                                    continue
                                for row in perf_reader:
                                    for metric in metrics:
                                        value = int(row[metric]) if metric in row and row[metric] else 0
                                        era_stats[era][metric].append(value)
                    except StopIteration:
                        print(f"Personal file empty: {personal_file}")
                        continue

    print(f"Total players processed: {processed_players}")
    print(f"Players per era: {era_counts}")

    era_data = {}
    for era, stats in era_stats.items():
        era_data[era] = {}
        for metric, values in stats.items():
            if values:
                era_data[era][f"{metric}_mean"] = mean(values)
                era_data[era][f"{metric}_std"] = stdev(values) if len(values) > 1 else 0
            else:
                era_data[era][f"{metric}_mean"] = 0
                era_data[era][f"{metric}_std"] = 0

    return era_data

def write_era_stats_to_csv(era_data, output_file):
    metrics = ['kicks', 'marks', 'handballs', 'disposals', 'goals', 'behinds', 'hit_outs', 'tackles', 
               'rebound_50s', 'inside_50s', 'clearances', 'clangers', 'free_kicks_for', 'free_kicks_against', 
               'brownlow_votes', 'contested_possessions', 'uncontested_possessions', 'contested_marks', 
               'marks_inside_50', 'one_percenters', 'bounces', 'goal_assist', 'percentage_of_game_played']
    headers = ["Era"] + [f"{metric} Mean" for metric in metrics] + [f"{metric} Std" for metric in metrics]

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for era in sorted(era_data.keys()):
            stats = era_data[era]
            row = [era] + [stats.get(f"{metric}_mean", 0) for metric in metrics] + [stats.get(f"{metric}_std", 0) for metric in metrics]
            writer.writerow(row)

player_data_folder = "data/player_data"
era_data = calculate_era_stats(player_data_folder)
write_era_stats_to_csv(era_data, "data/era_stats.csv")

for era, stats in era_data.items():
    print(f"Era: {era}")
    for metric in ['kicks', 'marks', 'handballs', 'disposals', 'goals', 'behinds', 'hit_outs', 'tackles']:
        print(f"  {metric} Mean: {stats[f'{metric}_mean']:.2f}, Std: {stats[f'{metric}_std']:.2f}")