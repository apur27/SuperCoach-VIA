import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- Data Loading and Preparation ---
data_dir = "./data/player_data/"
performance_files = [f for f in os.listdir(data_dir) if f.endswith("_performance_details.csv")]

dfs = []
for file in performance_files:
    file_path = os.path.join(data_dir, file)
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
        if 'team' not in df.columns:
            team_name = file.split('_performance_details.csv')[0]
            df['team'] = team_name
        dfs.append(df)
    except Exception as e:
        print(f"Error reading or processing {file_path}: {e}")

if not dfs:
    raise ValueError("No player performance data files found or processed.")

master_df = pd.concat(dfs, ignore_index=True)

# Extract year from date
master_df['year'] = master_df['date'].dt.year

# Filter for games since January 1, 2021 (inclusive)
start_year = 2021
master_df = master_df[master_df['date'] >= datetime(start_year, 1, 1)]

# Define key performance metrics to analyze
selected_metrics = ['disposals', 'goals', 'tackles', 'marks', 'hit_outs', 'clearances']

# Ensure all selected metrics are present and numeric globally first
for metric in selected_metrics:
    if metric not in master_df.columns:
        print(f"Warning: Metric '{metric}' not found in the data. Skipping globally.")
    else:
        master_df[metric] = pd.to_numeric(master_df[metric], errors='coerce').fillna(0)

# Remove metrics not found globally
selected_metrics = [m for m in selected_metrics if m in master_df.columns]
if not selected_metrics:
     raise ValueError("None of the selected metrics were found in the data.")

# --- Loop Through Each Year and Generate Heatmap ---
# Determine the years present in the filtered data
available_years = sorted(master_df['year'].unique())

for year in available_years:
    print(f"Processing year: {year}...")

    # Filter data for the current year
    year_df = master_df[master_df['year'] == year].copy() # Use .copy() to avoid SettingWithCopyWarning

    if year_df.empty:
        print(f"No data found for year {year}. Skipping.")
        continue

    # Check if 'team' column is valid for this year
    if 'team' not in year_df.columns or year_df['team'].isnull().all():
        print(f"Warning: The 'team' column is missing or empty for year {year}. Skipping heatmap generation for this year.")
        continue

    # Ensure required columns are not all NaN before grouping
    year_df.dropna(subset=['team', 'date'] + selected_metrics, how='all', inplace=True)
    if year_df.empty:
        print(f"No valid rows after dropping NaNs for year {year}. Skipping.")
        continue

    # Aggregate performance metrics by team and game for the specific year
    team_game_df_year = year_df.groupby(['team', 'date'])[selected_metrics].sum().reset_index()

    # Calculate the average per game for each team for the specific year
    team_avg_df_year = team_game_df_year.groupby('team')[selected_metrics].mean().reset_index()

    if team_avg_df_year.empty:
        print(f"No team averages calculated for year {year}. Skipping heatmap.")
        continue

    # Standardize the averages to z-scores *within the current year*
    z_score_cols = []
    for metric in selected_metrics:
        # Calculate mean and std dev based on this year's averages
        mean_val = team_avg_df_year[metric].mean()
        std_val = team_avg_df_year[metric].std()
        z_col_name = f'{metric}_z'
        z_score_cols.append(z_col_name)

        if std_val == 0 or pd.isna(std_val):
            team_avg_df_year[z_col_name] = 0 # Assign 0 if no variation or NaN std dev
        else:
            team_avg_df_year[z_col_name] = (team_avg_df_year[metric] - mean_val) / std_val

    # Prepare data for heatmap
    heatmap_data = team_avg_df_year.set_index('team')[z_score_cols]
    # Optional: Sort teams
    # heatmap_data = heatmap_data.sort_index()

    # Create the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=.5,
        cbar_kws={'label': f'Z-score ({year} Standardized Average per Game)'}
    )

    # Customize the plot
    plt.title(f'AFL Team Performance Comparison ({year})', fontsize=14)
    plt.xlabel('Performance Metrics (Standardized)', fontsize=10)
    plt.ylabel('Team', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the plot to a file with the year in the name
    output_filename = f'team_performance_heatmap_{year}.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Heatmap saved to {output_filename}")

    # Close the plot figure to free memory
    plt.close()

print("Finished processing all years.")
