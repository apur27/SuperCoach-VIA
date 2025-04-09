import pandas as pd
import matplotlib.pyplot as plt
import os

# Check where files will be saved
print("Current working directory:", os.getcwd())

# Load the data
df = pd.read_csv('data/era_stats.csv')
print("Columns in the DataFrame:", df.columns.tolist())
print(df.head())

# Define the correct chronological order of eras
correct_order = ['pre-1970', '1970-1990', '1991-1995', '1996-2000', 
                 '2001-2005', '2006-2010', '2011-2015', '2016-2020', '2021-2025']

# Sort the eras
df['Era'] = pd.Categorical(df['Era'], categories=correct_order, ordered=True)
df = df.sort_values('Era')

# Find mean columns (adjusted for ' Mean' naming)
mean_columns = [col for col in df.columns if ' Mean' in col]
print("Statistics to plot:", mean_columns)

# Set plot style
plt.style.use('seaborn-v0_8-muted')

# Loop through each statistic
for mean_col in mean_columns:
    # Extract statistic name (e.g., 'kicks' from 'kicks Mean')
    stat_name = mean_col.replace(' Mean', '')
    # Corresponding std column (e.g., 'kicks Std')
    std_col = f"{stat_name} Std"
    error_values = df[std_col] if std_col in df.columns else None
    
    try:
        plt.figure(figsize=(12, 6))
        plt.bar(df['Era'], df[mean_col], yerr=error_values, capsize=5, 
                color='skyblue', edgecolor='black')
        plt.xlabel('Era', fontsize=14, fontweight='bold')
        plt.ylabel(f'Average {stat_name.capitalize()} per Player per Game', 
                   fontsize=14, fontweight='bold')
        plt.title(f'Evolution of {stat_name.capitalize()} Across Eras', 
                  fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        output_file = f"{stat_name}_bar_chart.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Chart for {stat_name} saved as {output_file}")
    except Exception as e:
        print(f"Error creating chart for {stat_name}: {e}")
    plt.close()

# Test saving a simple plot
plt.figure()
plt.plot([1, 2, 3], [4, 5, 6])
plt.savefig('test_plot.png')
print("Test plot saved as test_plot.png")