import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('data/era_stats.csv')

# Define the correct chronological order of eras
correct_order = ['pre-1970', '1970-1990', '1991-1995', '1996-2000', '2001-2005', '2006-2010', '2011-2015', '2016-2020', '2021-2025']
df['Era'] = pd.Categorical(df['Era'], categories=correct_order, ordered=True)
df = df.sort_values('Era')

# Define columns to plot (both raw and normalized)
stats_to_plot = ['kicks Mean', 'disposals Mean', 'goals Mean', 'tackles Mean', 'inside_50s Mean']
norm_stats_to_plot = ['kicks Norm Mean', 'disposals Norm Mean', 'goals Norm Mean', 'tackles Norm Mean', 'inside_50s Norm Mean']

# Map column names to readable labels
label_mapping = {
    'kicks Mean': 'Kicks (Raw)', 'kicks Norm Mean': 'Kicks (Norm)',
    'disposals Mean': 'Disposals (Raw)', 'disposals Norm Mean': 'Disposals (Norm)',
    'goals Mean': 'Goals (Raw)', 'goals Norm Mean': 'Goals (Norm)',
    'tackles Mean': 'Tackles (Raw)', 'tackles Norm Mean': 'Tackles (Norm)',
    'inside_50s Mean': 'Inside 50s (Raw)', 'inside_50s Norm Mean': 'Inside 50s (Norm)'
}

# Validate columns
all_stats = stats_to_plot + norm_stats_to_plot
for stat in all_stats:
    if stat not in df.columns:
        raise ValueError(f"Column '{stat}' not found in CSV. Available columns: {df.columns.tolist()}")

# Extract Eras for the X-axis
eras = df['Era']

# Set plot style and create subplots
plt.style.use('seaborn-v0_8-muted')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

# Define distinct markers and line styles
markers = ['o', 's', 'D', '^', 'v']
line_styles = ['-', '--']

# Plot raw stats on the first subplot
for stat, marker in zip(stats_to_plot, markers):
    ax1.plot(eras, df[stat], marker=marker, linewidth=2.5, markersize=8, linestyle=line_styles[0], label=label_mapping[stat])

# Plot normalized stats on the second subplot
for stat, marker in zip(norm_stats_to_plot, markers):
    ax2.plot(eras, df[stat], marker=marker, linewidth=2.5, markersize=8, linestyle=line_styles[1], label=label_mapping[stat])

# Customize the subplots
ax1.set_title('Raw Average per Player per Game', fontsize=16, fontweight='bold')
ax1.set_ylabel('Raw Average', fontsize=14, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(title='Statistic', fontsize=12, title_fontsize='13')

ax2.set_title('Normalized Average per Player per 100 Minutes', fontsize=16, fontweight='bold')
ax2.set_xlabel('Era', fontsize=14, fontweight='bold')
ax2.set_ylabel('Normalized Average', fontsize=14, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(title='Statistic', fontsize=12, title_fontsize='13')

# Improve readability of x-axis
plt.xticks(rotation=45, fontsize=12)
ax1.tick_params(axis='y', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)

# Adjust layout and save
plt.tight_layout()
output_file = 'afl_stats_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Chart successfully saved as {output_file}")

# Close the plot
plt.close()