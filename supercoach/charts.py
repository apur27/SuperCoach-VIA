import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    df = pd.read_csv('data/era_stats.csv')

    stats_to_plot = ['kicks', 'disposals', 'goals', 'tackles', 'inside_50s']
    correct_order = ['pre-1965', '1965-1990', '1991-2010', '2011-present']

    label_mapping = {
        'kicks': 'Kicks',
        'disposals': 'Disposals',
        'goals': 'Goals',
        'tackles': 'Tackles',
        'inside_50s': 'Inside 50s',
    }

    # Filter to the 5 stats, enforce era order, sort
    df_f = df[df['metric'].isin(stats_to_plot)].copy()
    df_f['era'] = pd.Categorical(df_f['era'], categories=correct_order, ordered=True)
    df_f = df_f.sort_values('era')

    # Pivot long → wide: one row per era, one column per metric
    raw_wide = df_f.pivot(index='era', columns='metric', values='mean_per_game')[stats_to_plot]
    norm_wide = df_f.pivot(index='era', columns='metric', values='mean_per_100pct_played')[stats_to_plot]

    # NaNs (e.g. pre-1965 kicks/tackles/inside_50s not recorded) render as line gaps — intentional
    eras = raw_wide.index.astype(str)

    plt.style.use('seaborn-v0_8-muted')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    markers = ['o', 's', 'D', '^', 'v']

    for stat, marker in zip(stats_to_plot, markers):
        ax1.plot(eras, raw_wide[stat], marker=marker, linewidth=2.5, markersize=8,
                 linestyle='-', label=label_mapping[stat])

    for stat, marker in zip(stats_to_plot, markers):
        ax2.plot(eras, norm_wide[stat], marker=marker, linewidth=2.5, markersize=8,
                 linestyle='--', label=label_mapping[stat])

    ax1.set_title('Raw Average per Player per Game', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Raw Average', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(title='Statistic', fontsize=12, title_fontsize='13')

    ax2.set_title('Normalized Average per Player per 100% Game Played', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Era', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average per 100% Game Played', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(title='Statistic', fontsize=12, title_fontsize='13')

    plt.xticks(rotation=45, fontsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)

    plt.tight_layout()
    output_file = 'afl_stats_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Chart successfully saved as {output_file}")
    plt.close()


if __name__ == "__main__":
    main()
