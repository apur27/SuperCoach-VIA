"""Diagnostic: for each player of interest, list yearly ranks & era-adjusted year_scores.

Reads the persisted yearly CSVs in data/top100/yearly/, derives rank from row order
(yearly CSVs are written sorted by raw score desc — same order used by the all-time
aggregator), applies ERA_COMPLETENESS, and prints what feeds the all-time formula.

This is read-only — no side effects, no model fit, no decisions ride on the output.
"""
import os
import glob
import pandas as pd

ERA_COMPLETENESS = {
    'pre_1965':  0.65,
    '1965_1990': 0.78,
    '1990_2010': 0.90,
    'post_2010': 0.80,
    'unknown':   0.65,
}

def get_era(year):
    if 1897 <= year <= 1964: return 'pre_1965'
    if 1965 <= year <= 1990: return '1965_1990'
    if 1991 <= year <= 2010: return '1990_2010'
    if 2011 <= year <= 2030: return 'post_2010'
    return 'unknown'

PLAYERS = [
    'carey_wayne_27051971',
    'lockett_tony_09031966',
    'johnson_brad_18071976',
    'lloyd_matthew_16041978',
    'matthews_leigh_01031952',
    'ablett_gary_01101961',     # Ablett Sr
    'ablett_gary_14051984',     # Ablett Jr
    'bartlett_kevin_06031947',
    'oliver_clayton_22071997',
    'pendlebury_scott_07011988',
    'macrae_jack_03081994',
    'dunstall_jason_14081964',
    'franklin_lance_30011987',
    'voss_michael_07071975',
    'hird_james_04021973',
]

ranks_by_player = {p: [] for p in PLAYERS}

yearly_dir = '/home/abhi/git/SuperCoach-VIA/data/top100/yearly/'
files = sorted(glob.glob(os.path.join(yearly_dir, 'year_*.csv')))
total_rows_in = 0
total_rows_out = 0
for f in files:
    df = pd.read_csv(f)
    total_rows_in += len(df)
    year = int(os.path.basename(f).replace('year_', '').replace('.csv', ''))
    era = get_era(year)
    ec = ERA_COMPLETENESS[era]
    for rank, row in enumerate(df.itertuples(index=False), start=1):
        p = row.player
        if p in ranks_by_player:
            year_score = (100 - rank + 1) / 100.0
            adj = year_score * ec
            ranks_by_player[p].append((year, rank, era, year_score, adj))
            total_rows_out += 1

print(f"Yearly files scanned: {len(files)}; total rows in: {total_rows_in}; player-rows captured: {total_rows_out}")
print()

for p in PLAYERS:
    rows = ranks_by_player[p]
    if not rows:
        print(f"{p}: NO TOP-100 APPEARANCES")
        continue
    seasons = len(rows)
    rows_sorted = sorted(rows, key=lambda x: -x[3])  # by year_score desc
    top8 = rows_sorted[:8]
    mean_adj = sum(r[4] for r in top8) / len(top8)
    best_rank = min(r[1] for r in rows)
    best_year = next(r[0] for r in rows if r[1] == best_rank)
    n_top10 = sum(1 for r in rows if r[1] <= 10)
    n_top20 = sum(1 for r in rows if r[1] <= 20)
    print(f"{p}")
    print(f"  seasons in top 100: {seasons}; best_rank={best_rank} (yr {best_year}); "
          f"top10_count={n_top10}; top20_count={n_top20}")
    print(f"  mean_adj_top8 = {mean_adj:.4f}")
    print(f"  top-8 (year, rank, era, year_score, adj): " + ", ".join(f"({r[0]},#{r[1]},{r[3]:.2f}*ec={r[4]:.3f})" for r in top8))
    print()
