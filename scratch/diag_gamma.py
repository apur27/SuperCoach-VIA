"""Sensitivity check: how does GAMMA (year_score curve) reorder Carey vs Johnson/Lloyd?

Replays the all-time aggregation against the persisted yearly CSVs for a small
set of GAMMA values, prints the all-time score for the canary players. No
side effects.
"""
import os
import glob
import pandas as pd

ERA_COMPLETENESS = {
    'pre_1965':  0.85,
    '1965_1990': 0.92,
    '1990_2010': 0.95,
    'post_2010': 0.85,
}

def get_era(year):
    if 1897 <= year <= 1964: return 'pre_1965'
    if 1965 <= year <= 1990: return '1965_1990'
    if 1991 <= year <= 2010: return '1990_2010'
    if 2011 <= year <= 2030: return 'post_2010'
    return 'pre_1965'

CANARIES = [
    'carey_wayne_27051971',
    'lockett_tony_09031966',
    'johnson_brad_18071976',
    'lloyd_matthew_16041978',
    'matthews_leigh_01031952',
    'ablett_gary_01101961',
    'ablett_gary_14051984',
    'bartlett_kevin_06031947',
    'oliver_clayton_22071997',
    'pendlebury_scott_07011988',
    'dunstall_jason_14081964',
    'franklin_lance_30011987',
    'pavlich_matthew_31121981',
    'richardson_matthew_19031975',
]

# Career games — read from full data aggregate via yearly CSVs (not exact, but
# directional: yearly CSVs only count games-while-in-top-100, so we use the
# already-saved all_time_top_100.csv as a sanity reference and instead pull
# career_games from the actual files. To keep this diag self-contained we'll
# use a hardcoded reference set for the canaries.
CAREER_GAMES = {
    'carey_wayne_27051971': 272,
    'lockett_tony_09031966': 281,
    'johnson_brad_18071976': 364,
    'lloyd_matthew_16041978': 270,
    'matthews_leigh_01031952': 332,
    'ablett_gary_01101961': 248,
    'ablett_gary_14051984': 357,
    'bartlett_kevin_06031947': 403,
    'oliver_clayton_22071997': 175,   # approximate
    'pendlebury_scott_07011988': 400,
    'dunstall_jason_14081964': 269,
    'franklin_lance_30011987': 354,
    'pavlich_matthew_31121981': 353,
    'richardson_matthew_19031975': 282,
}

# Gather yearly ranks for each canary
ranks = {p: [] for p in CANARIES}
yearly_dir = '/home/abhi/git/SuperCoach-VIA/data/top100/yearly/'
for f in sorted(glob.glob(os.path.join(yearly_dir, 'year_*.csv'))):
    df = pd.read_csv(f)
    year = int(os.path.basename(f).replace('year_', '').replace('.csv', ''))
    era = get_era(year)
    for rank, row in enumerate(df.itertuples(index=False), start=1):
        if row.player in ranks:
            ranks[row.player].append((year, rank, era))

def all_time_score(rank_list, career_g, gamma):
    adj = []
    for year, rank, era in rank_list:
        ys = ((101 - rank) / 100.0) ** gamma
        adj.append(ys * ERA_COMPLETENESS[era])
    top8 = sorted(adj, reverse=True)[:8]
    if not top8:
        return 0.0
    mean_adj = sum(top8) / len(top8)
    cb = 0.30 * min(career_g / 300.0, 1.0)
    return mean_adj * (1.0 + cb)

print(f"{'Player':<35} " + "  ".join(f"γ={g:.1f}" for g in [1.0, 1.5, 2.0, 2.5, 3.0]))
for p in CANARIES:
    row = f"{p:<35} "
    for gamma in [1.0, 1.5, 2.0, 2.5, 3.0]:
        s = all_time_score(ranks[p], CAREER_GAMES.get(p, 200), gamma)
        row += f" {s:.4f}"
    print(row)
