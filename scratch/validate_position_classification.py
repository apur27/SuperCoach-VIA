"""
Validate proposed 3-group career-based position classification.
LOW/MEDIUM blast radius: diagnostic — informs a methodology decision but
results here will be re-validated by full pipeline before any production use.
"""

import os
import glob
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

DATA_DIR = config.PLAYER_DATA_DIR

WEIGHTS = {
    'goals': 55.0, 'behinds': 1.5,
    'kicks': 4.5, 'handballs': 3.0, 'marks': 2.5,
    'goal_assist': 4.0, 'contested_marks': 7.0,
    'contested_possessions': 5.5, 'tackles': 3.5,
    'one_percenters': 3.0, 'clearances': 5.5,
}


def era_stats(year: int):
    if year < 1965:
        return ['goals', 'behinds']
    if year <= 1990:
        return ['goals', 'behinds', 'kicks', 'handballs']
    if year <= 2010:
        return ['goals', 'behinds', 'kicks', 'handballs', 'marks']
    return ['goals', 'behinds', 'kicks', 'handballs', 'marks',
            'contested_possessions', 'tackles', 'clearances',
            'contested_marks', 'goal_assist']


def parse_player_name(filename: str) -> str:
    base = os.path.basename(filename).replace('_performance_details.csv', '')
    parts = base.rsplit('_', 1)  # split off DDMMYYYY date suffix
    name_part = parts[0]
    date_token = parts[1] if len(parts) == 2 else ''
    surname_first = name_part.split('_', 1)
    if len(surname_first) == 2:
        surname, given = surname_first
        return f"{given.title()} {surname.title()}", date_token
    return name_part.title(), date_token


def load_career(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception:
        return pd.DataFrame()
    return df


def career_summary(df: pd.DataFrame) -> dict:
    if df.empty or 'year' not in df.columns:
        return {}
    games = len(df)
    if games == 0:
        return {}
    out = {'games': games, 'year_min': int(df['year'].min()),
           'year_max': int(df['year'].max())}
    for col in ['goals', 'kicks', 'handballs', 'marks', 'tackles', 'clearances',
                'contested_possessions', 'contested_marks', 'goal_assist',
                'behinds', 'one_percenters']:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).sum()
        else:
            out[col] = 0.0
    return out


def classify(gpg: float) -> str:
    if gpg >= 3.0:
        return 'key_forward'
    if gpg >= 0.80:
        return 'forward_mid'
    return 'other'


def main():
    np.random.seed(42)
    perf_files = sorted(glob.glob(os.path.join(DATA_DIR, '*_performance_details.csv')))
    print(f"[INSPECT] Found {len(perf_files):,} performance files")

    rows = []
    careers = {}
    for fp in perf_files:
        df = load_career(fp)
        if df.empty:
            continue
        summ = career_summary(df)
        if not summ:
            continue
        name, token = parse_player_name(fp)
        unique = name
        if unique in careers:
            unique = f"{name} [{token}]"
        careers[unique] = (df, summ, fp)
        rows.append({
            'player': unique,
            'games': summ['games'],
            'goals': summ['goals'],
            'g_per_game': summ['goals'] / summ['games'],
            'year_min': summ['year_min'],
            'year_max': summ['year_max'],
        })

    career_df = pd.DataFrame(rows)
    print(f"[INSPECT] Loaded {len(career_df):,} careers (>=1 game)")
    print(f"[INSPECT] >=50 games: {len(career_df[career_df.games>=50]):,}")
    print(f"[INSPECT] >=100 games: {len(career_df[career_df.games>=100]):,}")
    print(f"[INSPECT] year range: {int(career_df.year_min.min())}-{int(career_df.year_max.max())}\n")

    cdf = career_df.set_index('player')

    # ============ SECTION 1: Top 50 by career g/game ============
    print("=" * 90)
    print("SECTION 1 — TOP 50 by CAREER GOALS/GAME (min 100 games)")
    print("=" * 90)
    top50 = career_df[career_df.games >= 100].sort_values('g_per_game', ascending=False).head(50)
    print(top50.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ============ SECTION 2: Key player classification ============
    print("\n" + "=" * 90)
    print("SECTION 2 — KEY PLAYER CLASSIFICATION (KF >=3.0, FM 0.80-2.99, OTHER <0.80)")
    print("=" * 90)
    key_players = [
        'Tony Lockett', 'Jason Dunstall', 'Matthew Lloyd', 'Lance Franklin',
        'Wayne Carey', 'Leigh Matthews', 'Stewart Loewe', 'Barry Hall',
        'Patrick Dangerfield', 'Kevin Bartlett', 'Brad Johnson', 'Warren Tredrea',
        'Bob Skilton', 'Scott Pendlebury', 'Luke Parker', 'Lachie Neale',
        'Gary Ablett',
    ]
    for kp in key_players:
        matches = [p for p in cdf.index if p == kp or p.startswith(kp + ' [')]
        for m in matches:
            row = cdf.loc[m]
            gpg = row['g_per_game']
            grp = classify(gpg)
            print(f"  {m:<40} games={int(row['games']):4d}  goals={int(row['goals']):4d}  "
                  f"g/g={gpg:5.3f}  -> {grp:<12}  ({int(row['year_min'])}-{int(row['year_max'])})")

    # ============ SECTION 3: Distribution ============
    print("\n" + "=" * 90)
    print("SECTION 3 — CAREER g/game DISTRIBUTION (>=100 games)")
    print("=" * 90)
    eligible = career_df[career_df.games >= 100]
    n = len(eligible)
    n_kf = (eligible.g_per_game >= 3.0).sum()
    n_fm = ((eligible.g_per_game >= 0.80) & (eligible.g_per_game < 3.0)).sum()
    n_ot = (eligible.g_per_game < 0.80).sum()
    print(f"  N = {n:,}")
    print(f"  key_forward (>=3.00):       {n_kf:5d}  ({100*n_kf/n:5.2f}%)")
    print(f"  forward_mid (0.80-2.99):    {n_fm:5d}  ({100*n_fm/n:5.2f}%)")
    print(f"  other       (<0.80):        {n_ot:5d}  ({100*n_ot/n:5.2f}%)")
    quants = eligible.g_per_game.quantile([0.5, 0.75, 0.85, 0.90, 0.95, 0.97, 0.99, 0.995])
    print("  Quantiles of g/game:")
    for q, v in quants.items():
        print(f"    p{int(q*1000)/10:5.1f}  {v:.3f}")

    # ============ SECTION 4: Within-group z for 1996 and 1995 ============
    for YEAR in (1996, 1995):
        print("\n" + "=" * 90)
        print(f"SECTION 4 — WITHIN-GROUP Z-SCORE FOR {YEAR}")
        print("=" * 90)
        stats_y = era_stats(YEAR)
        print(f"  Stats included: {stats_y}")
        ydat = []
        for name, (df, summ, _) in careers.items():
            sub = df[df['year'] == YEAR]
            if len(sub) < 5:
                continue
            score = 0.0
            for s in stats_y:
                if s in sub.columns:
                    v = pd.to_numeric(sub[s], errors='coerce').fillna(0).sum()
                    score += v * WEIGHTS[s]
            gpg = summ['goals'] / summ['games']
            ydat.append({'player': name, 'games_yr': len(sub), 'raw': score,
                         'career_gpg': gpg, 'group': classify(gpg)})
        yr_df = pd.DataFrame(ydat)
        print(f"  N players (>=5 games): {len(yr_df)}")
        yr_df['z_within'] = 0.0
        for g, gdf in yr_df.groupby('group'):
            mu = gdf['raw'].mean()
            sd = gdf['raw'].std(ddof=0)
            yr_df.loc[gdf.index, 'z_within'] = (gdf['raw'] - mu) / sd
            print(f"    {g:<12} N={len(gdf):4d} mean_raw={mu:7.1f} std_raw={sd:6.1f}")
        mu_all = yr_df['raw'].mean()
        sd_all = yr_df['raw'].std(ddof=0)
        yr_df['z_full'] = (yr_df['raw'] - mu_all) / sd_all
        print(f"    full_cohort  N={len(yr_df):4d} mean_raw={mu_all:7.1f} std_raw={sd_all:6.1f}")

        for g in ['key_forward', 'forward_mid', 'other']:
            sub = yr_df[yr_df.group == g].sort_values('z_within', ascending=False).head(10)
            print(f"\n  TOP 10 in group '{g}' ({YEAR}):")
            print(sub[['player', 'games_yr', 'raw', 'z_within', 'z_full']].to_string(
                index=False, float_format=lambda x: f"{x:.3f}"))

        focus = ['Wayne Carey', 'Tony Lockett', 'Leigh Matthews', 'Kevin Bartlett',
                 'Stewart Loewe', 'Barry Hall', 'Gary Ablett', 'Patrick Dangerfield',
                 'Scott Pendlebury', 'Brad Johnson', 'Warren Tredrea',
                 'Jason Dunstall', 'Matthew Lloyd']
        print(f"\n  KEY PLAYERS in {YEAR}:")
        for kp in focus:
            for nm in yr_df['player']:
                if nm == kp or nm.startswith(kp + ' ['):
                    row = yr_df[yr_df.player == nm].iloc[0]
                    print(f"    {nm:<40} grp={row['group']:<12} raw={row['raw']:7.1f}  "
                          f"z_within={row['z_within']:+.3f}  z_full(current)={row['z_full']:+.3f}")

    # ============ SECTION 5: All-seasons mean_z_top8 within-group ============
    print("\n" + "=" * 90)
    print("SECTION 5 — mean_z_top8 (WITHIN-GROUP) FOR KEY PLAYERS — ALL SEASONS")
    print("=" * 90)
    print("  (estimate: simulates within-group z for every season >=5 games, averages top 8)")
    print("  NOTE: no era shrinkage / no single-stat cap applied. Pure z aggregation.\n")

    target = ['Tony Lockett', 'Jason Dunstall', 'Wayne Carey', 'Leigh Matthews',
              'Stewart Loewe', 'Barry Hall', 'Patrick Dangerfield', 'Scott Pendlebury',
              'Kevin Bartlett', 'Gary Ablett', 'Matthew Lloyd', 'Lance Franklin',
              'Warren Tredrea', 'Brad Johnson', 'Lachie Neale', 'Luke Parker',
              'Bob Skilton']
    target_groups = {}
    for kp in target:
        matches = [p for p in cdf.index if p == kp or p.startswith(kp + ' [')]
        for m in matches:
            target_groups[m] = (classify(cdf.loc[m, 'g_per_game']),
                                float(cdf.loc[m, 'g_per_game']))

    player_group = {nm: classify(s['goals']/s['games'])
                    for nm, (_, s, _) in careers.items()}

    target_z = {tp: [] for tp in target_groups}
    by_year = {}
    for name, (df, _, _) in careers.items():
        if 'year' not in df.columns:
            continue
        for yr, sub in df.groupby('year'):
            try:
                yr_int = int(yr)
            except Exception:
                continue
            if len(sub) < 5:
                continue
            stats_y = era_stats(yr_int)
            score = 0.0
            for s in stats_y:
                if s in sub.columns:
                    v = pd.to_numeric(sub[s], errors='coerce').fillna(0).sum()
                    score += v * WEIGHTS[s]
            by_year.setdefault(yr_int, []).append((name, score, player_group[name]))

    for yr_int, plist in by_year.items():
        dfy = pd.DataFrame(plist, columns=['player', 'raw', 'group'])
        dfy['z'] = 0.0
        for g, gdf in dfy.groupby('group'):
            mu = gdf['raw'].mean()
            sd = gdf['raw'].std(ddof=0)
            if sd > 0:
                dfy.loc[gdf.index, 'z'] = (gdf['raw'] - mu) / sd
        for tp in target_groups:
            row = dfy[dfy.player == tp]
            if len(row):
                target_z[tp].append((yr_int, float(row.iloc[0]['z'])))

    summary = []
    for tp, (grp, gpg) in target_groups.items():
        zs_sorted = sorted([z for _, z in target_z[tp]], reverse=True)
        top8 = zs_sorted[:8]
        top3 = zs_sorted[:3]
        m8 = float(np.mean(top8)) if top8 else float('nan')
        m3 = float(np.mean(top3)) if top3 else float('nan')
        best3 = sorted(target_z[tp], key=lambda x: x[1], reverse=True)[:3]
        best3_str = ", ".join([f"{y}:{z:+.2f}" for y, z in best3])
        summary.append((tp, grp, gpg, len(zs_sorted), m8, m3, best3_str))

    summary.sort(key=lambda r: r[4], reverse=True)
    print(f"  {'#':>2}  {'Player':<32} {'Group':<12} {'g/g':>5}  {'#szn':>4} "
          f"{'top8_meanZ':>11} {'top3_meanZ':>11}  best 3 seasons")
    print("  " + "-" * 130)
    for i, r in enumerate(summary, 1):
        print(f"  {i:2d}  {r[0]:<32} {r[1]:<12} {r[2]:5.2f}  {r[3]:4d}  "
              f"{r[4]:+11.3f} {r[5]:+11.3f}  {r[6]}")

    # ============ SECTION 6: Comparison with current full-cohort z ============
    print("\n" + "=" * 90)
    print("SECTION 6 — COMPARISON: WITHIN-GROUP vs FULL-COHORT mean_z_top8")
    print("=" * 90)
    target_z_full = {tp: [] for tp in target_groups}
    for yr_int, plist in by_year.items():
        dfy = pd.DataFrame(plist, columns=['player', 'raw', 'group'])
        mu = dfy['raw'].mean()
        sd = dfy['raw'].std(ddof=0)
        if sd > 0:
            dfy['z_full'] = (dfy['raw'] - mu) / sd
        else:
            dfy['z_full'] = 0.0
        for tp in target_groups:
            row = dfy[dfy.player == tp]
            if len(row):
                target_z_full[tp].append(float(row.iloc[0]['z_full']))

    print(f"  {'Player':<32} {'Group':<12} {'within_top8':>12} {'full_top8':>11} {'delta':>8}")
    print("  " + "-" * 80)
    rows = []
    for tp, (grp, _) in target_groups.items():
        within = [z for _, z in target_z[tp]]
        full = target_z_full[tp]
        within.sort(reverse=True)
        full.sort(reverse=True)
        m8w = float(np.mean(within[:8])) if within else float('nan')
        m8f = float(np.mean(full[:8])) if full else float('nan')
        rows.append((tp, grp, m8w, m8f, m8w - m8f))
    rows.sort(key=lambda r: r[2], reverse=True)
    for tp, grp, m8w, m8f, dlt in rows:
        print(f"  {tp:<32} {grp:<12} {m8w:+12.3f} {m8f:+11.3f} {dlt:+8.3f}")


if __name__ == '__main__':
    main()
