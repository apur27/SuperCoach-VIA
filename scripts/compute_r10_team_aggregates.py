"""
Aggregate Richmond and Adelaide stats from the Round 10 2026 game (now in
each player's performance CSV). Produces the team-level totals needed for the
post-mortem: tackles, inside-50s, clearances, contested possessions,
clangers, hit-outs.

Also pulls each side's 2026-season aggregates (R1-R9 for Adelaide, R1-R9
for Richmond) for the "was this representative" question.
"""

import glob
import pandas as pd

DATE = "2026-05-10"
GAME_ROUND = 10
YEAR = 2026

ROSTERS = {
    "Richmond": [
        "balta_noah_23101999", "broad_nathan_15041993", "brown_tom_30072003",
        "burton_tom_09012007", "campbell_seth_29122004", "cumming_sam_27072007",
        "faull_jonty_01022006", "gray_campbell_01092003", "green_steely_09012004",
        "hayes-brown_oliver_28042000", "hopper_jacob_06021997", "lefau_mykelti_22061998",
        "lynch_tom_31101992", "mcauliffe_kane_01032005", "miller_ben_31081999",
        "retschko_patrick_28022006", "ross_jack_03092000", "short_jayden_24011996",
        "sonsie_tyler_27012003", "taranto_tim_28011998", "trainor_luke_10042006",
        "trezise_james_15062002", "vlastuin_nick_19041994",
    ],
    "Adelaide": [
        "berry_sam_12022002", "bond_hugh_25092004", "borlase_james_18072002",
        "cook_brayden_18072002", "cumming_isaac_11081998", "curtin_daniel_08032005",
        "dawson_jordan_09041997", "keays_ben_23021997", "laird_rory_29121993",
        "mcandrew_lachlan_26052000", "michalanney_max_26022004",
        "milera_wayne_14091997", "murray_nick_18122000", "murray_toby_03112003",
        "neal-bullen_alex_09011996", "peatling_james_21082000",
        "pedlar_luke_17052002", "rachele_josh_11042003", "rankine_izak_23042000",
        "soligo_jake_25012003", "taylor_zac_31012003", "thilthorpe_riley_07072002",
        "worrell_josh_11042001",
    ],
}


def sum_team_game(team_key, year, round_num):
    rows = []
    for slug in ROSTERS[team_key]:
        path = f"/home/abhi/git/SuperCoach-VIA/data/player_data/{slug}_performance_details.csv"
        df = pd.read_csv(path)
        # Force consistent types so '10' and 10 both match
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['round'] = pd.to_numeric(df['round'], errors='coerce')
        sub = df[(df['year'] == year) & (df['round'] == round_num)]
        if len(sub) == 1:
            rows.append(sub.iloc[0])
        elif len(sub) > 1:
            # Duplicate; take last
            rows.append(sub.iloc[-1])
    return pd.DataFrame(rows)


def main():
    metrics = ['kicks', 'handballs', 'disposals', 'marks', 'goals', 'behinds',
               'tackles', 'hit_outs', 'clangers', 'clearances', 'inside_50s',
               'rebound_50s', 'contested_possessions', 'uncontested_possessions',
               'contested_marks', 'marks_inside_50', 'one_percenters',
               'free_kicks_for', 'free_kicks_against']

    print("=== Round 10 (Richmond vs Adelaide, 2026-05-10) team totals ===")
    for team in ('Richmond', 'Adelaide'):
        sub = sum_team_game(team, YEAR, GAME_ROUND)
        row = {'team': team, 'players': len(sub)}
        for m in metrics:
            if m in sub.columns:
                row[m] = pd.to_numeric(sub[m], errors='coerce').sum()
        print(team)
        for k, v in row.items():
            print(f"  {k}: {v}")
        print()

    print("=== Richmond 2026 season-to-date (R1-R10) averages ===")
    rich_files = [f"/home/abhi/git/SuperCoach-VIA/data/player_data/{s}_performance_details.csv"
                  for s in ROSTERS['Richmond']]
    all_games = []
    for path in rich_files:
        df = pd.read_csv(path)
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['round'] = pd.to_numeric(df['round'], errors='coerce')
        sub = df[df['year'] == YEAR]
        all_games.append(sub)
    rich_all = pd.concat(all_games, ignore_index=True)
    # Group by round to get per-game team totals
    season_team = rich_all.groupby('round').sum(numeric_only=True).reset_index()
    print(f"Games this season: {len(season_team)} (rounds {sorted(season_team['round'].astype(int).tolist())})")
    print(f"Avg disposals/g: {season_team['disposals'].mean():.1f}")
    print(f"Avg kicks/g: {season_team['kicks'].mean():.1f}")
    print(f"Avg marks/g: {season_team['marks'].mean():.1f}")
    print(f"Avg tackles/g: {season_team['tackles'].mean():.1f}")
    print(f"Avg goals/g: {season_team['goals'].mean():.1f}")
    print(f"Avg clangers/g: {season_team['clangers'].mean():.1f}")
    print(f"Avg hit_outs/g: {season_team['hit_outs'].mean():.1f}")
    print(f"Avg inside_50s/g: {season_team['inside_50s'].mean():.1f}")
    print(f"Avg contested_possessions/g: {season_team['contested_possessions'].mean():.1f}")
    print("\nRichmond round-by-round disposals & tackles & inside_50s & clangers:")
    print(season_team[['round', 'disposals', 'kicks', 'tackles', 'inside_50s', 'clangers', 'goals']]
          .sort_values('round').to_string(index=False))

    print()
    print("=== Adelaide 2026 season-to-date (R1-R10) averages ===")
    ade_files = [f"/home/abhi/git/SuperCoach-VIA/data/player_data/{s}_performance_details.csv"
                 for s in ROSTERS['Adelaide']]
    all_games = []
    for path in ade_files:
        df = pd.read_csv(path)
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['round'] = pd.to_numeric(df['round'], errors='coerce')
        sub = df[df['year'] == YEAR]
        all_games.append(sub)
    ade_all = pd.concat(all_games, ignore_index=True)
    season_team_a = ade_all.groupby('round').sum(numeric_only=True).reset_index()
    print(f"Games this season: {len(season_team_a)}")
    print(f"Avg tackles/g: {season_team_a['tackles'].mean():.1f}")
    print(f"Avg contested_possessions/g: {season_team_a['contested_possessions'].mean():.1f}")
    print(f"Avg hit_outs/g: {season_team_a['hit_outs'].mean():.1f}")

    print("\n=== Sanity: Richmond losing margin per round 2026 ===")
    matches = pd.read_csv('/home/abhi/git/SuperCoach-VIA/data/matches/matches_2026.csv')
    rich_matches = matches[(matches['team_1_team_name'] == 'Richmond') | (matches['team_2_team_name'] == 'Richmond')].copy()

    def richmond_margin(row):
        if row['team_1_team_name'] == 'Richmond':
            rich_score = row['team_1_final_goals'] * 6 + row['team_1_final_behinds']
            opp_score = row['team_2_final_goals'] * 6 + row['team_2_final_behinds']
            opp = row['team_2_team_name']
        else:
            rich_score = row['team_2_final_goals'] * 6 + row['team_2_final_behinds']
            opp_score = row['team_1_final_goals'] * 6 + row['team_1_final_behinds']
            opp = row['team_1_team_name']
        return pd.Series({'opp': opp, 'rich': rich_score, 'opp_score': opp_score,
                          'margin': rich_score - opp_score})

    rich_matches[['opp', 'rich', 'opp_score', 'margin']] = rich_matches.apply(richmond_margin, axis=1)
    print(rich_matches[['round_num', 'opp', 'rich', 'opp_score', 'margin']].sort_values('round_num').to_string(index=False))
    print(f"\nRichmond avg margin 2026: {rich_matches['margin'].mean():.1f}")
    print(f"Richmond W/L: {(rich_matches['margin'] > 0).sum()}W - {(rich_matches['margin'] < 0).sum()}L")


if __name__ == "__main__":
    main()
