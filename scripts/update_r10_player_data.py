"""
Append Round 10 (Richmond vs Adelaide, 2026-05-10) entries to each player's
performance CSV.

Stats come from afltables.com (authoritative). The live snapshot's
goals/behinds/clangers columns are unreliable for this game — verified by a
cross-check that found 102 field mismatches vs afltables, concentrated in
those three fields.

Run with: /home/abhi/sourceCode/python/coding/.venv/bin/python scripts/update_r10_player_data.py
"""

import os
import pandas as pd

PLAYER_DIR = "/home/abhi/git/SuperCoach-VIA/data/player_data"
DATE = "2026-05-10"
YEAR = 2026
ROUND = 10
OPP = {"Richmond": "Adelaide", "Adelaide": "Richmond"}
RESULT = {"Richmond": "L", "Adelaide": "W"}

# Tuple schema:
# (first, surname, team, jumper,
#  kicks, marks, handballs, disposals,
#  goals, behinds, hit_outs, tackles,
#  ff, fa, cp, up, cm, mi, op_1pct, bo, ga, pct,
#  i50, r50, cl_clearances, clangers, br, fname)
DATA = [
    # Richmond
    ("Noah",        "Balta",        "Richmond", 21, 11, 6,  3,  14, 1, 1, 8,  1, 2, 0, 7,  8,  3, 0, 1,  0, 0, 83, 3, 1, 3, 1, 0, "balta_noah_23101999_performance_details.csv"),
    ("Nathan",      "Broad",        "Richmond", 35, 10, 4,  2,  12, 0, 0, 0,  0, 1, 0, 3,  7,  0, 0, 6,  1, 0, 87, 2, 3, 0, 1, 0, "broad_nathan_15041993_performance_details.csv"),
    ("Tom",         "Brown",        "Richmond", 30, 1,  0,  0,  1,  0, 0, 0,  0, 0, 0, 0,  0,  0, 0, 1,  0, 0, 4,  0, 1, 0, 0, 0, "brown_tom_30072003_performance_details.csv"),
    ("Tom",         "Burton",       "Richmond", 45, 5,  2,  4,  9,  0, 1, 0,  1, 0, 0, 5,  4,  0, 1, 0,  1, 0, 78, 2, 1, 1, 2, 0, "burton_tom_09012007_performance_details.csv"),
    ("Seth",        "Campbell",     "Richmond", 44, 9,  4,  3,  12, 1, 0, 0,  0, 0, 2, 5,  8,  0, 0, 1,  1, 1, 84, 4, 0, 1, 5, 0, "campbell_seth_29122004_performance_details.csv"),
    ("Sam",         "Cumming",      "Richmond", 22, 4,  1,  4,  8,  1, 0, 0,  0, 0, 0, 2,  5,  0, 0, 0,  0, 0, 73, 0, 1, 0, 1, 0, "cumming_sam_27072007_performance_details.csv"),
    ("Jonty",       "Faull",        "Richmond", 8,  7,  3,  4,  11, 0, 0, 0,  1, 0, 0, 3,  7,  1, 0, 4,  0, 1, 82, 2, 0, 0, 4, 0, "faull_jonty_01022006_performance_details.csv"),
    ("Campbell",    "Gray",         "Richmond", 50, 8,  4,  1,  9,  0, 0, 0,  4, 1, 3, 1,  8,  0, 0, 9,  0, 0, 87, 1, 3, 0, 3, 0, "gray_campbell_01092003_performance_details.csv"),
    ("Steely",      "Green",        "Richmond", 48, 5,  5,  5,  10, 1, 0, 0,  5, 0, 0, 3,  7,  0, 1, 0,  0, 1, 84, 0, 1, 0, 0, 0, "green_steely_09012004_performance_details.csv"),
    ("Oliver",      "Hayes-Brown",  "Richmond", 47, 1,  1,  10, 11, 0, 0, 9,  1, 1, 0, 7,  4,  1, 0, 2,  0, 0, 48, 0, 0, 1, 0, 0, "hayes-brown_oliver_28042000_performance_details.csv"),
    ("Jacob",       "Hopper",       "Richmond", 2,  13, 3,  10, 23, 0, 0, 0,  3, 1, 2, 9,  14, 0, 1, 1,  0, 0, 80, 6, 2, 2, 7, 0, "hopper_jacob_06021997_performance_details.csv"),
    ("Mykelti",     "Lefau",        "Richmond", 42, 4,  3,  7,  11, 1, 1, 2,  2, 1, 0, 6,  5,  1, 1, 2,  0, 0, 85, 2, 0, 1, 2, 0, "lefau_mykelti_22061998_performance_details.csv"),
    ("Tom",         "Lynch",        "Richmond", 19, 8,  3,  6,  14, 3, 0, 0,  1, 4, 0, 10, 5,  2, 2, 2,  0, 2, 83, 1, 0, 1, 3, 0, "lynch_tom_31101992_performance_details.csv"),
    ("Kane",        "McAuliffe",    "Richmond", 28, 8,  3,  2,  10, 0, 1, 0,  0, 1, 3, 1,  10, 0, 0, 0,  0, 1, 58, 5, 0, 1, 5, 0, "mcauliffe_kane_01032005_performance_details.csv"),
    ("Ben",         "Miller",       "Richmond", 12, 13, 7,  5,  18, 0, 0, 0,  3, 1, 1, 6,  12, 1, 0, 10, 0, 0, 93, 1, 7, 1, 4, 0, "miller_ben_31081999_performance_details.csv"),
    ("Patrick",     "Retschko",     "Richmond", 33, 12, 6,  10, 22, 0, 0, 0,  0, 0, 1, 3,  21, 0, 0, 1,  0, 0, 94, 3, 2, 2, 4, 0, "retschko_patrick_28022006_performance_details.csv"),
    ("Jack",        "Ross",         "Richmond", 5,  8,  5,  18, 26, 0, 0, 0,  6, 0, 1, 10, 16, 0, 0, 1,  0, 0, 86, 4, 2, 3, 4, 0, "ross_jack_03092000_performance_details.csv"),
    ("Jayden",      "Short",        "Richmond", 15, 15, 10, 11, 26, 0, 0, 0,  0, 0, 1, 4,  19, 0, 0, 0,  0, 1, 83, 2, 4, 0, 2, 0, "short_jayden_24011996_performance_details.csv"),
    ("Tyler",       "Sonsie",       "Richmond", 40, 8,  3,  5,  13, 1, 0, 0,  1, 2, 2, 8,  7,  0, 1, 0,  0, 0, 89, 1, 0, 1, 2, 0, "sonsie_tyler_27012003_performance_details.csv"),
    ("Tim",         "Taranto",      "Richmond", 14, 12, 0,  16, 28, 0, 1, 0,  4, 2, 0, 18, 10, 0, 0, 1,  0, 0, 86, 2, 2, 9, 4, 0, "taranto_tim_28011998_performance_details.csv"),
    ("Luke",        "Trainor",      "Richmond", 11, 11, 10, 7,  18, 0, 0, 0,  1, 0, 0, 4,  13, 2, 0, 1,  0, 0, 86, 1, 2, 0, 1, 0, "trainor_luke_10042006_performance_details.csv"),
    ("James",       "Trezise",      "Richmond", 36, 12, 7,  6,  18, 0, 0, 0,  1, 1, 0, 4,  12, 0, 0, 1,  1, 0, 76, 3, 1, 0, 0, 0, "trezise_james_15062002_performance_details.csv"),
    ("Nick",        "Vlastuin",     "Richmond", 1,  11, 2,  10, 21, 0, 0, 0,  3, 2, 2, 8,  11, 0, 0, 2,  1, 0, 92, 0, 5, 0, 2, 0, "vlastuin_nick_19041994_performance_details.csv"),
    # Adelaide
    ("Sam",         "Berry",        "Adelaide", 3,  7,  4,  13, 20, 0, 1, 0,  3, 0, 1, 9,  12, 0, 0, 0,  0, 0, 68, 1, 0, 5, 2, 0, "berry_sam_12022002_performance_details.csv"),
    ("Hugh",        "Bond",         "Adelaide", 21, 6,  2,  6,  12, 0, 0, 0,  2, 1, 0, 6,  6,  0, 0, 0,  0, 0, 80, 1, 2, 1, 1, 0, "bond_hugh_25092004_performance_details.csv"),
    ("James",       "Borlase",      "Adelaide", 35, 11, 7,  4,  15, 0, 0, 0,  1, 0, 1, 3,  11, 1, 0, 6,  0, 0, 91, 0, 3, 0, 1, 0, "borlase_james_18072002_performance_details.csv"),
    ("Brayden",     "Cook",         "Adelaide", 15, 14, 7,  8,  22, 0, 0, 0,  1, 0, 0, 5,  17, 0, 0, 0,  2, 1, 81, 3, 6, 1, 1, 0, "cook_brayden_18072002_performance_details.csv"),
    ("Isaac",       "Cumming",      "Adelaide", 44, 12, 8,  3,  15, 1, 0, 0,  1, 0, 2, 1,  14, 0, 1, 0,  1, 1, 76, 4, 2, 1, 3, 0, "cumming_isaac_11081998_performance_details.csv"),
    ("Daniel",      "Curtin",       "Adelaide", 6,  9,  5,  2,  11, 1, 0, 0,  3, 2, 1, 4,  7,  0, 0, 1,  1, 1, 81, 4, 1, 1, 2, 0, "curtin_daniel_08032005_performance_details.csv"),
    ("Jordan",      "Dawson",       "Adelaide", 12, 22, 12, 7,  29, 1, 0, 0,  4, 1, 2, 10, 19, 1, 0, 0,  1, 1, 82, 7, 3, 3, 6, 0, "dawson_jordan_09041997_performance_details.csv"),
    ("Ben",         "Keays",        "Adelaide", 2,  9,  5,  6,  15, 1, 0, 0,  3, 1, 1, 5,  10, 1, 1, 0,  0, 1, 89, 3, 0, 0, 1, 0, "keays_ben_23021997_performance_details.csv"),
    ("Rory",        "Laird",        "Adelaide", 29, 16, 12, 11, 27, 0, 0, 0,  1, 0, 0, 6,  21, 0, 0, 0,  1, 0, 84, 1, 6, 1, 2, 0, "laird_rory_29121993_performance_details.csv"),
    ("Lachlan",     "McAndrew",     "Adelaide", 42, 6,  4,  3,  9,  1, 0, 33, 4, 1, 2, 7,  4,  1, 1, 0,  0, 0, 69, 2, 2, 3, 3, 0, "mcandrew_lachlan_26052000_performance_details.csv"),
    ("Max",         "Michalanney",  "Adelaide", 16, 11, 8,  3,  14, 0, 0, 0,  0, 0, 1, 0,  14, 0, 0, 3,  1, 0, 88, 2, 3, 0, 1, 0, "michalanney_max_26022004_performance_details.csv"),
    ("Wayne",       "Milera",       "Adelaide", 30, 18, 6,  16, 34, 0, 0, 0,  2, 1, 0, 10, 20, 1, 0, 2,  0, 0, 78, 2, 0, 1, 2, 0, "milera_wayne_14091997_performance_details.csv"),
    ("Nick",        "Murray",       "Adelaide", 9,  4,  2,  3,  7,  1, 0, 0,  3, 0, 3, 2,  5,  0, 0, 5,  0, 0, 92, 0, 0, 0, 4, 0, "murray_nick_18122000_performance_details.csv"),
    ("Toby",        "Murray",       "Adelaide", 39, 7,  2,  2,  9,  1, 1, 4,  6, 6, 1, 7,  4,  0, 0, 5,  0, 1, 68, 1, 0, 2, 1, 0, "murray_toby_03112003_performance_details.csv"),
    ("Alex",        "Neal-Bullen",  "Adelaide", 28, 11, 9,  5,  16, 2, 2, 0,  1, 0, 1, 2,  14, 1, 4, 0,  0, 1, 76, 2, 0, 0, 1, 0, "neal-bullen_alex_09011996_performance_details.csv"),
    ("James",       "Peatling",     "Adelaide", 25, 12, 7,  7,  19, 0, 1, 0,  5, 1, 0, 6,  13, 0, 0, 2,  0, 0, 69, 2, 4, 2, 3, 0, "peatling_james_21082000_performance_details.csv"),
    ("Luke",        "Pedlar",       "Adelaide", 10, 7,  6,  1,  8,  1, 1, 0,  3, 0, 0, 4,  4,  2, 1, 0,  0, 0, 69, 2, 0, 0, 0, 0, "pedlar_luke_17052002_performance_details.csv"),
    ("Josh",        "Rachele",      "Adelaide", 8,  8,  5,  6,  14, 2, 1, 0,  1, 1, 1, 6,  7,  0, 1, 0,  0, 2, 75, 3, 0, 1, 2, 0, "rachele_josh_11042003_performance_details.csv"),
    ("Izak",        "Rankine",      "Adelaide", 23, 24, 10, 9,  33, 1, 1, 0,  9, 2, 1, 10, 22, 0, 0, 1,  0, 0, 78, 8, 2, 4, 5, 0, "rankine_izak_23042000_performance_details.csv"),
    ("Jake",        "Soligo",       "Adelaide", 14, 5,  2,  3,  8,  0, 0, 0,  2, 0, 2, 5,  4,  0, 0, 1,  1, 0, 64, 1, 0, 2, 3, 0, "soligo_jake_25012003_performance_details.csv"),
    ("Zac",         "Taylor",       "Adelaide", 19, 10, 6,  5,  15, 0, 1, 0,  3, 1, 0, 3,  11, 1, 1, 1,  0, 0, 67, 4, 0, 1, 1, 0, "taylor_zac_31012003_performance_details.csv"),
    ("Riley",       "Thilthorpe",   "Adelaide", 7,  7,  5,  5,  12, 1, 3, 0,  1, 0, 0, 5,  6,  2, 4, 0,  0, 1, 86, 0, 0, 0, 0, 0, "thilthorpe_riley_07072002_performance_details.csv"),
    ("Josh",        "Worrell",      "Adelaide", 24, 13, 7,  8,  21, 0, 0, 0,  3, 0, 0, 2,  17, 0, 0, 2,  0, 0, 92, 1, 2, 0, 2, 0, "worrell_josh_11042001_performance_details.csv"),
]


def blank_if_zero(v):
    return v if v else ''


def main():
    updated, skipped, missing = [], [], []
    for row in DATA:
        (first, surname, team, jumper,
         k, m, h, d,
         g, b, ho, t,
         ff, fa, cp, up, cm, mi, op, bo, ga, pct,
         i50, r50, cl_clear, clangers, br, fname) = row
        path = os.path.join(PLAYER_DIR, fname)
        if not os.path.exists(path):
            missing.append((first, surname, fname))
            continue
        df = pd.read_csv(path)
        mask = (df['year'] == YEAR) & (df['round'] == ROUND)
        if mask.any():
            skipped.append((first, surname, "already present"))
            continue
        # games_played from previous row
        if 'games_played' in df.columns and len(df) > 0:
            try:
                gp = int(df['games_played'].iloc[-1]) + 1
            except Exception:
                gp = len(df) + 1
        else:
            gp = len(df) + 1

        new_row = {
            'team': team, 'year': YEAR, 'games_played': gp, 'opponent': OPP[team],
            'round': ROUND, 'result': RESULT[team], 'jersey_num': jumper,
            'kicks': k, 'marks': m, 'handballs': h, 'disposals': d,
            'goals': blank_if_zero(g), 'behinds': blank_if_zero(b),
            'hit_outs': blank_if_zero(ho), 'tackles': blank_if_zero(t),
            'rebound_50s': blank_if_zero(r50), 'inside_50s': blank_if_zero(i50),
            'clearances': blank_if_zero(cl_clear),
            'clangers': blank_if_zero(clangers),
            'free_kicks_for': blank_if_zero(ff), 'free_kicks_against': blank_if_zero(fa),
            'brownlow_votes': blank_if_zero(br),
            'contested_possessions': blank_if_zero(cp),
            'uncontested_possessions': blank_if_zero(up),
            'contested_marks': blank_if_zero(cm),
            'marks_inside_50': blank_if_zero(mi),
            'one_percenters': blank_if_zero(op),
            'bounces': blank_if_zero(bo),
            'goal_assist': blank_if_zero(ga),
            'percentage_of_game_played': pct, 'date': DATE,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(path, index=False)
        updated.append((first, surname))

    print(f"Updated: {len(updated)}")
    for u in updated:
        print(f"  + {u[0]} {u[1]}")
    print(f"Skipped: {len(skipped)}")
    for s in skipped:
        print(f"  - {s[0]} {s[1]}: {s[2]}")
    print(f"Missing: {len(missing)}")
    for x in missing:
        print(f"  X {x[0]} {x[1]} -> {x[2]}")


if __name__ == "__main__":
    main()
