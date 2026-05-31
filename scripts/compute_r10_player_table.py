"""
Generate the full player stat table for R10 2026 (Richmond vs Adelaide),
combining:
- afltables canonical box-score numbers (kicks, marks, handballs, disposals,
  goals, behinds, hit_outs, tackles, clangers)
- snapshot AF / SC / Q1-Q4 AF breakdown (only fields the snapshot is reliable
  for; verified against the AF formula)

Output: markdown table sorted by SC descending for each team.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config


SNAPSHOT = os.path.join(config.LIVE_SNAPSHOTS_DIR, "9781_20260510_1751_final-siren.json")


# afltables truth: (team, name) -> dict
AFL = {
    # Richmond
    ("RI", "Noah Balta"):       dict(k=11, m=6, h=3, d=14, g=1, b=1, ho=8, t=1, c=1),
    ("RI", "Nathan Broad"):     dict(k=10, m=4, h=2, d=12, g=0, b=0, ho=0, t=0, c=1),
    ("RI", "Tom Brown"):        dict(k=1, m=0, h=0, d=1, g=0, b=0, ho=0, t=0, c=0),
    ("RI", "Tom Burton"):       dict(k=5, m=2, h=4, d=9, g=0, b=1, ho=0, t=1, c=2),
    ("RI", "Seth Campbell"):    dict(k=9, m=4, h=3, d=12, g=1, b=0, ho=0, t=0, c=5),
    ("RI", "Sam Cumming"):      dict(k=4, m=1, h=4, d=8, g=1, b=0, ho=0, t=0, c=1),
    ("RI", "Jonty Faull"):      dict(k=7, m=3, h=4, d=11, g=0, b=0, ho=0, t=1, c=4),
    ("RI", "Campbell Gray"):    dict(k=8, m=4, h=1, d=9, g=0, b=0, ho=0, t=4, c=3),
    ("RI", "Steely Green"):     dict(k=5, m=5, h=5, d=10, g=1, b=0, ho=0, t=5, c=0),
    ("RI", "Oliver Hayes-Brown"):dict(k=1, m=1, h=10, d=11, g=0, b=0, ho=9, t=1, c=0),
    ("RI", "Jacob Hopper"):     dict(k=13, m=3, h=10, d=23, g=0, b=0, ho=0, t=3, c=7),
    ("RI", "Mykelti Lefau"):    dict(k=4, m=3, h=7, d=11, g=1, b=1, ho=2, t=2, c=2),
    ("RI", "Tom J. Lynch"):     dict(k=8, m=3, h=6, d=14, g=3, b=0, ho=0, t=1, c=3),
    ("RI", "Kane McAuliffe"):   dict(k=8, m=3, h=2, d=10, g=0, b=1, ho=0, t=0, c=5),
    ("RI", "Ben Miller"):       dict(k=13, m=7, h=5, d=18, g=0, b=0, ho=0, t=3, c=4),
    ("RI", "Patrick Retschko"): dict(k=12, m=6, h=10, d=22, g=0, b=0, ho=0, t=0, c=4),
    ("RI", "Jack Ross"):        dict(k=8, m=5, h=18, d=26, g=0, b=0, ho=0, t=6, c=4),
    ("RI", "Jayden Short"):     dict(k=15, m=10, h=11, d=26, g=0, b=0, ho=0, t=0, c=2),
    ("RI", "Tyler Sonsie"):     dict(k=8, m=3, h=5, d=13, g=1, b=0, ho=0, t=1, c=2),
    ("RI", "Tim Taranto"):      dict(k=12, m=0, h=16, d=28, g=0, b=1, ho=0, t=4, c=4),
    ("RI", "Luke Trainor"):     dict(k=11, m=10, h=7, d=18, g=0, b=0, ho=0, t=1, c=1),
    ("RI", "James Trezise"):    dict(k=12, m=7, h=6, d=18, g=0, b=0, ho=0, t=1, c=0),
    ("RI", "Nick Vlastuin"):    dict(k=11, m=2, h=10, d=21, g=0, b=0, ho=0, t=3, c=2),
    # Adelaide
    ("AD", "Sam Berry"):        dict(k=7, m=4, h=13, d=20, g=0, b=1, ho=0, t=3, c=2),
    ("AD", "Hugh Bond"):        dict(k=6, m=2, h=6, d=12, g=0, b=0, ho=0, t=2, c=1),
    ("AD", "James Borlase"):    dict(k=11, m=7, h=4, d=15, g=0, b=0, ho=0, t=1, c=1),
    ("AD", "Brayden Cook"):     dict(k=14, m=7, h=8, d=22, g=0, b=0, ho=0, t=1, c=1),
    ("AD", "Isaac Cumming"):    dict(k=12, m=8, h=3, d=15, g=1, b=0, ho=0, t=1, c=3),
    ("AD", "Daniel Curtin"):    dict(k=9, m=5, h=2, d=11, g=1, b=0, ho=0, t=3, c=2),
    ("AD", "Jordan Dawson"):    dict(k=22, m=12, h=7, d=29, g=1, b=0, ho=0, t=4, c=6),
    ("AD", "Ben Keays"):        dict(k=9, m=5, h=6, d=15, g=1, b=0, ho=0, t=3, c=1),
    ("AD", "Rory Laird"):       dict(k=16, m=12, h=11, d=27, g=0, b=0, ho=0, t=1, c=2),
    ("AD", "Lachlan McAndrew"): dict(k=6, m=4, h=3, d=9, g=1, b=0, ho=33, t=4, c=3),
    ("AD", "Max Michalanney"):  dict(k=11, m=8, h=3, d=14, g=0, b=0, ho=0, t=0, c=1),
    ("AD", "Wayne Milera"):     dict(k=18, m=6, h=16, d=34, g=0, b=0, ho=0, t=2, c=2),
    ("AD", "Nick Murray"):      dict(k=4, m=2, h=3, d=7, g=1, b=0, ho=0, t=3, c=4),
    ("AD", "Toby Murray"):      dict(k=7, m=2, h=2, d=9, g=1, b=1, ho=4, t=6, c=1),
    ("AD", "Alex Neal-Bullen"): dict(k=11, m=9, h=5, d=16, g=2, b=2, ho=0, t=1, c=1),
    ("AD", "James Peatling"):   dict(k=12, m=7, h=7, d=19, g=0, b=1, ho=0, t=5, c=3),
    ("AD", "Luke Pedlar"):      dict(k=7, m=6, h=1, d=8, g=1, b=1, ho=0, t=3, c=0),
    ("AD", "Josh Rachele"):     dict(k=8, m=5, h=6, d=14, g=2, b=1, ho=0, t=1, c=2),
    ("AD", "Izak Rankine"):     dict(k=24, m=10, h=9, d=33, g=1, b=1, ho=0, t=9, c=5),
    ("AD", "Jake Soligo"):      dict(k=5, m=2, h=3, d=8, g=0, b=0, ho=0, t=2, c=3),
    ("AD", "Zac Taylor"):       dict(k=10, m=6, h=5, d=15, g=0, b=1, ho=0, t=3, c=1),
    ("AD", "Riley Thilthorpe"): dict(k=7, m=5, h=5, d=12, g=1, b=3, ho=0, t=1, c=0),
    ("AD", "Josh Worrell"):     dict(k=13, m=7, h=8, d=21, g=0, b=0, ho=0, t=3, c=2),
}


def main():
    with open(SNAPSHOT) as f:
        d = json.load(f)

    rows = []
    for p in d['players']:
        name = f"{p['first_name']} {p['surname']}"
        key = (p['team'], name)
        afl = AFL.get(key)
        if not afl:
            print(f"NO AFL FOR: {key}")
            continue
        rows.append({
            'team': p['team'],
            'name': name,
            'd': afl['d'], 'k': afl['k'], 'h': afl['h'], 'm': afl['m'],
            'g': afl['g'], 'b': afl['b'], 'ho': afl['ho'], 't': afl['t'], 'c': afl['c'],
            'af': p['af'], 'sc': p['sc'],
            'q1': p['af_q1'], 'q2': p['af_q2'], 'q3': p['af_q3'], 'q4': p['af_q4'],
            'sc_q1': p['sc_q1'], 'sc_q2': p['sc_q2'], 'sc_q3': p['sc_q3'], 'sc_q4': p['sc_q4'],
            'tog': p['tog_pct'], 'de': p['de_pct'],
        })

    # Print Richmond first then Adelaide, each sorted by SC desc
    for team_code, team_name in [('RI', 'Richmond'), ('AD', 'Adelaide')]:
        team_rows = [r for r in rows if r['team'] == team_code]
        team_rows.sort(key=lambda r: -r['sc'])
        print(f"\n## {team_name}")
        print()
        print("| # | Player | D | K | H | M | G | B | HO | T | C | AF | SC | Q1 AF | Q2 AF | Q3 AF | Q4 AF | TOG% | DE% |")
        print("|---|--------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for i, r in enumerate(team_rows, 1):
            print(f"| {i} | {r['name']} | {r['d']} | {r['k']} | {r['h']} | {r['m']} | {r['g']} | {r['b']} | {r['ho']} | {r['t']} | {r['c']} | {r['af']} | {r['sc']} | {r['q1']} | {r['q2']} | {r['q3']} | {r['q4']} | {r['tog']} | {r['de']} |")

        # Team totals
        print()
        print(f"**{team_name} totals:** "
              f"D {sum(r['d'] for r in team_rows)}, "
              f"K {sum(r['k'] for r in team_rows)}, "
              f"H {sum(r['h'] for r in team_rows)}, "
              f"M {sum(r['m'] for r in team_rows)}, "
              f"G {sum(r['g'] for r in team_rows)}, "
              f"B {sum(r['b'] for r in team_rows)}, "
              f"HO {sum(r['ho'] for r in team_rows)}, "
              f"T {sum(r['t'] for r in team_rows)}, "
              f"C {sum(r['c'] for r in team_rows)}, "
              f"AF {sum(r['af'] for r in team_rows)}, "
              f"SC {sum(r['sc'] for r in team_rows)}")

        # Q1-Q4 team AF totals
        print(f"  Q1 AF {sum(r['q1'] for r in team_rows)}, "
              f"Q2 AF {sum(r['q2'] for r in team_rows)}, "
              f"Q3 AF {sum(r['q3'] for r in team_rows)}, "
              f"Q4 AF {sum(r['q4'] for r in team_rows)}")


if __name__ == "__main__":
    main()
