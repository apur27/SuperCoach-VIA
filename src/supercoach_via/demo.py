"""Deterministic synthetic DEMO corpus in the legacy CSV layout.

Every club/player name is prefixed "Demo" and every value is generated from a fixed
seed; nothing here is a real statistic. The corpus deliberately contains the edge cases
the pipeline must handle: a drawn final plus replay, a postponed low-numbered round played
after later rounds, two matches on the same day, a same-name player pair, a mid-career
club transfer, blank (unrecorded) stat cells in the oldest season, a zero-disposal game
and a final round of scheduled fixtures with no scores (the forecast target).
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

CLUBS = ["Demo Harbour", "Demo Ridge", "Demo Valley", "Demo Coast", "Demo Plains", "Demo Forest"]
SEASONS = (2024, 2025, 2026)
ROUNDS = 10
SQUAD = 22
SEED = 20260923

MATCH_COLS = [
    "round_num",
    "venue",
    "date",
    "year",
    "attendance",
    *[
        f"team_1_{k}"
        for k in (
            "team_name",
            "q1_goals",
            "q1_behinds",
            "q2_goals",
            "q2_behinds",
            "q3_goals",
            "q3_behinds",
            "final_goals",
            "final_behinds",
        )
    ],
    *[
        f"team_2_{k}"
        for k in (
            "team_name",
            "q1_goals",
            "q1_behinds",
            "q2_goals",
            "q2_behinds",
            "q3_goals",
            "q3_behinds",
            "final_goals",
            "final_behinds",
        )
    ],
]
PLAYER_COLS = [
    "team",
    "year",
    "games_played",
    "opponent",
    "round",
    "result",
    "jersey_num",
    "kicks",
    "marks",
    "handballs",
    "disposals",
    "goals",
    "behinds",
    "hit_outs",
    "tackles",
    "rebound_50s",
    "inside_50s",
    "clearances",
    "clangers",
    "free_kicks_for",
    "free_kicks_against",
    "brownlow_votes",
    "contested_possessions",
    "uncontested_possessions",
    "contested_marks",
    "marks_inside_50",
    "one_percenters",
    "bounces",
    "goal_assist",
    "percentage_of_game_played",
    "date",
]
MODERN_ONLY = (
    "contested_possessions",
    "uncontested_possessions",
    "contested_marks",
    "marks_inside_50",
    "one_percenters",
    "bounces",
    "goal_assist",
    "percentage_of_game_played",
    "rebound_50s",
    "inside_50s",
    "clearances",
    "clangers",
)
VENUES = {c: f"Demo Oval {i + 1}" for i, c in enumerate(CLUBS)}


@dataclass
class DemoPlayer:
    slug: str
    first: str
    last: str
    dob: date
    clubs: dict[int, str]  # season -> club
    skill: float
    games: int = 0


def _players(rng: random.Random) -> list[DemoPlayer]:
    players: list[DemoPlayer] = []
    n = 0
    for ci, club in enumerate(CLUBS):
        for j in range(SQUAD):
            n += 1
            dob = date(1995, 1, 1) + timedelta(days=rng.randrange(0, 3650))
            last = f"Player{n:03d}"
            first = "Demo"
            players.append(
                DemoPlayer(
                    slug=f"{last.lower()}_{first.lower()}_{dob.strftime('%d%m%Y')}",
                    first=first,
                    last=last,
                    dob=dob,
                    clubs={s: club for s in SEASONS},
                    skill=rng.uniform(8, 30),
                )
            )
            if ci == 0 and j == 0:
                players[-1].skill = 3.0  # low-volume player; produces zero-disposal games
    # same-name pair on different clubs, distinguished only by DOB
    players[1].first, players[1].last = "Demo", "Samename"
    players[1].slug = f"samename_demo_{players[1].dob.strftime('%d%m%Y')}"
    players[SQUAD + 1].first, players[SQUAD + 1].last = "Demo", "Samename"
    players[SQUAD + 1].slug = f"samename_demo_{players[SQUAD + 1].dob.strftime('%d%m%Y')}"
    if players[1].slug == players[SQUAD + 1].slug:  # pragma: no cover - seed guard
        raise RuntimeError("demo seed produced colliding same-name DOBs")
    # a transfer: from club 2 to club 3 from 2026
    players[2 * SQUAD + 3].clubs[2026] = CLUBS[3]
    return players


def _fixture(season: int) -> list[tuple[str, date, str, str, str]]:
    """(round_label, date, time, home, away) — includes a postponed round 2."""
    rows: list[tuple[str, date, str, str, str]] = []
    start = date(season, 3, 14)
    for r in range(1, ROUNDS + 1):
        rot = CLUBS[:1] + CLUBS[1:][r % 5 :] + CLUBS[1:][: r % 5]
        pairs = [(rot[0], rot[5]), (rot[1], rot[4]), (rot[2], rot[3])]
        day = start + timedelta(days=7 * (r - 1))
        if r == 2:
            day = start + timedelta(days=7 * 4 + 3)  # postponed: played after round 4
        for k, (h, a) in enumerate(pairs):
            t = "13:45" if k < 2 else "19:40"
            if k == 1 and r == 5:
                t = "16:35"  # two matches the same day (k=0 and k=1 already share a day)
            rows.append((str(r), day + timedelta(days=0 if k < 2 else 1), t, h, a))
    return rows


def _score(rng: random.Random, strength: float) -> list[int]:
    qs, g, b = [], 0, 0
    for _ in range(4):
        g += max(0, int(rng.gauss(3 * strength, 1.5)))
        b += max(0, int(rng.gauss(2.5, 1.2)))
        qs += [g, b]
    return qs


def write_demo_corpus(root: Path, seed: int = SEED) -> dict[str, int]:
    """Write the demo corpus under ``root/data``; returns row counts."""
    rng = random.Random(seed)  # noqa: S311 - seeded synthetic demo data, not security-relevant
    players = _players(rng)
    data = root / "data"
    for sub in ("matches", "player_data", "lineups"):
        (data / sub).mkdir(parents=True, exist_ok=True)
    strength = {c: rng.uniform(0.8, 1.2) for c in CLUBS}
    perf: dict[str, list[dict[str, str]]] = {p.slug: [] for p in players}
    lineups: dict[str, list[list[str]]] = {c: [] for c in CLUBS}
    counts = {"matches": 0, "player_games": 0, "scheduled": 0}

    for season in SEASONS:
        matches: list[dict[str, str]] = []
        games = [(lbl, d, t, h, a, lbl) for lbl, d, t, h, a in _fixture(season)]
        last_day = max(g[1] for g in games)
        finals = [
            ("Qualifying Final", "QF", last_day + timedelta(days=7), CLUBS[0], CLUBS[1]),
            ("Grand Final", "GF", last_day + timedelta(days=14), CLUBS[0], CLUBS[2]),
        ]
        for name, token, d, h, a in finals:
            games.append((name, d, "14:30", h, a, token))
        if season == 2025:  # drawn grand final, replayed a week later
            games.append(("Grand Final", last_day + timedelta(days=21), "14:30", CLUBS[0], CLUBS[2], "GF"))
        scheduled_round = str(ROUNDS) if season == 2026 else None
        gf_draw_done = False
        for label, d, t, home, away, token in games:
            row = {
                "round_num": label,
                "venue": VENUES[home],
                "date": f"{d.isoformat()} {t}",
                "year": str(season),
                "attendance": str(rng.randrange(8000, 60000)),
                "team_1_team_name": home,
                "team_2_team_name": away,
            }
            future = scheduled_round == label or (season == 2026 and token in ("QF", "GF"))
            if future:
                for side in ("team_1", "team_2"):
                    for k in (
                        "q1_goals",
                        "q1_behinds",
                        "q2_goals",
                        "q2_behinds",
                        "q3_goals",
                        "q3_behinds",
                        "final_goals",
                        "final_behinds",
                    ):
                        row[f"{side}_{k}"] = ""
                row["attendance"] = ""
                counts["scheduled"] += 1
                if token not in ("QF", "GF"):
                    matches.append(row)
                continue
            hs, as_ = _score(rng, strength[home]), _score(rng, strength[away])
            if season == 2025 and token == "GF" and not gf_draw_done:  # noqa: S105 - round token, not a secret
                as_ = list(hs)  # drawn grand final
                gf_draw_done = True
            for side, qs in (("team_1", hs), ("team_2", as_)):
                for i, k in enumerate(("q1", "q2", "q3", "final")):
                    row[f"{side}_{k}_goals"], row[f"{side}_{k}_behinds"] = str(qs[2 * i]), str(qs[2 * i + 1])
            matches.append(row)
            counts["matches"] += 1
            h_pts, a_pts = hs[6] * 6 + hs[7], as_[6] * 6 + as_[7]
            for club, opp, pts, opp_pts in ((home, away, h_pts, a_pts), (away, home, a_pts, h_pts)):
                result = "W" if pts > opp_pts else "L" if pts < opp_pts else "D"
                squad = [p for p in players if p.clubs[season] == club]
                named = []
                for jersey, p in enumerate(squad[:18], start=1):
                    p.games += 1
                    named.append(f"{p.first} {p.last}")
                    kicks_n = max(0, round(rng.gauss(p.skill * 0.55, 3)))
                    hb = max(0, round(rng.gauss(p.skill * 0.45, 3)))
                    if p.skill < 4 and rng.random() < 0.3:
                        kicks_n = hb = 0
                    stats = {
                        "kicks": kicks_n,
                        "handballs": hb,
                        "disposals": kicks_n + hb,
                        "marks": max(0, round(rng.gauss(4, 2))),
                        "goals": max(0, round(rng.gauss(0.8, 1))),
                        "behinds": max(0, round(rng.gauss(0.6, 0.8))),
                        "hit_outs": 0,
                        "tackles": max(0, round(rng.gauss(3, 2))),
                        "rebound_50s": max(0, round(rng.gauss(1, 1))),
                        "inside_50s": max(0, round(rng.gauss(2, 1.5))),
                        "clearances": max(0, round(rng.gauss(2, 1.5))),
                        "clangers": max(0, round(rng.gauss(2, 1))),
                        "free_kicks_for": max(0, round(rng.gauss(1, 1))),
                        "free_kicks_against": max(0, round(rng.gauss(1, 1))),
                        "brownlow_votes": 0,
                        "contested_possessions": max(0, round(rng.gauss(p.skill * 0.4, 2))),
                        "uncontested_possessions": max(0, round(rng.gauss(p.skill * 0.6, 2))),
                        "contested_marks": max(0, round(rng.gauss(1, 1))),
                        "marks_inside_50": max(0, round(rng.gauss(0.7, 1))),
                        "one_percenters": max(0, round(rng.gauss(2, 1.5))),
                        "bounces": max(0, round(rng.gauss(0.5, 0.7))),
                        "goal_assist": max(0, round(rng.gauss(0.5, 0.7))),
                        "percentage_of_game_played": rng.randrange(60, 100),
                    }
                    rec = {c: str(v) for c, v in stats.items()}
                    if season == 2024:
                        for c in MODERN_ONLY:
                            rec[c] = ""  # not recorded in the oldest demo season
                    perf[p.slug].append(
                        {
                            **rec,
                            "team": club,
                            "year": str(season),
                            "games_played": str(p.games),
                            "opponent": opp,
                            "round": token,
                            "result": result,
                            "jersey_num": str(jersey),
                            "date": d.isoformat(),
                        }
                    )
                    counts["player_games"] += 1
                lineups[club].append([str(season), f"{d.isoformat()} {t}", label, club, ";".join(named)])
        with (data / "matches" / f"matches_{season}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=MATCH_COLS, lineterminator="\n")
            w.writeheader()
            w.writerows(matches)

    for p in players:
        with (data / "player_data" / f"{p.slug}_performance_details.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=PLAYER_COLS, lineterminator="\n")
            w.writeheader()
            w.writerows(perf[p.slug])
        with (data / "player_data" / f"{p.slug}_personal_details.csv").open("w", newline="", encoding="utf-8") as fh:
            fh.write("first_name,last_name,born_date,debut_date,height,weight\n")
            fh.write(f"{p.first},{p.last},{p.dob.strftime('%d-%m-%Y')},,,\n")
    for club, rows in lineups.items():
        fname = club.lower().replace(" ", "_")
        with (data / "lineups" / f"team_lineups_{fname}.csv").open("w", newline="", encoding="utf-8") as fh:
            lw = csv.writer(fh, lineterminator="\n")
            lw.writerow(["year", "date", "round_num", "team_name", "players"])
            lw.writerows(rows)
    return counts
