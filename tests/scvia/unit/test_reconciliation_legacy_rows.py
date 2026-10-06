"""Whole legacy CSV rows rebuilt from captured pages: a missing player-game row and a corrupted match row."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from supercoach_via.reconciliation import corrections as CO
from supercoach_via.reconciliation import legacy_rows as LR
from supercoach_via.reconciliation import source as R

FX = Path(__file__).resolve().parents[1] / "fixtures" / "reconciliation"
PERF_HEADER = [
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
MATCH_HEADER = [
    "round_num",
    "venue",
    "date",
    "year",
    "attendance",
    "team_1_team_name",
    "team_1_q1_goals",
    "team_1_q1_behinds",
    "team_1_q2_goals",
    "team_1_q2_behinds",
    "team_1_q3_goals",
    "team_1_q3_behinds",
    "team_1_final_goals",
    "team_1_final_behinds",
    "team_2_team_name",
    "team_2_q1_goals",
    "team_2_q1_behinds",
    "team_2_q2_goals",
    "team_2_q2_behinds",
    "team_2_q3_goals",
    "team_2_q3_behinds",
    "team_2_final_goals",
    "team_2_final_behinds",
]


def _profile() -> R.ProfileFacts:
    url = "https://afltables.com/afl/stats/players/S/Scott_Pendlebury.html"
    return R.read_profile((FX / "profile_pendlebury.html").read_bytes(), url)


def test_a_player_game_row_is_the_profile_row_as_printed_with_the_match_date() -> None:
    g = next(x for x in _profile().games if x.rd_token == "SF" and x.season == 2007)
    row = LR.player_row(PERF_HEADER, g, "2007-09-14")
    rec = dict(zip(PERF_HEADER, row, strict=True))
    assert rec["team"] == "Collingwood" and rec["year"] == "2007" and rec["games_played"] == "31"
    assert (
        rec["opponent"] == "West Coast" and rec["round"] == "SF" and rec["result"] == "W" and rec["jersey_num"] == "10"
    )
    assert rec["kicks"] == "14" and rec["disposals"] == "26" and rec["hit_outs"] == ""  # blank stays blank
    assert rec["percentage_of_game_played"] == "81" and rec["date"] == "2007-09-14"


def test_a_match_row_is_built_from_the_match_page_in_the_files_column_order() -> None:
    m = R.read_match(
        (FX / "match_2021_r6.html").read_bytes(), "https://afltables.com/afl/stats/games/2021/162020210424.html"
    )
    rec = dict(zip(MATCH_HEADER, LR.match_row(MATCH_HEADER, m), strict=True))
    assert rec["round_num"] == "6" and rec["venue"] == "Carrara" and rec["date"] == "2021-04-24 13:45"
    assert rec["year"] == "2021" and rec["attendance"] == "9819"
    assert (
        rec["team_1_team_name"] == "Gold Coast"
        and rec["team_1_q1_goals"] == "4"
        and rec["team_1_final_behinds"] == "10"
    )
    assert rec["team_2_team_name"] == "Sydney" and rec["team_2_final_goals"] == "9"


def _files(tmp_path: Path) -> Path:
    pd = tmp_path / "data" / "player_data"
    md = tmp_path / "data" / "matches"
    pd.mkdir(parents=True)
    md.mkdir(parents=True)
    rows = [PERF_HEADER, ["Alpha", "2026", "1", "Beta", "1"] + [""] * (len(PERF_HEADER) - 6) + ["2026-03-05"],
            ["Alpha", "2026", "3", "Beta", "3"] + [""] * (len(PERF_HEADER) - 6) + ["2026-03-19"]]  # fmt: skip
    (pd / "a_b_01011990_performance_details.csv").write_text("\n".join(",".join(r) for r in rows) + "\n")
    m = [MATCH_HEADER, ["1", "MCG", "2026-03-05 19:30", "2026", "1", "Alpha"] + ["1"] * 8 + ["Beta"] + ["1"] * 8,
         ["QF", "MCG", "2026-09-05 19:30", "2026", "1", "Alpha"] + ["1"] * 8 + ["15.24.114"] + ["0"] * 8]  # fmt: skip
    (md / "matches_2026.csv").write_text("\n".join(",".join(r) for r in m) + "\n")
    return tmp_path


def test_a_missing_player_game_row_is_inserted_in_career_counter_order(tmp_path: Path) -> None:
    root = _files(tmp_path)
    new = ["Alpha", "2026", "2", "Beta", "2"] + [""] * (len(PERF_HEADER) - 6) + ["2026-03-12"]
    rel = "data/player_data/a_b_01011990_performance_details.csv"
    CO.apply_legacy(root, [CO.Change("legacy_csv", f"{rel}#insert", "row", None, json.dumps(new), "R-MISSING-LOCAL",
                                     "f", "u", "s")])  # fmt: skip
    counters = [line.split(",")[2] for line in (root / rel).read_text().splitlines()[1:]]
    assert counters == ["1", "2", "3"]


def test_a_corrupted_match_row_is_replaced_only_when_it_holds_the_audited_bytes(tmp_path: Path) -> None:
    root = _files(tmp_path)
    rel = "data/matches/matches_2026.csv"
    old = ["QF", "MCG", "2026-09-05 19:30", "2026", "1", "Alpha"] + ["1"] * 8 + ["15.24.114"] + ["0"] * 8
    new = ["QF", "MCG", "2026-09-05 19:30", "2026", "1", "Alpha"] + ["1"] * 8 + ["Gamma"] + ["2"] * 8
    stale = CO.Change("legacy_csv", f"{rel}#2", "row", json.dumps(new), json.dumps(old), "R-MATCH", "f", "u", "s")
    with pytest.raises(CO.CorrectionConflict, match="row"):
        CO.apply_legacy(root, [stale])
    CO.apply_legacy(root, [CO.Change("legacy_csv", f"{rel}#2", "row", json.dumps(old), json.dumps(new), "R-MATCH",
                                     "f", "u", "s")])  # fmt: skip
    assert (root / rel).read_text().splitlines()[2].split(",")[14] == "Gamma"


def test_a_missing_match_row_is_appended_in_date_order(tmp_path: Path) -> None:
    root = _files(tmp_path)
    rel = "data/matches/matches_2026.csv"
    new = ["GF", "MCG", "2026-09-26 14:30", "2026", "9", "Gamma"] + ["3"] * 8 + ["Delta"] + ["4"] * 8
    CO.apply_legacy(root, [CO.Change("legacy_csv", f"{rel}#insert", "row", None, json.dumps(new), "R-MATCH-MISSING",
                                     "f", "u", "s")])  # fmt: skip
    assert [line.split(",")[0] for line in (root / rel).read_text().splitlines()] == ["round_num", "1", "QF", "GF"]


def test_a_new_player_gets_both_legacy_files(tmp_path: Path) -> None:
    root = _files(tmp_path)
    personal = {"first_name": "Jack", "last_name": "Dalton", "born_date": "15-04-2004"}
    rows = [["Hawthorn", "2026", "1", "Geelong", "5"] + [""] * (len(PERF_HEADER) - 6) + ["2026-04-06"]]
    CO.apply_legacy(root, [CO.Change("legacy_csv", "data/player_data/dalton_jack_15042004", "create_player", None,
                                     json.dumps({"personal": personal, "rows": rows}), "R-PLAYER-MISSING", "f", "u",
                                     "s")])  # fmt: skip
    d = root / "data" / "player_data"
    assert (
        (d / "dalton_jack_15042004_personal_details.csv")
        .read_text()
        .splitlines()[1]
        .startswith("Jack,Dalton,15-04-2004")
    )
    perf = (d / "dalton_jack_15042004_performance_details.csv").read_text().splitlines()
    assert perf[0].split(",") == PERF_HEADER and perf[1].startswith("Hawthorn,2026,1,Geelong,5")
    with pytest.raises(CO.CorrectionConflict, match="exists"):
        CO.apply_legacy(root, [CO.Change("legacy_csv", "data/player_data/dalton_jack_15042004", "create_player", None,
                                         json.dumps({"personal": personal, "rows": rows}), "R", "f", "u", "s")])  # fmt: skip


def test_inserting_never_moves_an_existing_row_whose_counter_carries_a_sub_arrow(tmp_path: Path) -> None:
    root = _files(tmp_path)
    rel = "data/player_data/a_b_01011990_performance_details.csv"
    path = root / rel
    lines = path.read_text().splitlines()
    lines[1] = lines[1].replace("Alpha,2026,1,", "Alpha,2026,1↑,", 1)
    path.write_text("\n".join(lines) + "\n")
    new = ["Alpha", "2026", "2", "Beta", "2"] + [""] * (len(PERF_HEADER) - 6) + ["2026-03-12"]
    CO.apply_legacy(root, [CO.Change("legacy_csv", f"{rel}#insert", "row", None, json.dumps(new), "R", "f", "u", "s")])
    assert [line.split(",")[2] for line in path.read_text().splitlines()[1:]] == ["1↑", "2", "3"]


def test_an_extra_time_matchs_final_columns_hold_the_result_after_extra_time() -> None:
    m = R.read_match((FX / "match_2021_r6.html").read_bytes(), "u")
    gc, syd = m.teams
    et = m.model_copy(update={"teams": (
        gc.model_copy(update={"quarters": ((2, 3), (6, 12), (9, 12), (12, 19), (15, 24)), "points": (15, 48, 66, 91, 114)}),
        syd.model_copy(update={"quarters": ((4, 5), (5, 7), (10, 11), (13, 13), (14, 16)), "points": (29, 37, 71, 91, 100)}),
    )})  # fmt: skip
    rec = dict(zip(MATCH_HEADER, LR.match_row(MATCH_HEADER, et), strict=True))
    assert (rec["team_1_q3_goals"], rec["team_1_final_goals"], rec["team_1_final_behinds"]) == ("9", "15", "24")
    assert (rec["team_2_final_goals"], rec["team_2_final_behinds"]) == ("14", "16")
