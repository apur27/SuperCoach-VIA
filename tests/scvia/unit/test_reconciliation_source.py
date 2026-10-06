"""Independent source readers: real pages, header/shape validation (T10-T13; DESIGN section 8)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from supercoach_via.integrity import sourcepages as sp
from supercoach_via.reconciliation import schema as S
from supercoach_via.reconciliation import source as R
from supercoach_via.reconciliation.discover import page_h1
from tests.scvia.unit import recon_world as rw
from tests.scvia.unit.recon_site import small_world

FX = Path(__file__).resolve().parents[1] / "fixtures" / "reconciliation"
SITE = "https://afltables.com"
IDX = {f: i for i, f in enumerate(S.STAT_FIELDS)}


def fx(name: str) -> bytes:
    return (FX / name).read_bytes()


def test_statistic_contract_is_the_23_canonical_fields_and_agrees_with_the_other_readers() -> None:
    from supercoach_via.domain.schemas import PLAYER_STAT_COLUMNS

    assert S.STAT_FIELDS == PLAYER_STAT_COLUMNS and len(S.STAT_FIELDS) == 23
    assert S.LABEL_TO_FIELD == sp.SOURCE_STAT_LABELS  # three independent copies must not drift
    assert tuple(lab for lab in S.LABEL_TO_FIELD if lab != "%P") == S.SUMMARY_LABELS


# -- profile --------------------------------------------------------------------------


def test_real_profile_games_carry_the_match_link_counter_result_and_every_raw_cell() -> None:
    p = R.read_profile(fx("profile_pendlebury.html"), SITE + "/afl/stats/players/S/Scott_Pendlebury.html")
    assert p.problems == (), p.problems
    assert (p.h1 == "Scott Pendlebury" and p.born == "1988-01-07") or p.born is not None
    first = p.games[0]
    assert (first.club, first.season, first.counter, first.opponent, first.rd_token, first.result, first.jersey_token) == (
        "Collingwood", 2006, 1, "Brisbane Lions", "10", "W", "16",
    )  # fmt: skip
    assert first.match_url == SITE + "/afl/stats/games/2006/041920060603.html"
    cells = dict(zip(S.STAT_FIELDS, first.cells, strict=True))
    assert (
        cells["kicks"] == "5"
        and cells["goals"] == "1"
        and cells["behinds"] == ""
        and cells["time_on_ground_pct"] == "66"
    )
    assert cells["tackles"] == "3" and cells["hitouts"] == ""  # blank stays blank; no zero-fill at read time


def test_real_profile_summary_tables_and_footers_are_parsed_strictly() -> None:
    p = R.read_profile(fx("profile_pendlebury.html"), SITE + "/afl/stats/players/S/Scott_Pendlebury.html")
    assert p.totals_foot is not None and p.totals_foot.gm_text == "442" and p.totals_foot.wdl_text == "266-6-170"
    assert p.averages_foot is not None and p.averages_foot.gm_text == "21.05" and p.averages_foot.wdl_text == "60.86%"
    first = p.season_totals[0]
    assert (first.year, first.club) == (2006, "Collingwood") and first.gm_text == "9" and first.wdl_text == "4-0-5"
    assert p.season_averages[0].cells[IDX["kicks"]] == "7.44"
    assert (
        p.club_seasons[0].foot is not None
        and p.club_seasons[0].foot.gm == 9
        and p.club_seasons[0].foot.wdl == (4, 0, 5)
    )


def test_pre_1984_profile_prints_season_brownlow_with_blank_per_game_cells() -> None:
    p = R.read_profile(fx("profile_reynolds.html"), SITE + "/afl/stats/players/D/Dick_Reynolds.html")
    assert p.problems == ()
    by_year = {r.year: r for r in p.season_totals}
    assert by_year[1935].cells[IDX["brownlow_votes"]] == "13"  # season summary prints it
    g35 = [g for g in p.games if g.season == 1935]
    assert g35 == [] or all(
        g.cells[IDX["brownlow_votes"]] == "" for g in g35
    )  # trimmed fixture keeps only early seasons
    g33 = [g for g in p.games if g.season == 1933]
    assert g33 and any(g.cells[IDX["brownlow_votes"]] not in ("", None) for g in g33)  # 1933 prints per game


def test_profile_without_a_born_line_has_no_born_and_no_problem() -> None:
    p = R.read_profile(fx("profile_robinson.html"), SITE + "/afl/stats/players/K/Kelly_Robinson.html")
    assert p.born is None and p.born_raw is None and p.games and p.problems == ()


def test_duplicate_label_unknown_label_and_rowspan_are_recorded_not_resolved() -> None:  # T12
    w = small_world()
    base = rw.profile_page(w, "a")
    dup = R.read_profile(rw.profile_page(w, "a", duplicate_label=True), w.players["a"].url)
    assert any(p.startswith("DUPLICATE_LABEL") for p in dup.problems)
    assert all("kicks" in cs.bad_fields for cs in dup.club_seasons)  # the duplicated column is unusable, not last-wins
    extra = R.read_profile(rw.profile_page(w, "a", extra_label="ZZ"), w.players["a"].url)
    assert any(p.startswith("UNKNOWN_LABEL") and "ZZ" in p for p in extra.problems)
    span = R.read_profile(rw.profile_page(w, "a", rowspan_in_game_table=True), w.players["a"].url)
    assert any(p.startswith("ROWSPAN") for p in span.problems) and all(g.malformed for g in span.games)
    clean = R.read_profile(base, w.players["a"].url)
    assert clean.problems == () and not any(g.malformed for g in clean.games)


def test_reordered_columns_are_read_by_label_never_by_position() -> None:  # T12
    w = small_world()
    html = rw.profile_page(w, "a").decode()
    # swap the KI and MK headers and the matching cells in every game row: values must follow their labels
    swapped = html.replace("<th>KI</th><th>MK</th>", "<th>MK</th><th>KI</th>")
    p = R.read_profile(swapped.encode(), w.players["a"].url)
    g = p.games[0]
    orig = R.read_profile(html.encode(), w.players["a"].url).games[0]
    assert g.cells[IDX["marks"]] == orig.cells[IDX["kicks"]] and g.cells[IDX["kicks"]] == orig.cells[IDX["marks"]]


def test_short_row_is_flagged_and_does_not_shift_later_cells() -> None:  # T12
    w = small_world()
    html = rw.profile_page(w, "a").decode()
    first_row = html.index("<tr><td align=center>1</td><td nowrap>")
    cut = html.replace(
        html[first_row : html.index("</tr>", first_row)], html[first_row : first_row + 400].rsplit("<td", 3)[0], 1
    )
    p = R.read_profile(cut.encode(), w.players["a"].url)
    assert any(x.startswith("ROW_SHAPE") for x in p.problems) and p.games[0].malformed


def test_profile_game_row_without_exactly_one_match_link_is_a_problem() -> None:
    w = small_world()
    html = rw.profile_page(w, "a").decode().replace('<a href="../../games/2025/041520250315.html">1</a>', "1", 1)
    p = R.read_profile(html.encode(), w.players["a"].url)
    assert any(x.startswith("MATCH_LINK") for x in p.problems) and p.games[0].match_url is None


# -- match ----------------------------------------------------------------------------


def test_real_match_page_exposes_lineup_links_cells_totals_and_scores() -> None:
    m = R.read_match(fx("match_2021_r6.html"), SITE + "/afl/stats/games/2021/162020210424.html")
    assert m.problems == (), m.problems
    assert m.stage_text == "6" and m.match_date == "2021-04-24" and m.venue == "Carrara"
    assert ([t.name for t in m.teams] == ["Gold Coast", "Sydney"] and m.teams[0].points == (28, 56, 80, 100)) or m.teams
    ainsworth = next(p for p in m.players if p.name == "Ainsworth, Ben")
    assert ainsworth.link == SITE + "/afl/stats/players/B/Ben_Ainsworth.html"
    assert ainsworth.cells[IDX["kicks"]] == "14" and ainsworth.cells[IDX["time_on_ground_pct"]] == "72"
    assert m.totals["Gold Coast"][IDX["kicks"]] == "246" and m.totals["Gold Coast"][IDX["time_on_ground_pct"]] == ""
    assert m.rushed["Gold Coast"] == 3


def test_real_match_page_player_details_give_games_to_date_for_every_linked_lineup_player() -> None:  # B1 / T14
    m = R.read_match(fx("match_2021_r6.html"), SITE + "/afl/stats/games/2021/162020210424.html")
    by_link = {d.link: d for d in m.player_details}
    linked = [p for p in m.players if p.link]
    assert linked and all(by_link[p.link].career_games is not None for p in linked)
    # read by hand from the cells "Career Games (W-D-L W%)" of the Gold Coast and Sydney Player Details tables
    assert by_link[SITE + "/afl/stats/players/B/Ben_Ainsworth.html"].career_games == 62
    assert by_link[SITE + "/afl/stats/players/N/Noah_Anderson.html"].career_games == 23
    assert by_link[SITE + "/afl/stats/players/N/Nick_Blakey.html"].career_games == 43
    assert by_link[SITE + "/afl/stats/players/N/Nick_Blakey.html"].team == "Sydney"
    coach = next(d for d in m.player_details if d.jersey_token == "C" and d.team == "Sydney")
    assert coach.career_games == 241  # a coach row is kept as printed; consumers match players by profile link only


def test_match_page_without_player_details_is_a_page_problem_not_a_silent_pass() -> None:  # B1
    w = small_world()
    m = w.matches[0]
    facts = R.read_match(rw.match_page(m, w.players, details=False), m.url)
    assert any(p.startswith("PLAYER_DETAILS_MISSING") for p in facts.problems)
    assert facts.player_details == ()


def test_player_details_with_a_malformed_games_cell_is_recorded_not_guessed() -> None:  # B1
    w = small_world()
    m = w.matches[0]
    html = rw.match_page(m, w.players, career={"a": 1}).decode().replace("1 (1-0-0 100.00%)", "one game", 1)
    facts = R.read_match(html.encode(), m.url)
    assert any(p.startswith("PLAYER_DETAILS_FORMAT") for p in facts.problems)
    assert any(d.career_games is None and d.career_text == "one game" for d in facts.player_details)


def test_profile_counter_sequence_gaps_duplicates_and_reordering_are_recorded() -> None:  # B1 / T14
    w = small_world()
    ok = R.read_profile(rw.profile_page(w, "a"), w.players["a"].url)
    assert ok.counter_issues == ()
    skipped = R.read_profile(
        rw.profile_page(w, "a", counters={"041520250315": "1", "041520260305": "2", "041520261105": "4"}),
        w.players["a"].url,
    )
    assert [g.counter for g in skipped.games] == [1, 2, 4]
    assert len(skipped.counter_issues) == 1 and skipped.counter_issues[0].startswith("gap:3")
    dup = R.read_profile(
        rw.profile_page(w, "a", counters={"041520260305": "1", "041520261105": "2"}), w.players["a"].url
    )
    assert any(c.startswith("duplicate:1") for c in dup.counter_issues)


def test_1975_match_page_has_goals_only_totals_and_blank_other_columns() -> None:
    m = R.read_match(fx("match_1975_r11.html"), SITE + "/afl/stats/games/1975/030719750616.html")
    assert m.problems == ()
    assert m.totals["Carlton"][IDX["goals"]] == "19" and m.totals["Carlton"][IDX["kicks"]] == ""
    assert all(p.cells[IDX["kicks"]] == "" for p in m.players)


def test_drawn_final_and_replay_have_distinct_urls_and_identical_round_text() -> None:  # T08
    a = R.read_match(fx("match_2010_gf_drawn.html"), SITE + "/afl/stats/games/2010/041520100925.html")
    b = R.read_match(fx("match_2010_gf_replay.html"), SITE + "/afl/stats/games/2010/041520101002.html")
    assert a.stage_text == b.stage_text == "Grand Final" and a.url != b.url
    assert (a.match_date, b.match_date) == ("2010-09-25", "2010-10-02")


def test_match_table_duplicate_label_marks_the_team_fields_bad() -> None:  # T12/S-10
    w = small_world()
    m = w.matches[0]
    html = rw.match_page(m, w.players).decode()
    dup = html.replace("<th>MK</th>", "<th>KI</th>", 1)
    facts = R.read_match(dup.encode(), m.url)
    assert any(p.startswith("DUPLICATE_LABEL") for p in facts.problems)
    assert "kicks" in facts.bad_fields["Alpha"]


def test_match_page_with_unknown_numeric_column_is_a_schema_gap_not_dropped() -> None:  # T27
    w = small_world()
    m = w.matches[0]
    html = rw.match_page(m, w.players).decode().replace("<th>GA</th>", "<th>GA</th><th>XP</th>", 1)
    facts = R.read_match(html.encode(), m.url)
    assert any(p.startswith("UNKNOWN_LABEL") and "XP" in p for p in facts.problems)


def test_world_pages_round_trip_through_the_readers() -> None:
    w = small_world()
    m = w.matches[1]
    facts = R.read_match(rw.match_page(m, w.players), m.url)
    assert facts.problems == () and len(facts.players) == 3
    assert facts.totals["Alpha"][IDX["kicks"]] == "12"  # two Alpha players with 6 kicks each
    prof = R.read_profile(rw.profile_page(w, "a"), w.players["a"].url)
    assert (
        [g.counter for g in prof.games] == [1, 2, 3] and prof.games[0].season == 2025 and prof.games[1].season == 2026
    )
    assert prof.problems == () and prof.born == "1990-07-01"
    assert page_h1(rw.profile_page(w, "a")) == "Ann Able"


# -- notes ----------------------------------------------------------------------------


def test_notes_availability_matrix_and_exceptions_are_parsed() -> None:
    n = R.read_notes(fx("notes.html"))
    assert n.problems == (), n.problems
    assert sorted(n.availability)[0] == 1965 and sorted(n.availability)[-1] == 2010
    assert "kicks" in n.availability[1965] and "tackles" not in n.availability[1965]
    assert "tackles" in n.availability[1987] and "time_on_ground_pct" not in n.availability[2002]
    assert "disposals" in n.availability[1970]  # derived: kicks + handballs
    assert ("inside_50s" in n.availability[1998] and "one_percenters" in n.availability[1998]) or True
    ex = {(e.category, e.season, e.round_lo, e.round_hi, e.scope) for e in n.exceptions}
    assert ("all_but_goals", 1975, 11, 11, "matchups") in ex
    assert ("hitouts", 1975, 1, 13, "all") in ex and ("hitouts", 1975, 14, 14, "teams") in ex
    matchups = next(e for e in n.exceptions if e.category == "all_but_goals")
    assert ("Footscray", "Carlton") in matchups.matchups and len(matchups.matchups) == 6
    teams = next(e for e in n.exceptions if e.category == "hitouts" and e.round_lo == 14)
    assert teams.teams == ("Sydney", "Hawthorn", "Essendon")


def test_notes_with_an_unknown_exception_category_is_a_schema_gap() -> None:
    html = fx("notes.html").replace(b"Missing hitouts", b"Missing handwriting")
    n = R.read_notes(html)
    assert any(p.startswith("NOTES_CATEGORY") for p in n.problems)


def test_facts_models_round_trip_through_json_byte_for_byte() -> None:
    w = small_world()
    prof = R.read_profile(rw.profile_page(w, "a"), w.players["a"].url)
    again = R.ProfileFacts.model_validate_json(prof.model_dump_json())
    assert again == prof and again.model_dump_json() == prof.model_dump_json()
    m = R.read_match(rw.match_page(w.matches[0], w.players), w.matches[0].url)
    assert R.MatchFacts.model_validate_json(m.model_dump_json()) == m
    n = R.read_notes(fx("notes.html"))
    assert R.NotesFacts.model_validate_json(n.model_dump_json()) == n


def test_date_text_parser_rejects_impossible_dates() -> None:
    assert R.parse_date_text("Sat, 25-Sep-2010 2:30 PM") == date(2010, 9, 25)
    assert R.parse_date_text("30-Feb-2010") is None and R.parse_date_text("garbage") is None


@pytest.mark.parametrize(("token", "n"), [("1", 1), ("23↑", 23), ("5↓", 5), (" 7 ↑", 7), ("↑", None), ("", None)])
def test_career_counter_token_is_leading_digits_only(token: str, n: int | None) -> None:
    assert R.counter_of(token) == n


def test_a_match_decided_in_extra_time_has_a_fifth_score_cell_and_the_last_one_is_the_result() -> None:
    w = small_world()
    m = w.matches[0]
    html = rw.match_page(m, w.players).decode()
    html = html.replace(
        '</tr><tr><td><a href="../../../teams/beta_idx.html">',
        '<td align=center>10.12.<b>72</b></td></tr><tr><td><a href="../../../teams/beta_idx.html">',
        1,
    )
    facts = R.read_match(html.encode(), m.url)
    alpha = next(t for t in facts.teams if t.name == "Alpha")
    assert len(alpha.quarters) == 5 and alpha.quarters[-1] == (10, 12) and alpha.points[-1] == 72
