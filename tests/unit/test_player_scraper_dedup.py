"""Regression test for the drawn-Grand-Final dedup collapse (deliverable 1).

Commit 58f1a4f20 and scrapers/player_scraper.py deduped appended player rows on
(team, year, round, opponent) with keep='last'. The 2010 drawn Grand Final and
its replay share (team=Collingwood, year=2010, round='GF', opponent='St Kilda'),
so that key collapsed two distinct games into one and silently deleted the draw's
stat line.

`dedup_player_performance` must:
  * KEEP both the draw (games_played=35, result 'D') and the replay
    (games_played=36, result 'W') because games_played authoritatively
    distinguishes the two games; and
  * still REMOVE a genuine exact re-scrape of the same game (same
    team/year/round/opponent AND same games_played counter).
"""
import pandas as pd

from scrapers.player_scraper import dedup_player_performance


def _row(gp, result, disposals):
    return {
        "team": "Collingwood",
        "year": 2010,
        "games_played": gp,
        "opponent": "St Kilda",
        "round": "GF",
        "result": result,
        "disposals": disposals,
    }


def test_drawn_gf_and_replay_both_survive():
    df = pd.DataFrame([_row(35, "D", 19), _row(36, "W", 25)])
    out = dedup_player_performance(df)
    assert len(out) == 2, "draw and replay are distinct games; both must survive"
    assert set(out["games_played"]) == {35, 36}
    assert set(out["result"]) == {"D", "W"}


def test_true_exact_duplicate_is_removed():
    # Same game scraped twice: identical team/year/round/opponent AND counter.
    df = pd.DataFrame([_row(35, "D", 19), _row(35, "D", 19)])
    out = dedup_player_performance(df)
    assert len(out) == 1, "an exact re-scrape of one game must collapse to one row"
    assert int(out.iloc[0]["games_played"]) == 35


def test_drawn_pair_plus_true_duplicate_together():
    # The replay is scraped twice (true dup) alongside the draw: end with 2 rows.
    df = pd.DataFrame([_row(35, "D", 19), _row(36, "W", 25), _row(36, "W", 25)])
    out = dedup_player_performance(df)
    assert len(out) == 2
    assert set(out["games_played"]) == {35, 36}


def test_arrow_annotated_counter_is_normalised():
    # games_played can carry ↑/↓ debut/milestone arrows; they must not create a
    # phantom distinct game nor block dedup of the same counter.
    a = _row("35", "D", 19)
    b = _row("↑35", "D", 19)  # same game, arrow-annotated re-scrape
    out = dedup_player_performance(pd.DataFrame([a, b]))
    assert len(out) == 1, "arrow-annotated counter is the same game, must collapse"
