"""_load_top30_player_deviation must pick the newest run by FULL timestamp.

Backtest artifacts are named `..._<YEAR>_<YYYYMMDD>_<HHMMSS>.csv`. The dedup key
took `base.rsplit("_", 1)[-1]`, which is the HHMMSS field alone — the DATE was
discarded. Sorting on time-of-day means a re-run that starts earlier in the clock
day than the run it supersedes silently loses, and the STALE vintage is published.

Today all ten multi-vintage 2026 rounds happen to resolve correctly, purely
because the authoritative runs also had the later wall-clock times. That is luck,
not correctness, and it is the same failure class as the tainted-provenance
incident that started this cycle: a selector that looks right because its inputs
have so far been kind to it.

Hermetic: synthetic CSVs under tmp_path, no real data files touched.
"""
import os
import sys

import pandas as pd
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import update_team_analysis as uta

COLS = ["player", "team", "round", "year",
        "predicted_disposals", "actual_disposals", "error", "abs_error"]


def _vintage(bt_dir, rnd, ts, actual, year=2026, player="Smith John"):
    pd.DataFrame(
        [[player, "Test Team", rnd, year, 20.0, actual, 20.0 - actual,
          abs(20.0 - actual)]],
        columns=COLS,
    ).to_csv(bt_dir / f"prediction_vs_actual_round_{rnd}_{year}_{ts}.csv", index=False)


def test_later_date_wins_even_with_an_earlier_time_of_day(tmp_path):
    """THE BUG: the superseding run started earlier in the clock day.

    05-18 at 19:18 is authoritative; 05-25 at 18:21 supersedes it by date but
    loses on HHMMSS. A date-blind key publishes the stale actual (10.0).
    """
    _vintage(tmp_path, 1, "20260518_191837", actual=10.0)   # older date, later time
    _vintage(tmp_path, 1, "20260525_182141", actual=99.0)   # newer date, earlier time

    df = uta._load_top30_player_deviation(2026, str(tmp_path))

    assert not df.empty
    assert df.iloc[0]["avg_actual"] == 99.0, (
        "selected the stale vintage — the dedup key is comparing time-of-day, "
        "not the full YYYYMMDD_HHMMSS timestamp"
    )


def test_later_time_wins_within_the_same_date(tmp_path):
    """Control: the fix must not break same-day ordering."""
    _vintage(tmp_path, 2, "20260518_101500", actual=10.0)
    _vintage(tmp_path, 2, "20260518_191837", actual=42.0)

    df = uta._load_top30_player_deviation(2026, str(tmp_path))
    assert df.iloc[0]["avg_actual"] == 42.0


def test_current_repo_vintages_still_resolve_the_same(tmp_path):
    """The real files resolve correctly today by luck; the fix must be a no-op
    for them, so this change cannot move a published figure."""
    _vintage(tmp_path, 1, "20260430_142823", actual=1.0)
    _vintage(tmp_path, 1, "20260511_191837", actual=2.0)    # authoritative
    df = uta._load_top30_player_deviation(2026, str(tmp_path))
    assert df.iloc[0]["avg_actual"] == 2.0


def test_malformed_timestamp_does_not_crash_the_loader(tmp_path):
    """A file that does not carry a parseable timestamp must be skipped, not
    allowed to throw and take the whole backtest section down."""
    _vintage(tmp_path, 3, "20260518_191837", actual=7.0)
    pd.DataFrame(
        [["Smith John", "Test Team", 3, 2026, 20.0, 5.0, 15.0, 15.0]], columns=COLS
    ).to_csv(tmp_path / "prediction_vs_actual_round_3_2026_NOTATIMESTAMP.csv",
             index=False)

    df = uta._load_top30_player_deviation(2026, str(tmp_path))
    assert not df.empty
    assert df.iloc[0]["avg_actual"] == 7.0


# ---------------------------------------------------------------------------
# Second defect, same function: a per-(player, round) dedupe does not supersede
# a ROUND. When a re-score covers FEWER players than the run it replaces, the
# missing players survive from the stale vintage and the published table becomes
# a hybrid of two runs. This is the identical failure class as the by_team
# dedupe fixed in scripts/update_eval_surface.sh (commit ce6ff6163).
# ---------------------------------------------------------------------------


def _multi(bt_dir, rnd, ts, rows, year=2026):
    """Write one vintage file. `rows` is a list of (player, predicted, actual).

    `actual` may be None to represent an unscored forward prediction — a row the
    round summary does not count in n_players.
    """
    recs = []
    for player, pred, actual in rows:
        if actual is None:
            err = abs_err = None
        else:
            err, abs_err = pred - actual, abs(pred - actual)
        recs.append([player, "Test Team", rnd, year, pred, actual, err, abs_err])
    pd.DataFrame(recs, columns=COLS).to_csv(
        bt_dir / f"prediction_vs_actual_round_{rnd}_{year}_{ts}.csv", index=False)


def test_a_narrower_rescore_supersedes_the_whole_round(tmp_path):
    """THE BUG: R18 2026 was re-scored with a vintage covering four fewer clubs.

    Deduping on (player, round) only replaces players the NEW run also has. Any
    player it dropped survives from the OLD file, so the round is served from two
    vintages at once. Superseding must be by FILE per (year, round).
    """
    _multi(tmp_path, 18, "20260707_154033",
           [("Alpha A", 20.0, 20.0), ("Beta B", 20.0, 30.0), ("Gamma G", 20.0, 40.0)])
    _multi(tmp_path, 18, "20260710_214217",      # narrower re-score: Alpha only
           [("Alpha A", 20.0, 25.0)])

    df = uta._load_top30_player_deviation(2026, str(tmp_path))

    assert set(df["player"]) == {"Alpha A"}, (
        "players absent from the superseding vintage survived from the stale one — "
        f"got {sorted(df['player'])}; the round is being served from two vintages"
    )
    assert float(df.iloc[0]["avg_actual"]) == 25.0


def test_pooled_rows_reconcile_with_the_round_summary_n_players(tmp_path):
    """Reconciliation invariant: scored player-rounds == summed summary n_players.

    Each surviving (player, round) is one scored player-round, so the pooled row
    count is `rounds_tracked.sum()`. It must equal what the round summaries say
    was scored. A mismatch means either a crossed vintage (too many) or a lost
    file (too few) — the same detector that caught the per-team hybrid.
    """
    # Round 1: 3 players, re-scored down to 2. Summary for the winning run: 2.
    _multi(tmp_path, 1, "20260430_142823",
           [("Alpha A", 20.0, 20.0), ("Beta B", 20.0, 21.0), ("Gamma G", 20.0, 22.0)])
    _multi(tmp_path, 1, "20260511_191837",
           [("Alpha A", 20.0, 23.0), ("Beta B", 20.0, 24.0)])
    # Round 2: single vintage, 2 rows but one is an unscored forward prediction.
    # The round summary counts 1.
    _multi(tmp_path, 2, "20260511_191837",
           [("Alpha A", 20.0, 25.0), ("Delta D", 40.0, None)])

    summary_n_players = 2 + 1

    df = uta._load_top30_player_deviation(2026, str(tmp_path))
    pooled = int(df["rounds_tracked"].sum())

    assert pooled == summary_n_players, (
        f"loader pooled {pooled} player-rounds but the round summaries account "
        f"for {summary_n_players} (delta {pooled - summary_n_players:+d})"
    )


def test_unscored_rows_do_not_contaminate_the_averages(tmp_path):
    """A row with no actual must not enter avg_predicted or rounds_tracked.

    `error` is null on such a row, so avg_error already skips it — but
    avg_predicted and rounds_tracked did not, which computed the two halves of
    the published comparison over different populations.
    """
    _multi(tmp_path, 1, "20260511_191837", [("Alpha A", 20.0, 20.0)])
    _multi(tmp_path, 2, "20260518_144551", [("Alpha A", 40.0, None)])

    df = uta._load_top30_player_deviation(2026, str(tmp_path))

    assert len(df) == 1
    assert int(df.iloc[0]["rounds_tracked"]) == 1, "counted an unscored round"
    assert float(df.iloc[0]["avg_predicted"]) == 20.0, (
        "avg_predicted absorbed a prediction that was never scored")
    assert float(df.iloc[0]["avg_actual"]) == 20.0


# ---------------------------------------------------------------------------
# The "What to look for" callout carried a hardcoded "Round 1 (MAE ~4.9)". The
# real R1 2026 MAE is 4.83, and a literal cannot self-correct when rounds are
# added or re-scored. It is now derived from the summary table it sits under.
# ---------------------------------------------------------------------------


def test_opening_round_mae_callout_is_derived_not_hardcoded(tmp_path, monkeypatch):
    """Hermetic: REPO_ROOT is redirected so the real assets/ chart is not rewritten."""
    bt = tmp_path / "data" / "prediction" / "backtest"
    bt.mkdir(parents=True)
    pd.DataFrame(
        [[3, 2026, 100, 9.99, 12.0, 8.0, -0.1, 40.0, 70.0],
         [4, 2026, 100, 3.00, 4.0, 2.0, 0.1, 80.0, 96.0]],
        columns=["round", "year", "n_players", "mae", "rmse",
                 "median_abs_error", "bias", "pct_within_5", "pct_within_10"],
    ).to_csv(bt / "backtest_summary_20260511_191837.csv", index=False)
    monkeypatch.setattr(uta, "REPO_ROOT", str(tmp_path))

    md = uta.generate_backtest_section(2026)

    # The figure now carries a **[data]** tag (DataSentinel flagged the callout as
    # an untagged restatement of a real number), so match around the tag rather
    # than pinning the exact rendering.
    assert "A spike in Round 3 (MAE 9.99 **[data]**)" in md, (
        "the opening-round callout is not derived from the summary table")
    assert "~4.9" not in md, "a hardcoded MAE literal survives in the callout"


def test_pre_registered_threshold_is_evaluated_and_published(tmp_path, monkeypatch):
    """The page pre-registers a "concerning round" bar and must apply it to itself.

    It defined a round as concerning when it carries more than five outright misses
    (error > 10 disposals), then never evaluated the rule against its own table.
    User decision 2026-07-27: publish the breach rather than re-calibrate the bar,
    because a pre-registered threshold quietly moved when breached is not a threshold.
    Outright misses per round are derived from the published within-10 column:
    n * (100 - pct_within_10) / 100.
    """
    bt = tmp_path / "data" / "prediction" / "backtest"
    bt.mkdir(parents=True)
    # R3: 100 * (100-80)/100 = 20 misses -> concerning.
    # R4: 100 * (100-99)/100 =  1 miss   -> not concerning.
    pd.DataFrame(
        [[3, 2026, 100, 4.0, 5.0, 3.0, -0.1, 70.0, 80.0],
         [4, 2026, 100, 3.0, 4.0, 2.0, 0.1, 90.0, 99.0]],
        columns=["round", "year", "n_players", "mae", "rmse",
                 "median_abs_error", "bias", "pct_within_5", "pct_within_10"],
    ).to_csv(bt / "backtest_summary_20260101_000000.csv", index=False)

    monkeypatch.setattr(uta, "REPO_ROOT", str(tmp_path))
    md = uta.generate_backtest_section(2026)

    assert "1 of 2 rounds" in md, f"threshold not evaluated against the table: {md[:600]}"
    assert "**[data]**" in md
    assert "re-calibrate" in md, "the decision not to move the bar is not disclosed"
