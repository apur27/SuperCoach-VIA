#!/usr/bin/env python3
"""
refresh_readme.py
=================

Master entry point for refreshing every dynamic piece of README.md in one
shot. Designed to run at the very end of `refresh_and_rank.sh`, AFTER the
upstream data has been refreshed and the all-time top 100 has been
recalculated, so this script only has to render outputs from already-fresh
inputs.

What it does (in order):

  Step 1  Era / historical charts  (era_scoring_trends, era_stat_evolution,
                                    scoring_by_decade, era_tackles_clearances)
  Step 2  Top-100 charts           (top10_alltime, top100_position_breakdown)
  Step 3  Current-season team text + 6 charts via update_team_analysis.main()
          (radar, heatmap, 5-year scatter, goals-vs-disposals, form trend +
          rewrites the YEAR-TEAM-ANALYSIS / 5YEAR-TEAM-PROFILES sections of
          README.md so the goals-disposals and form-trend chart markers are
          inserted/refreshed inside that block)
  Step 4  Print a summary — what was updated, current round, current date.

Usage:

    /home/abhi/sourceCode/python/coding/.venv/bin/python refresh_readme.py

The script is also importable as a module:

    from refresh_readme import refresh_all
    refresh_all()                 # do everything
    refresh_all(skip_team=True)   # skip the per-season team analysis step

Design notes
------------
* Idempotent — running twice with the same source data produces byte-identical
  PNGs and the same README content.
* Missing-data resilient — every chart group is wrapped in try/except so a
  failure in (say) the form-trend chart does not abort the era-history charts.
  Failures are logged but the script keeps going and reports them in the
  summary.
* Auto-detects the current year from the most recent player-game data so it
  works season-over-season without code changes.
* No data is mutated — only assets/charts/*.png and the dynamic README
  blocks (between the marker pairs documented in the header).
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = "/home/abhi/git/SuperCoach-VIA"


def _step_static_charts() -> Tuple[List[str], List[str]]:
    """Refresh era/historical charts that depend on era CSVs + match files.
    Returns (paths_written, error_messages)."""
    try:
        from generate_readme_charts import regenerate_static_charts
    except Exception as e:
        return [], [f"could not import generate_readme_charts: {e}"]
    try:
        paths = regenerate_static_charts()
        return paths, []
    except Exception as e:
        return [], [f"regenerate_static_charts: {e}\n{traceback.format_exc()}"]


def _step_top100_charts() -> Tuple[List[str], List[str]]:
    """Refresh top-100 charts (top10 horizontal bar, top100 donut+bar)."""
    try:
        from generate_readme_charts import regenerate_top100_charts
    except Exception as e:
        return [], [f"could not import generate_readme_charts: {e}"]
    try:
        paths = regenerate_top100_charts()
        return paths, []
    except Exception as e:
        return [], [f"regenerate_top100_charts: {e}\n{traceback.format_exc()}"]


def _detect_year() -> Optional[int]:
    """Best-effort auto-detect of the current season — used for the summary
    only. The real detection lives inside update_team_analysis."""
    try:
        from generate_readme_charts import _detect_current_year
        return int(_detect_current_year())
    except Exception:
        return None


def _step_team_analysis() -> Tuple[Dict[str, object], List[str]]:
    """Run the full update_team_analysis pipeline — refreshes the
    YEAR-TEAM-ANALYSIS + 5YEAR-TEAM-PROFILES blocks in README.md, regenerates
    radar/heatmap/scatter/goals-disposals/form-trend charts, and emits the
    GOALS-DISPOSALS-CHART and FORM-TREND-CHART markers inside the team
    analysis block.

    Returns (info_dict, error_messages). info_dict carries the year, max_round,
    n_teams + 5-year window picked up by the script when available.
    """
    info: Dict[str, object] = {}
    try:
        import update_team_analysis as uta
    except Exception as e:
        return info, [f"could not import update_team_analysis: {e}"]
    try:
        # Replicate the upstream main() but capture year/round metadata for
        # the summary. Stays in lock-step with update_team_analysis.main()
        # so changes there flow through.
        games = uta.load_all_player_games()
        year = uta.detect_current_year(games)
        info["year"] = year

        team_game = uta.build_team_game_table(games, year)
        max_round = uta.detect_max_round(team_game)
        n_teams = team_game["team"].nunique()
        info["max_round"] = max_round
        info["n_teams"] = int(n_teams)

        summary = uta.per_team_summary(team_game)
        league = uta.league_averages(team_game)
        summary_with_ranks = uta.add_ranks(summary, uta.SUMMARY_STATS + ["rebound_50s"])
        summary_with_ranks = uta.add_paragraph_ranks(summary_with_ranks)
        summary_with_ranks["form_tag"] = summary_with_ranks.apply(
            lambda r: uta.form_tag(r, summary_with_ranks), axis=1
        )

        top_scorers = uta.per_team_top_disposal_player(games, year)
        body = uta.build_section_body(year, max_round, summary_with_ranks, summary, league, top_scorers)

        with open(uta.README_PATH, "r", encoding="utf-8") as f:
            readme_text = f.read()
        new_readme = uta.replace_section(readme_text, year, body)

        five_year_body, year_window = uta.generate_5year_profiles(year)
        info["window"] = (int(year_window[0]), int(year_window[-1]))
        new_readme = uta.replace_5year_section(new_readme, year, year_window, five_year_body)

        # Finals pathway block — uses the live ladder from matches_<year>.csv
        # paired with the same summary_with_ranks shown in the team analysis
        # section above. If matches data is missing the helper returns an
        # empty body and we leave the README untouched.
        pathway_body, _ladder = uta.generate_finals_pathway(
            year, max_round, summary_with_ranks
        )
        if pathway_body:
            new_readme = uta.replace_finals_pathway_section(new_readme, year, pathway_body)
        # Regenerate the finals-pathway chart alongside the text.
        if not _ladder.empty:
            try:
                uta.generate_finals_pathway_chart(_ladder, year, max_round)
            except Exception as e:
                print(f"  [warn] finals_pathway_chart: {e}", file=sys.stderr)

        # Brownlow Medal vote-proxy block — composite of disposals,
        # clearances, contested possessions, effective-disposals (proxy)
        # and goals, z-scored across all eligible 2026 players. Validated
        # on 2010-2025 historical Brownlow votes during development.
        # The helper also writes the chart to assets/charts/.
        brownlow_body, _brownlow = uta.generate_brownlow_predictor(
            games, year, max_round
        )
        if brownlow_body:
            new_readme = uta.replace_brownlow_predictor_section(
                new_readme, year, brownlow_body
            )

        # Player performance stats explainer block — leaderboards,
        # distributions and per-game correlation drivers for each of the
        # AFL stats commonly tracked for performance prediction. Generates
        # assets/charts/player_stat_leaders_<year>.png alongside the prose.
        try:
            matches_for_stats = uta.load_match_results(year)
        except Exception as e:
            matches_for_stats = pd.DataFrame()
            print(f"  [warn] stat-leaders matches load: {e}", file=sys.stderr)
        stat_body = uta.generate_stat_leaders_section(
            games, matches_for_stats, year, max_round,
        )
        if stat_body:
            new_readme = uta.replace_stat_leaders_section(
                new_readme, year, stat_body,
            )

        if new_readme != readme_text:
            with open(uta.README_PATH, "w", encoding="utf-8") as f:
                f.write(new_readme)

        # Regenerate the season-specific charts (radar, heatmap, scatter,
        # goals-disposals, form-trend). The extended regenerate_team_charts
        # in generate_readme_charts handles all five.
        uta.regenerate_charts(year)

        return info, []
    except Exception as e:
        return info, [f"team analysis: {e}\n{traceback.format_exc()}"]


def refresh_all(skip_static: bool = False, skip_top100: bool = False,
                skip_team: bool = False) -> Dict[str, object]:
    """Run every refresh step. Returns a result dict that mirrors what main()
    prints, suitable for callers that want to react to specific steps."""
    started = time.time()
    static_paths: List[str] = []
    top100_paths: List[str] = []
    team_info: Dict[str, object] = {}
    errors: List[str] = []

    if not skip_static:
        print("=========================================")
        print("[Step 1] Era / historical charts")
        print("=========================================")
        static_paths, errs = _step_static_charts()
        for p in static_paths:
            print(f"  wrote {os.path.relpath(p, REPO_ROOT)}")
        for e in errs:
            print(f"  [error] {e}", file=sys.stderr)
        errors.extend(errs)

    if not skip_top100:
        print("=========================================")
        print("[Step 2] All-time top 100 charts")
        print("=========================================")
        top100_paths, errs = _step_top100_charts()
        for p in top100_paths:
            print(f"  wrote {os.path.relpath(p, REPO_ROOT)}")
        for e in errs:
            print(f"  [error] {e}", file=sys.stderr)
        errors.extend(errs)

    if not skip_team:
        print("=========================================")
        print("[Step 3] Current-season team analysis + charts + README block")
        print("=========================================")
        team_info, errs = _step_team_analysis()
        for e in errs:
            print(f"  [error] {e}", file=sys.stderr)
        errors.extend(errs)

    elapsed = time.time() - started
    print("=========================================")
    print("[Step 4] Summary")
    print("=========================================")
    today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    year = team_info.get("year") or _detect_year()
    max_round = team_info.get("max_round")
    window = team_info.get("window")
    print(f"  date              : {today}")
    if year:
        print(f"  current year      : {year}")
    if max_round is not None:
        print(f"  max round played  : {max_round}")
    if window:
        print(f"  5-year window     : {window[0]}-{window[1]}")
    print(f"  static charts     : {len(static_paths)} written")
    print(f"  top-100 charts    : {len(top100_paths)} written")
    print(f"  team-analysis run : "
          f"{'OK' if not skip_team and 'year' in team_info else ('SKIPPED' if skip_team else 'FAILED')}")
    if errors:
        print(f"  errors            : {len(errors)} (see above)")
    print(f"  elapsed           : {elapsed:.1f}s")

    return {
        "year": year,
        "max_round": max_round,
        "window": window,
        "static_paths": static_paths,
        "top100_paths": top100_paths,
        "team_info": team_info,
        "errors": errors,
        "elapsed_s": elapsed,
        "date": today,
    }


def main() -> int:
    result = refresh_all()
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
