"""Integration tier — do the PUBLISHED artifacts still agree with the real data?

The unit tier is hermetic: it proves the generators are correct, using tmp fixtures.
That cannot catch the failure mode that has cost this repo the most, which is a
generator that is correct but never ran — leaving a shipped file quietly stale:

  * docs/hall-of-fame-top100.md profile prose froze while its table refreshed
    weekly, because the regeneration step lived in a code path no harness invoked.
  * docs/banner.svg's aria-label announced "R1-R13: MAE 4.020" to screen-reader
    users for seven weeks while the visible pills tracked R1-R20 / MAE 3.958.

Both were invisible to unit tests and to DataSentinel. These checks read the real
data/ tree and the real docs, so they run in the weekly harness (Phase 0b), before
a ~2h cycle builds on an already-inconsistent surface.

Every test here is marked `integration` and is excluded from the pre-commit tier.
"""
import glob
import os
import re
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

pytestmark = pytest.mark.integration

BANNER = os.path.join(_REPO_ROOT, "docs", "banner.svg")
HOF_DOC = os.path.join(_REPO_ROOT, "docs", "hall-of-fame-top100.md")


# ------------------------------------------------------------------ banner


def _banner():
    with open(BANNER, encoding="utf-8") as f:
        return f.read()


def _dash_norm(s):
    return re.sub(r"(?:&#8211;|&#x2013;|–|-)", "-", s)


def test_banner_aria_label_agrees_with_visible_pills():
    """A screen-reader user and a sighted user must be told the same numbers."""
    svg = _banner()
    aria = re.search(
        r"2026 season (R\d+(?:&#8211;|–|-)R\d+): MAE (\d+\.\d+), ([\d.]+)% within 5 disposals",
        svg,
    )
    assert aria, "banner aria-label summary line not found"

    win = re.search(r'<text x="181"[^>]*>(R\d+&#8211;R\d+) &#183; 2026</text>', svg)
    mae = re.search(r'<text x="365"[^>]*>MAE (\d+\.\d+)</text>', svg)
    w5 = re.search(r'<text x="563"[^>]*>([\d.]+)% within 5</text>', svg)
    assert win and mae and w5, "banner pills not found"

    assert _dash_norm(aria.group(1)) == _dash_norm(win.group(1)), (
        f"aria-label window {aria.group(1)} != pill {win.group(1)}"
    )
    assert aria.group(2) == mae.group(1), (
        f"aria-label MAE {aria.group(2)} != pill {mae.group(1)}"
    )
    assert aria.group(3) == w5.group(1), (
        f"aria-label within-5 {aria.group(3)} != pill {w5.group(1)}"
    )


def test_banner_player_file_count_matches_reality():
    """Both places the banner states a player-file count must match the tree."""
    svg = _banner()
    actual = len(
        glob.glob(os.path.join(_REPO_ROOT, "data", "player_data", "*performance_details.csv"))
    )
    assert actual > 0, "no player performance files found — fixture/env problem"

    aria = re.search(r"130 years of AFL data, ([\d,]+) player files", svg)
    band = re.search(r"130 seasons &#183; ([\d,]+) player files", svg)
    assert aria and band, "player-file counts not found in banner"

    assert int(aria.group(1).replace(",", "")) == actual, (
        f"banner aria-label claims {aria.group(1)} player files, tree has {actual:,}"
    )
    assert int(band.group(1).replace(",", "")) == actual, (
        f"banner band claims {band.group(1)} player files, tree has {actual:,}"
    )


# --------------------------------------------------------- README eval figures


def _merged_backtest(year=2026):
    """Summary and per-team frames, merged the way update_eval_surface.sh does.

    Per-team rows supersede by FILE per (year, round): deduping per-team keeps rows
    from an older vintage when the newer run covered fewer teams.
    """
    import pandas as pd

    bt = os.path.join(_REPO_ROOT, "data", "prediction", "backtest")
    need = {"round", "year", "n_players", "mae", "rmse",
            "pct_within_5", "pct_within_10", "bias"}
    frames = []
    for p in sorted(glob.glob(os.path.join(bt, "backtest_summary_*.csv")),
                    key=os.path.getmtime):
        try:
            c = pd.read_csv(p)
        except Exception:
            continue
        if need.issubset(c.columns):
            frames.append(c)
    assert frames, "no usable backtest summary CSVs"
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["year", "round"], keep="last")
    df = df[df["year"] == year]

    tframes = []
    for p in sorted(glob.glob(os.path.join(bt, "backtest_by_team_*.csv")),
                    key=os.path.getmtime):
        try:
            c = pd.read_csv(p)
        except Exception:
            continue
        if {"year", "round", "team", "n", "bias"}.issubset(c.columns):
            c["_mtime"] = os.path.getmtime(p)
            tframes.append(c)
    assert tframes, "no usable backtest by-team CSVs"
    t = pd.concat(tframes, ignore_index=True)
    t = t[t["_mtime"] == t.groupby(["year", "round"])["_mtime"].transform("max")]
    t = t.drop_duplicates(subset=["year", "round", "team"], keep="last")
    t = t[t["year"] == year]
    return df, t


def test_readme_eval_figures_match_the_backtest_csvs():
    """"The numbers" table is a published claim; it must not freeze.

    The aggregate-bias cell sat at its R1-R13 value for seven rounds because its
    regex required an ASCII minus while the row is authored with U+2212 — re.sub
    matched nothing and reported success. Unit tests prove the regeneration works;
    only this proves it was actually RUN against what shipped.
    """
    df, _ = _merged_backtest()
    md = open(README, encoding="utf-8").read()
    n = int(df["n_players"].sum())

    def w(col):
        return float((df[col] * df["n_players"]).sum() / n)

    def cell(label):
        m = re.search(rf"\| {re.escape(label)} \| \*\*\[data\]\*\* ([^|]+)\|", md)
        assert m, f"README row not found: {label}"
        return m.group(1).strip().replace("−", "-").rstrip("%")

    assert cell("Player-round predictions scored").replace(",", "") == str(n)
    assert cell("Mean absolute error (disposals)") == f"{w('mae'):.3f}"
    assert cell("Within 5 disposals") == f"{w('pct_within_5'):.1f}"
    assert cell("Within 10 disposals") == f"{w('pct_within_10'):.1f}"
    assert cell("Aggregate bias") == f"{w('bias'):.3f}"

    r_lo, r_hi = int(df["round"].min()), int(df["round"].max())
    assert f"(all {len(df)} rounds)" in md, f"round count is not {len(df)}"
    assert cell("Backtest window").startswith(f"R{r_lo}"), "backtest window drifted"
    assert f"R{r_hi}" in cell("Backtest window"), "backtest window drifted"


def test_per_team_rows_reconcile_with_the_round_summaries():
    """Detector for a crossed backtest vintage.

    R18 2026 was re-scored from an archived forward CSV that predated the full
    fixture, so a per-team dedupe kept four teams from the superseded run — 92
    phantom player-rounds, one carrying bias -3.26 over n=23, which alone moved the
    published "most under-predicted" team figure from -0.58 to -0.73.
    """
    df, t = _merged_backtest()
    team_n, summary_n = int(t["n"].sum()), int(df["n_players"].sum())
    assert team_n == summary_n, (
        f"per-team rows sum to {team_n:,} but round summaries sum to {summary_n:,} "
        f"(delta {team_n - summary_n:+,}) — two backtest vintages have been crossed"
    )


def test_top30_deviation_rows_reconcile_with_the_round_summaries():
    """Same detector, applied to the OTHER consumer of the same artifacts.

    docs/afl-backtest-2026.md's top-30 player table is built by a second loader
    that had the identical defect: deduping on (player, round) rather than
    superseding by file. The narrower R18 re-score leaked 128 phantom
    player-rounds into it — 7,419 pooled rows against 7,153 actually scored —
    and moved 15 of the 30 published rows. Each surviving (player, round) is one
    scored player-round, so rounds_tracked.sum() is the pooled row count.
    """
    import update_team_analysis as uta

    df, _ = _merged_backtest()
    bt = os.path.join(_REPO_ROOT, "data", "prediction", "backtest")
    dev = uta._load_top30_player_deviation(2026, bt)
    assert not dev.empty, "top-30 deviation loader returned nothing"

    pooled, summary_n = int(dev["rounds_tracked"].sum()), int(df["n_players"].sum())
    assert pooled == summary_n, (
        f"top-30 loader pooled {pooled:,} player-rounds but the round summaries "
        f"account for {summary_n:,} (delta {pooled - summary_n:+,}) — two backtest "
        f"vintages have been crossed, or a round's detail CSV is missing"
    )


# ------------------------------------------------- backtest doc (afl-backtest)

BACKTEST_DOC = os.path.join(_REPO_ROOT, "docs", "afl-backtest-2026.md")


def _pooled_detail(year=2026):
    """Per-player rows, newest FILE per (year, round) — same rule as everywhere."""
    import pandas as pd

    bt = os.path.join(_REPO_ROOT, "data", "prediction", "backtest")
    frames = []
    for p in sorted(glob.glob(os.path.join(bt, "prediction_vs_actual_round_*.csv")),
                    key=os.path.getmtime):
        m = re.search(r"round_(\d+)_(\d{4})_(\d{8}_\d{6})\.csv$", os.path.basename(p))
        if not m or int(m.group(2)) != year:
            continue
        try:
            c = pd.read_csv(p)
        except Exception:
            continue
        if not {"player", "predicted_disposals", "actual_disposals"}.issubset(c.columns):
            continue
        c["_mtime"] = os.path.getmtime(p)
        c["round"] = int(m.group(1))
        frames.append(c)
    assert frames, "no usable per-player backtest CSVs"
    d = pd.concat(frames, ignore_index=True)
    d = d[d["_mtime"] == d.groupby("round")["_mtime"].transform("max")]
    return d.dropna(subset=["predicted_disposals", "actual_disposals"])


def test_backtest_doc_cumulative_block_matches_source():
    """The cumulative summary froze at R1-R13 for seven cycles because nothing
    regenerated it — and its published values could not be reproduced from any
    vintage still on disk. This asserts it tracks the CSVs."""
    df, _ = _merged_backtest()
    md = open(BACKTEST_DOC, encoding="utf-8").read()
    block = re.search(r"<!-- CUMULATIVE-START -->(.*?)<!-- CUMULATIVE-END -->",
                      md, re.DOTALL)
    assert block, "CUMULATIVE markers missing — the block has no regenerator"
    block = block.group(1)

    n = int(df["n_players"].sum())
    mae = float((df["mae"] * df["n_players"]).sum() / n)
    bias = float((df["bias"] * df["n_players"]).sum() / n)
    # RMSE pools in squared space; a linear weighted mean is subtly wrong.
    rmse = float(((df["rmse"] ** 2 * df["n_players"]).sum() / n) ** 0.5)

    assert f"{n:,}" in block, f"player count drifted; expected {n:,}"
    assert f"{mae:.3f}" in block, f"MAE drifted; expected {mae:.3f}"
    assert f"{bias:.3f}" in block, f"bias drifted; expected {bias:.3f}"
    assert f"{rmse:.3f}" in block, f"RMSE drifted; expected {rmse:.3f}"
    assert f"R{int(df['round'].min())}–R{int(df['round'].max())}" in block


def test_backtest_doc_three_way_reconciliation():
    """Summary, per-team and pooled per-player rows must describe one population.

    Any disagreement means two backtest vintages have been blended — the defect
    that put a 14-club headline above an 18-club table on the same page.
    """
    df, t = _merged_backtest()
    d = _pooled_detail()
    summary_n = int(df["n_players"].sum())
    team_n = int(t["n"].sum())
    covered = sorted(d["round"].unique())
    expect = int(df[df["round"].isin(covered)]["n_players"].sum())

    assert team_n == summary_n, (
        f"per-team n {team_n:,} != summary n {summary_n:,}"
    )
    assert len(d) == expect, (
        f"pooled per-player rows {len(d):,} != summary n {expect:,} over rounds "
        f"{covered[0]}-{covered[-1]} — a vintage has been crossed"
    )


def test_backtest_doc_causal_prose_stays_deleted():
    """Removed on purpose: causal claims on associational evidence, with no
    derivation path. If a regenerator ever reinstates them, fail."""
    md = open(BACKTEST_DOC, encoding="utf-8").read()
    for phrase in ("outperforming the model's pre-2026 expectations",
                   "Recurring names worth noting",
                   "consistent with their lower 2026 contested-game volume"):
        assert phrase not in md, f"causal prose reinstated: {phrase!r}"


# ------------------------------------------------------------ agent registry

AGENTS_DIR = os.path.join(_REPO_ROOT, ".claude", "agents")
README = os.path.join(_REPO_ROOT, "README.md")

_NUMBER_WORDS = {
    2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
    8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
}


def _registry():
    """{agent name: declared model} from .claude/agents/*.md frontmatter."""
    out = {}
    for fn in sorted(os.listdir(AGENTS_DIR)):
        if not fn.endswith(".md"):
            continue
        with open(os.path.join(AGENTS_DIR, fn), encoding="utf-8") as f:
            head = f.read(4000)
        m = re.search(r"^model:\s*(\S+)", head, re.MULTILINE)
        out[fn[:-3]] = m.group(1).lower() if m else None
    return out


def _readme_agent_table():
    """{agent name: model} from the README council table, skipping externals."""
    md = open(README, encoding="utf-8").read()
    rows = re.findall(r"^\|\s*\d+\s*\|\s*\*\*([^*]+)\*\*\s*\|\s*([^|]+)\|", md, re.MULTILINE)
    return {
        name.strip(): model.strip().lower()
        for name, model in rows
        if model.strip().lower() != "external"
    }


def test_readme_agent_table_matches_the_registry():
    """The table was hand-typed and drifted: two agents missing, two wrong models.

    Same class as an earlier DataSentinel Haiku/Sonnet mix-up. Generated claims
    about the council must be checkable against `.claude/agents/` frontmatter.
    """
    registry = _registry()
    table = _readme_agent_table()

    missing = sorted(set(registry) - set(table))
    assert not missing, f"agents registered but absent from the README table: {missing}"

    phantom = sorted(set(table) - set(registry))
    assert not phantom, f"README table lists non-registered agents: {phantom}"

    wrong = {
        n: (table[n], registry[n])
        for n in sorted(registry)
        if registry[n] and table[n] != registry[n]
    }
    assert not wrong, f"model tier mismatch (README, registered): {wrong}"


def test_council_size_is_stated_consistently():
    """README said seven-agent twice and eight-agent once, against nine registered."""
    expected = _NUMBER_WORDS[len(_registry()) + 1]  # +1 for external Codex
    md = open(README, encoding="utf-8").read()
    found = set(re.findall(r"([a-z]+)-agent (?:AI )?council", md))
    assert found, "no council-size phrasing found in README"
    assert found == {expected}, (
        f"README states council size as {sorted(found)}; registry implies {expected!r} "
        f"({len(_registry())} registered + 1 external)"
    )


# --------------------------------------------------------------------- HOF


def test_hof_doc_is_consistent_with_the_ranking_csv():
    """The deterministic gate that CLAUDE.md's tag exemption rests on.

    docs/hall-of-fame-top100.md profile prose is exempt from inline [data] tags
    *because* this gate verifies it. If the gate does not hold on the shipped
    file, the exemption is unbacked and untagged wrong numbers are live.
    """
    import pandas as pd

    import update_team_analysis as uta

    if not (os.path.exists(uta.TOP100_CSV) and os.path.exists(uta.TOP100_SCORES_CSV)):
        pytest.fail(
            f"top-100 gate inputs missing ({uta.TOP100_CSV} / {uta.TOP100_SCORES_CSV}) "
            "— the consistency gate cannot run, so the doc's tag exemption is unbacked"
        )

    with open(uta.HALL_OF_FAME_PATH, encoding="utf-8") as f:
        hof_text = f.read()

    hard, _warnings = uta.check_top100_consistency(
        hof_text, pd.read_csv(uta.TOP100_CSV), pd.read_csv(uta.TOP100_SCORES_CSV)
    )
    assert not hard, (
        f"{len(hard)} hard mismatch(es) between the published HOF doc and the "
        f"ranking CSV: " + "; ".join(hard[:5])
    )
