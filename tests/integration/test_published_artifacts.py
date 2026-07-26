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
