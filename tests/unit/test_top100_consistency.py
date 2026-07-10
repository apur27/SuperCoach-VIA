"""Unit tests for the top-100 profile regeneration + consistency gate.

After the regeneration pass runs, every profiled player who is in the current
ranking (bio_df) must have a heading rank and italic stat-line that match the
ranking — this is the new gate. Players who have dropped out of the ranking
(honourable mention) and freshly-entered players with a placeholder profile are
reported as warnings, not hard failures.
"""
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import update_team_analysis as uta  # noqa: E402


def _bio(rows):
    """rows: list of (serial, name, teams, comment)."""
    return pd.DataFrame(rows, columns=["Serial Number", "Player Name", "Footy Teams", "Comment"])


def _scores(scores):
    return pd.DataFrame({"player": [f"p{i}" for i in range(len(scores))],
                         "all_time_score": scores})


# A doc where the profile order/ranks/stats are STALE relative to bio.
STALE_DOC = """# Top 100

<!-- ALL-TIME-TOP100-START -->
table goes here
<!-- ALL-TIME-TOP100-END -->

## Player profiles - FootyStrategy tactical reads

### #1 Bravo Player — Club B
*100 games · 50 goals · Score: 1.900*

Bravo prose stays intact.

### #2 Alpha Player — Club A
*200 games · 60 goals · Score: 1.800*

Alpha prose stays intact.

## Related

- [back](x.md)
"""


def _bio_fixture():
    # Ranking now: Alpha #1 (climbed, stats grew), Bravo #2. Charlie #3 is NEW
    # (no profile yet). Bravo's old stats were stale.
    bio = _bio([
        (1, "Alpha Player", "Club A",
         "Alpha Player played and 210 games. He recorded 999 total disposals and 65 goals."),
        (2, "Bravo Player", "Club B",
         "Bravo Player played and 105 games. He recorded 500 total disposals and 52 goals."),
        (3, "Charlie Player", "Club C",
         "Charlie Player played and 300 games. He recorded 800 total disposals and 70 goals."),
    ])
    scores = _scores([2.500, 2.400, 2.300])
    return bio, scores


def test_stale_doc_fails_gate_before_regen():
    bio, scores = _bio_fixture()
    hard, _warn = uta.check_top100_consistency(STALE_DOC, bio, scores)
    assert hard, "stale doc should report hard mismatches before regeneration"


def test_profile_rank_matches_table():
    bio, scores = _bio_fixture()
    new_text, _warn = uta.regenerate_top100_profiles(STALE_DOC, bio, scores)
    # After regen, Alpha must be #1 and Bravo #2 (rank order fixed).
    heads = re.findall(r"^### #(\d+) (.+?) — ", new_text, re.M)
    rank_by_name = {name: int(r) for r, name in heads}
    assert rank_by_name["Alpha Player"] == 1
    assert rank_by_name["Bravo Player"] == 2
    # New entrant Charlie gets a placeholder profile at #3.
    assert rank_by_name["Charlie Player"] == 3
    assert "FOOTYSTRATEGY INSERT" in new_text


def test_profile_stats_match_csv():
    bio, scores = _bio_fixture()
    new_text, _warn = uta.regenerate_top100_profiles(STALE_DOC, bio, scores)
    hard, _warn2 = uta.check_top100_consistency(new_text, bio, scores)
    assert hard == [], f"regenerated doc must pass the gate, got: {hard}"
    # Alpha's stat line reflects the current ranking numbers, not the stale ones.
    assert "210 games · 65 goals · 999 disposals · Score: 2.500" in new_text
    # Prose is preserved untouched.
    assert "Alpha prose stays intact." in new_text
    assert "Bravo prose stays intact." in new_text


def test_dropped_player_becomes_honourable_mention_warning():
    """A profiled player no longer in the ranking is moved, not deleted, and warned."""
    bio, scores = _bio_fixture()
    doc = STALE_DOC.replace("### #2 Alpha Player — Club A",
                            "### #2 Zulu Player — Club Z")
    doc = doc.replace("Alpha prose stays intact.", "Zulu prose stays intact.")
    new_text, warn = uta.regenerate_top100_profiles(doc, bio, scores)
    assert "Honourable Mention" in new_text
    assert "Zulu prose stays intact." in new_text  # prose preserved
    assert any("Zulu" in w for w in warn)
    # Gate does not hard-fail on the dropped player.
    hard, _ = uta.check_top100_consistency(new_text, bio, scores)
    assert hard == []
