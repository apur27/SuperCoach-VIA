"""National/rookie (Wikipedia) and DraftGuru pure draft adapters."""

from __future__ import annotations

import pytest

from supercoach_via.domain.schemas import TABLES, CheckOutcome
from supercoach_via.ingest import drafts

WIKI = """<html><body>
<h2>Mid-season rookie draft</h2>
<table class="wikitable"><tr><th>Pick</th><th>Player</th><th>Club</th><th>Recruited from</th></tr>
<tr><td>1</td><td>Mid Guy</td><td>Carlton</td><td>Werribee</td></tr></table>
<h2>National draft</h2>
<table class="wikitable"><tr><th>Round</th><th>Pick</th><th>Player</th><th>Club</th><th>Recruited from</th></tr>
<tr><td rowspan="2">1</td><td>1</td><td>Sam Lalor[1]</td><td>Richmond</td><td>Murray Bushrangers</td></tr>
<tr><td>2</td><td>Josh Smillie</td><td>Richmond</td><td>Eastern Ranges</td></tr>
<tr><td>2</td><td>20</td><td>Jed Walter</td><td>Gold Coast</td><td>Gold Coast Suns Academy</td></tr>
<tr><td colspan="5">Pass</td></tr></table>
<h3>Rookie draft</h3>
<table class="wikitable"><tr><th>Pick</th><th>Player</th><th>Club</th><th>Recruited from</th></tr>
<tr><td>1</td><td>Rookie One</td><td>West Coast</td><td>Claremont</td></tr></table>
</body></html>"""

GURU_HEADERS = ["Pick", "Draft", "#", "Club", "Signing", "Player", "Age", "Height", "Weight",
                "Original Club", "Grade", "Games", "Goals"]  # fmt: skip


def _guru(rows: list[list[str]]) -> str:
    th = "".join(f"<th>{h}</th>" for h in GURU_HEADERS)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<html><body><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></body></html>"


def test_wikipedia_national_and_rookie_tables_with_rowspan() -> None:
    res = drafts.parse_wikipedia_draft(WIKI, season=2023)
    assert res.outcome is CheckOutcome.PASS, res.issues
    national = [e for e in res.events if e.event_type == "national"]
    assert [(e.draft_round, e.pick, e.player_name) for e in national] == [
        (1, 1, "Sam Lalor"),
        (1, 2, "Josh Smillie"),
        (2, 20, "Jed Walter"),
    ]
    types = {e.event_type for e in res.events}
    assert types == {"national", "rookie_mid_season", "rookie_end_season"}
    row = national[0].to_row(source_url=drafts.wikipedia_url(2023), source_sha256="a" * 64)
    assert set(row) == set(TABLES["draft_events"].column_names)
    assert row["player_id"] is None  # identity resolution is not the adapter's job


def test_wikipedia_without_national_section_fails_not_empty_success() -> None:
    res = drafts.parse_wikipedia_draft("<html><h2>History</h2><p>none</p></html>", season=2023)
    assert res.outcome is CheckOutcome.FAIL and res.events == []


def test_draftguru_uses_hash_column_and_keeps_blank_games_unknown() -> None:
    html = _guru(
        [
            ["", "National", "5", "Hawthorn", "", "Lance Franklin", "17", "", "", "Perth", "A", "354 (22)", "1066"],
            ["", "Rookie", "1", "Hawthorn", "", "Some Rookie", "19", "", "", "Box Hill", "-", "-", ""],
        ]
    )
    res = drafts.parse_draftguru(html, season=2004)
    assert res.outcome is CheckOutcome.PASS, res.issues
    lance, rookie = res.events
    assert (lance.pick, lance.event_type, lance.grade, lance.games) == (5, "national", "A", 354)
    assert rookie.event_type == "rookie" and rookie.games is None and rookie.grade is None


def test_draftguru_header_drift_fails() -> None:
    assert drafts.parse_draftguru("<table><tr><th>Name</th></tr></table>", season=2004).outcome is CheckOutcome.FAIL


def test_urls() -> None:
    assert drafts.wikipedia_url(2023) == "https://en.wikipedia.org/wiki/2023_AFL_draft"
    assert drafts.draftguru_url(2004) == "https://www.draftguru.com.au/years/2004"
    with pytest.raises(ValueError):
        drafts.draftguru_url(1800)
