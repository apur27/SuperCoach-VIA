"""Versioned rules and identity overrides: strict loading, evidence required (DESIGN section 7-8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from supercoach_via.reconciliation.rules import RulesError, load_rules, parse_rules

BASE = "version = 1\n"
HEADER = "local_player_id,source_url,evidence_sha256,evidence_locator,reason,reviewer\n"


def test_shipped_rules_load_and_are_hashed() -> None:
    r = load_rules()
    assert r.version == 1 and r.notes_labels == {"I5": "IF", "OP": "1%"}
    assert len(r.rules_sha256) == 64 and len(r.overrides_sha256) == 64


def test_unknown_sections_and_versions_are_refused() -> None:
    with pytest.raises(RulesError, match="unknown sections"):
        parse_rules(BASE + "[surprise]\nx = 1\n")
    with pytest.raises(RulesError, match="version"):
        parse_rules("version = 2\n")


def test_one_sided_rule_needs_an_evidence_locator_and_a_reason() -> None:
    ok = (
        BASE
        + '[[one_sided_rule]]\nrule_id = "R-X"\nfield = "brownlow_votes"\nfirst_season = 1931\nlast_season = 1934\n'
        + 'printed_on = "profile"\nevidence_locator = "match 1933 page, BR column"\nreason = "profiles print per game"\n'
    )
    r = parse_rules(ok)
    assert r.one_sided[0].rule_id == "R-X" and r.one_sided[0].printed_on == "profile"
    with pytest.raises(RulesError, match="evidence locator"):
        parse_rules(ok.replace('evidence_locator = "match 1933 page, BR column"', 'evidence_locator = " "'))
    with pytest.raises(RulesError, match="bad field"):
        parse_rules(ok.replace("brownlow_votes", "vibes"))


BROWNLOW = (
    '[brownlow]\nevidence_url = "https://afltables.com/afl/brownlow/brownlow_idx.html"\n'
    f'evidence_sha256 = "{"ab" * 32}"\nevidence_locator = "footer row of the winners table"\n'
    'reason = "award seasons and votes per game are stated by the source"\n'
)


def test_brownlow_section_pins_an_evidence_url_digest_locator_and_reason_and_nothing_typed_in() -> None:  # A9
    r = parse_rules(BASE + BROWNLOW)
    assert r.brownlow_ref is not None and r.brownlow_ref.sha256 == "ab" * 32 and r.brownlow is None
    for missing in ("evidence_url", "evidence_sha256", "evidence_locator", "reason"):
        text = "\n".join(x for x in BROWNLOW.splitlines() if not x.startswith(missing))
        with pytest.raises(RulesError, match="missing"):
            parse_rules(BASE + text + "\n")
    with pytest.raises(RulesError, match="parsed from the evidence"):
        parse_rules(BASE + BROWNLOW + "no_award_seasons = [1942]\n")
    with pytest.raises(RulesError, match="sha256"):
        parse_rules(BASE + BROWNLOW.replace("ab" * 32, "xyz"))
    with pytest.raises(RulesError, match="evidence_url"):
        parse_rules(BASE + BROWNLOW.replace("brownlow_idx.html", "other.html"))


def test_attaching_evidence_verifies_the_digest_and_derives_the_award_facts() -> None:  # A9
    from tests.scvia.unit.test_reconciliation_evidence import brownlow_page

    body = brownlow_page()
    import hashlib

    pinned = BROWNLOW.replace("ab" * 32, hashlib.sha256(body).hexdigest())
    r = parse_rules(BASE + pinned).with_evidence(body)
    assert r.no_award_seasons == frozenset({1942, 1943, 1944, 1945})
    assert r.br_award_total(1932) == 6 and r.br_award_total(1976) == 12 and r.br_award_total(1943) is None
    with pytest.raises(RulesError, match="does not hash"):
        parse_rules(BASE + pinned).with_evidence(body + b" ")
    with pytest.raises(RulesError, match="no evidence"):
        parse_rules(BASE).with_evidence(body)


def test_without_attached_evidence_there_are_no_award_facts() -> None:
    r = parse_rules(BASE + BROWNLOW)
    assert r.no_award_seasons == frozenset() and r.br_award_total(2026) is None


def test_overrides_need_every_column_and_cannot_hide_in_a_free_text_row() -> None:
    row = "legacy:a_b_01011990,https://afltables.com/afl/stats/players/A/A_B.html,abc,page A row 3,same DOB and games,reviewer\n"
    r = parse_rules(BASE, HEADER + row)
    assert r.overrides[0].source_url.endswith("A_B.html")
    with pytest.raises(RulesError, match="every override"):
        parse_rules(BASE, HEADER + row.replace("same DOB and games", ""))
    with pytest.raises(RulesError, match="header"):
        parse_rules(BASE, "a,b\n1,2\n")


def test_rule_or_override_edits_change_the_hashes() -> None:
    a = parse_rules(BASE, HEADER)
    assert parse_rules(BASE + "# note\n", HEADER).rules_sha256 != a.rules_sha256
    row = "legacy:a,https://afltables.com/afl/stats/players/A/A.html,e,l,r,v\n"
    assert parse_rules(BASE, HEADER + row).overrides_sha256 != a.overrides_sha256


ALIAS = (
    '[[notes_club_alias]]\nseason = 1975\nname = "Sydney"\nclub = "South Melbourne"\n'
    'evidence_locator = "notes page row 1975 R14; 1975 season page lists South Melbourne"\n'
    'reason = "the notes table names the club by its current name"\n'
)


def test_club_alias_maps_notes_lineage_names_per_season() -> None:  # S-08 / A10
    r = parse_rules(BASE + ALIAS)
    assert r.club_alias(1975, "Sydney") == "South Melbourne" and r.club_alias(1976, "Sydney") == "Sydney"
    assert r.notes_club_aliases[0].evidence_locator.startswith("notes page") and r.notes_club_aliases[0].reason


def test_a_club_alias_without_an_evidence_locator_and_a_reason_is_refused() -> None:  # A10
    for key in ("evidence_locator", "reason"):
        text = "\n".join(x for x in ALIAS.splitlines() if not x.startswith(key))
        with pytest.raises(RulesError, match="missing"):
            parse_rules(BASE + text + "\n")
    with pytest.raises(RulesError, match="evidence locator"):
        parse_rules(BASE + ALIAS.replace("notes page row 1975 R14; 1975 season page lists South Melbourne", " "))


def test_the_shipped_rules_pin_the_captured_brownlow_page_and_map_the_1975_sydney_exception() -> None:
    import hashlib
    from pathlib import Path

    r = load_rules()
    fx = Path(__file__).resolve().parents[1] / "fixtures" / "reconciliation" / "brownlow_idx.html"
    assert r.brownlow_ref is not None and r.brownlow_ref.sha256 == hashlib.sha256(fx.read_bytes()).hexdigest()
    assert r.club_alias(1975, "Sydney") == "South Melbourne"
    assert r.with_evidence(fx.read_bytes()).br_award_total(1977) == 12


def test_the_per_match_sum_rule_is_gone_in_favour_of_the_evidence_derived_award_total() -> None:  # A8
    with pytest.raises(RulesError, match="unknown sections"):
        parse_rules(BASE + '[[sum_rule]]\nrule_id = "R-BR-SIX-PER-MATCH"\n')
    assert "sum_rule" not in (Path(__file__).resolve().parents[3] / "config" / "reconciliation_rules.toml").read_text()
