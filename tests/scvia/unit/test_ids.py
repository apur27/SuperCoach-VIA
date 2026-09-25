"""Identity registry: player slugs, birth dates, names, clubs/lineage, venues (D05-D08)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from supercoach_via.domain import ids
from supercoach_via.domain.schemas import BirthDateQuality, is_safe_id

REPO_CONFIG = Path(__file__).resolve().parents[3] / "config"


class TestPlayerIds:
    def test_slug_and_id(self) -> None:
        p = Path("data/player_data/cockatoo-collins_che_05031975_performance_details.csv")
        slug = ids.slug_from_filename(p.name)
        assert slug == "cockatoo-collins_che_05031975"
        assert ids.player_id_for_slug(slug) == "legacy:cockatoo-collins_che_05031975"
        assert ids.dob_token(slug) == "05031975"

    def test_same_name_players_stay_distinct(self) -> None:
        # D05: identical display names, different DOB tokens -> different IDs
        a = ids.player_id_for_slug("lynch_tom_31101990")
        b = ids.player_id_for_slug("lynch_tom_09101992")
        assert a != b

    def test_unsafe_slug_rejected(self) -> None:
        for bad in ["../etc_x_01011990", "a/b_c_01011990", ""]:
            with pytest.raises(ValueError):
                ids.player_id_for_slug(bad)

    def test_personal_details_filename_not_a_performance_slug(self) -> None:
        with pytest.raises(ValueError):
            ids.slug_from_filename("x_y_01011990_personal_details.csv")


class TestBirthDates:
    def test_source_birth_date(self) -> None:
        d, q, issue = ids.resolve_birth_date("06-03-1947", "bartlett_kevin_06031947")
        assert (d, q, issue) == (date(1947, 3, 6), BirthDateQuality.SOURCE, None)

    def test_fallback_to_filename_token(self) -> None:
        d, q, issue = ids.resolve_birth_date("", "bartlett_kevin_06031947")
        assert d == date(1947, 3, 6) and q is BirthDateQuality.LEGACY_FILENAME and issue == "birth_date_from_filename"
        d, q, _ = ids.resolve_birth_date("not a date", "bartlett_kevin_06031947")
        assert q is BirthDateQuality.LEGACY_FILENAME

    def test_fabricated_default_is_not_a_birth_date(self) -> None:
        # legacy parser substituted 01-01-1900 on failure: never present it as a DOB
        d, q, issue = ids.resolve_birth_date("01-01-1900", "casey_dick_01011900")
        assert d is None and q is BirthDateQuality.UNKNOWN and issue == "legacy_default_birth_date"

    def test_conflict_between_file_and_token(self) -> None:
        d, q, issue = ids.resolve_birth_date("22-10-2001", "steele_roan_19092002")
        assert (
            d == date(2001, 10, 22) and q is BirthDateQuality.CONFLICTING and issue == "birth_date_conflicts_filename"
        )

    def test_nothing_usable(self) -> None:
        assert ids.resolve_birth_date(None, "x_y_zz")[:2] == (None, BirthDateQuality.UNKNOWN)


class TestMeasures:
    @pytest.mark.parametrize(
        ("raw", "expected"), [("183", 183.0), ("0", None), ("-1", None), ("", None), ("abc", None)]
    )
    def test_non_positive_sentinels_are_null(self, raw: str, expected: float | None) -> None:
        assert ids.parse_measure(raw) == expected


class TestNames:
    def test_normalize(self) -> None:
        assert ids.normalize_name("  David  Rhys-Jones ") == ids.normalize_name("david rhys jones")
        assert ids.normalize_name("Jordan De Goey") == ids.normalize_name("Jordan de Goey")
        assert ids.normalize_name("Shaun O'Brien") == "shaun obrien"
        assert ids.normalize_name("Zoë Ñuñez") == "zoe nunez"

    @given(st.text(max_size=30))
    def test_normalize_idempotent(self, s: str) -> None:
        once = ids.normalize_name(s)
        assert ids.normalize_name(once) == once

    def test_registry_constants_are_safe(self) -> None:
        for dup in ids.KNOWN_DUPLICATES:
            assert is_safe_id(ids.player_id_for_slug(dup.duplicate_slug))
            assert is_safe_id(ids.player_id_for_slug(dup.canonical_slug))
            assert dup.source_url.startswith("https://afltables.com/")
        slugs = {d.duplicate_slug for d in ids.KNOWN_DUPLICATES}
        assert slugs == {"green_william_08092005", "steele_roan_19092002"}
        assert len(ids.VERIFIED_SOURCE_ALIASES) == 7
        # multiword names: the verified source name is the alias for the legacy slug
        by_slug = {a.slug: a.source_name for a in ids.VERIFIED_SOURCE_ALIASES}
        assert by_slug["wyk_alex_01072004"] == "Alex Van Wyk"
        assert by_slug["goey_jordan_15031996"] == "Jordan de Goey"


def _clubs(tmp_path: Path) -> ids.ClubRegistry:
    p = tmp_path / "team_aliases.csv"
    p.write_text(
        "alias,club_id,club_name,lineage_id,is_primary,valid_from_season,valid_to_season,note\n"
        "South Melbourne,south_melbourne,South Melbourne,swans,true,1897,1981,\n"
        "Sydney,sydney,Sydney,swans,true,1982,,\n"
        "Brisbane,brisbane_bears,Brisbane Bears,brisbane,false,1987,1995,\n"
        "Brisbane Bears,brisbane_bears,Brisbane Bears,brisbane,true,1987,1996,\n"
        "Brisbane,brisbane_lions,Brisbane Lions,brisbane,false,1997,,\n"
        "Brisbane Lions,brisbane_lions,Brisbane Lions,brisbane,true,1997,,\n"
        "Fitzroy,fitzroy,Fitzroy,fitzroy,true,1897,1996,\n",
        encoding="utf-8",
    )
    return ids.ClubRegistry.from_csv(p)


class TestClubs:
    def test_entities_distinct_lineage_shared(self, tmp_path: Path) -> None:
        # D08: South Melbourne and Sydney are separate entities sharing a lineage
        reg = _clubs(tmp_path)
        assert reg.resolve("South Melbourne", 1950) == "south_melbourne"
        assert reg.resolve("Sydney", 1990) == "sydney"
        assert reg.lineage("south_melbourne") == reg.lineage("sydney") == "swans"

    def test_no_automatic_merger(self, tmp_path: Path) -> None:
        reg = _clubs(tmp_path)
        assert reg.lineage("fitzroy") != reg.lineage("brisbane_lions")

    def test_interval_scoped_alias(self, tmp_path: Path) -> None:
        reg = _clubs(tmp_path)
        assert reg.resolve("Brisbane", 1990) == "brisbane_bears"
        assert reg.resolve("Brisbane", 2005) == "brisbane_lions"
        assert reg.resolve("Brisbane", 1996) is None  # gap: ambiguous, not guessed
        assert reg.resolve("Sydney", 1950) is None  # outside the entity's interval

    def test_unknown_club_created_with_own_lineage(self, tmp_path: Path) -> None:
        reg = _clubs(tmp_path)
        cid = reg.resolve_or_create("Demo Harbour", 2026)
        assert cid == "demo_harbour" and reg.lineage(cid) == cid
        assert reg.created == {"demo_harbour": "Demo Harbour"}
        assert reg.resolve_or_create("Demo Harbour", 2020) == "demo_harbour"
        # known names never auto-create, even outside their interval
        assert reg.resolve_or_create("Sydney", 1950) is None

    @pytest.mark.parametrize("bad", ["10.14.74", "", "  ", "123"])
    def test_malformed_names_never_become_clubs(self, tmp_path: Path, bad: str) -> None:
        assert _clubs(tmp_path).resolve_or_create(bad, 2007) is None

    def test_clubs_and_alias_rows(self, tmp_path: Path) -> None:
        reg = _clubs(tmp_path)
        clubs = {c["club_id"]: c for c in reg.club_rows()}
        assert clubs["sydney"]["active"] is True and clubs["sydney"]["last_season"] is None
        assert clubs["fitzroy"]["last_season"] == 1996 and clubs["fitzroy"]["active"] is False
        assert all(is_safe_id(c) for c in clubs)
        assert len(reg.alias_rows()) == 7

    def test_repo_config_loads_and_resolves_all_legacy_names(self) -> None:
        reg = ids.ClubRegistry.from_csv(REPO_CONFIG / "team_aliases.csv")
        assert reg.resolve("Kangaroos", 2003) == "kangaroos"
        assert reg.resolve("North Melbourne", 2003) == "kangaroos"
        assert reg.resolve("North Melbourne", 2010) == "north_melbourne"
        assert reg.lineage("kangaroos") == reg.lineage("north_melbourne")
        assert reg.lineage("footscray") == reg.lineage("western_bulldogs")
        assert reg.resolve("Footscray", 1997) is None


class TestVenues:
    def test_venue_registry(self, tmp_path: Path) -> None:
        p = tmp_path / "v.csv"
        p.write_text("source_name,venue_id,venue_name,timezone,note\nM.C.G.,mcg,M.C.G.,,\n", encoding="utf-8")
        reg = ids.VenueRegistry.from_csv(p)
        assert reg.resolve("M.C.G.") == "mcg"
        assert reg.resolve("Nowhere Oval") is None
        rows = reg.venue_rows({"mcg": {"M.C.G."}})
        assert rows == [{"venue_id": "mcg", "name": "M.C.G.", "source_names": '["M.C.G."]', "timezone": None}]

    def test_bad_timezone_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "v.csv"
        p.write_text("source_name,venue_id,venue_name,timezone,note\nX,x,X,Mars/Olympus,\n", encoding="utf-8")
        with pytest.raises(ValueError):
            ids.VenueRegistry.from_csv(p)
