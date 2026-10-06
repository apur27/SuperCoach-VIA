"""Identity corrections proposed from the audit's identity results (a local name or birth date the profile page
contradicts, a duplicate local player, and the verified profile URL each resolved player should carry)."""

from __future__ import annotations

from supercoach_via.reconciliation import identity as ID
from supercoach_via.reconciliation import idfix as IF

U = "https://afltables.com/afl/stats/players/"


def _p(url: str, name: str, born: str | None, apps: set[tuple[str, str]]) -> ID.SourceProfile:
    return ID.SourceProfile(url, name, born, frozenset(apps), frozenset(), True)


def _l(key: str, name: str, born: str | None, apps: set[tuple[str, str]]) -> ID.LocalIdentityRec:
    return ID.LocalIdentityRec("snapshot", key, name, born, (), frozenset(apps), frozenset(), "canonical", None)


A = {("m1", "Essendon"), ("m2", "Essendon")}
B = {("m3", "Carlton"), ("m4", "Carlton")}


def _run(profiles: list[ID.SourceProfile], locals_: list[ID.LocalIdentityRec]) -> list[IF.Proposal]:
    pmap = {p.url: p for p in profiles}
    res = ID.resolve_identities(pmap, locals_, ())
    return IF.identity_proposals(pmap, locals_, res)


def test_a_resolved_player_is_bound_to_its_verified_profile_url() -> None:
    props = _run(
        [_p(U + "A/Ann_Able.html", "Ann Able", "1990-07-01", A)], [_l("legacy:able", "Ann Able", "1990-07-01", A)]
    )
    assert [(p.kind, p.key, p.url, p.fields) for p in props] == [("bind", "legacy:able", U + "A/Ann_Able.html", {})]


def test_a_dropped_surname_prefix_is_repaired_from_the_profile_whose_games_contain_the_players() -> None:
    url = U + "P/Paul_Vander_Haar.html"
    props = _run([_p(url, "Paul Vander Haar", "1958-03-07", A)], [_l("legacy:haar_paul", "Paul Haar", "1958-03-07", A)])
    (fix,) = [p for p in props if p.kind == "repair"]
    assert fix.key == "legacy:haar_paul" and fix.url == url
    assert fix.fields == {"display_name": "Paul Vander Haar", "last_name": "Vander Haar"}


def test_a_wrong_birth_date_is_repaired_and_a_missing_local_game_does_not_block_it() -> None:
    url = U + "R/Ron_McEwin.html"
    props = _run(
        [_p(url, "Ron McEwin", "1928-01-02", A | {("m9", "Essendon")})],
        [_l("legacy:mcewin_ron", "Ron McEwin", "1928-01-01", A)],
    )
    (fix,) = [p for p in props if p.kind == "repair"]
    assert fix.fields == {"birth_date": "1928-01-02"}


def test_no_repair_without_a_unique_unclaimed_superset_profile() -> None:
    # two profiles contain all the local games: nothing is guessed
    props = _run(
        [_p(U + "X/X_One.html", "X One", "1950-01-02", A), _p(U + "X/X_Two.html", "X Two", "1951-01-02", A | B)],
        [_l("legacy:x", "Somebody Else", "1949-05-05", A)],
    )
    assert not [p for p in props if p.kind == "repair"]
    # a profile that another local player already owns is never taken
    props = _run(
        [_p(U + "Y/Y_Y.html", "Yan Yu", "1950-01-02", A)],
        [_l("legacy:yu", "Yan Yu", "1950-01-02", A), _l("legacy:other", "Bo Bo", "1940-02-02", {("m1", "Essendon")})],
    )
    assert not [p for p in props if p.kind == "repair" and p.key == "legacy:other"]


def test_two_local_players_with_identical_games_claiming_one_profile_keep_the_profile_name_as_canonical() -> None:
    url = U + "J/Jonathon_Ross.html"
    props = _run(
        [_p(url, "Jonathon Ross", "1973-11-03", B)],
        [
            _l("legacy:ross_jonathan", "Jonathan Ross", "1973-11-03", B),
            _l("legacy:ross_jonathon", "Jonathon Ross", "1973-11-03", B),
        ],
    )
    (dup,) = [p for p in props if p.kind == "duplicate"]
    assert dup.key == "legacy:ross_jonathan" and dup.fields == {"canonical": "legacy:ross_jonathon"} and dup.url == url
    assert [p.key for p in props if p.kind == "bind"] == ["legacy:ross_jonathon"]


def test_an_unresolved_player_whose_page_contradicts_nothing_is_bound_to_the_unclaimed_superset_profile() -> None:
    url = U + "R/Ron_McEwin.html"
    props = _run([_p(url, "Ron McEwin", "1928-01-02", A | {("m9", "Essendon")})],
                 [_l("legacy:mcewin_ron", "Ron McEwin", "1928-01-02", A)])  # fmt: skip
    assert [(p.kind, p.key, p.url) for p in props] == [("bind", "legacy:mcewin_ron", url)]


def test_a_stub_whose_games_are_a_subset_of_another_local_record_of_the_same_profile_is_the_duplicate() -> None:
    url = U + "R/Roan_Steele.html"
    full = A | B
    props = _run(
        [_p(url, "Roan Steele", "2001-10-22", full)],
        [_l("legacy:steele_roan_22102001", "Roan Steele", "2001-10-22", full),
         _l("legacy:steele_roan_19092002", "Roan Steele", "2002-09-19", A)],
    )  # fmt: skip
    (dup,) = [p for p in props if p.kind == "duplicate"]
    assert dup.key == "legacy:steele_roan_19092002" and dup.fields == {"canonical": "legacy:steele_roan_22102001"}
    # a subset that is NOT contained in the other record's games is never merged
    props = _run(
        [_p(url, "Roan Steele", "2001-10-22", full)],
        [_l("legacy:steele_roan_22102001", "Roan Steele", "2001-10-22", B),
         _l("legacy:steele_roan_19092002", "Roan Steele", "2002-09-19", A)],
    )  # fmt: skip
    assert not [p for p in props if p.kind == "duplicate"]


def test_a_stub_whose_games_appear_in_teammates_profiles_too_still_finds_its_same_named_complete_record() -> None:
    url = U + "R/Roan_Steele.html"
    props = _run(
        [_p(url, "Roan Steele", "2001-10-22", A | B), _p(U + "T/Team_Mate.html", "Team Mate", "1999-01-02", A | B)],
        [_l("legacy:steele_roan_22102001", "Roan Steele", "2001-10-22", A | B),
         _l("legacy:steele_roan_19092002", "Roan Steele", "2002-09-19", A),
         _l("legacy:mate_team", "Team Mate", "1999-01-02", A | B)],
    )  # fmt: skip
    assert [(p.key, p.fields) for p in props if p.kind == "duplicate"] == [
        ("legacy:steele_roan_19092002", {"canonical": "legacy:steele_roan_22102001"})
    ]
