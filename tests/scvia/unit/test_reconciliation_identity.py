"""Player identity and match alignment (T01, T03, T04, T08; DESIGN section 7)."""

from __future__ import annotations

from supercoach_via.reconciliation import identity as I
from supercoach_via.reconciliation.rules import IdentityOverride

URL = "https://afltables.com/afl/stats/players/{0}/{1}.html"


def prof(slug: str, name: str, born: str | None, apps: set[tuple[str, str]], *, census: bool = True) -> I.SourceProfile:
    url = URL.format(slug[0].upper(), slug)
    return I.SourceProfile(
        url=url,
        name=name,
        born=born,
        appearances=frozenset(apps),
        memberships=frozenset({(int(m.split("/")[-2][:4]) if False else 2020, c) for m, c in apps}),
        in_census=census,
    )


def mships(*pairs: tuple[int, str]) -> frozenset[tuple[int, str]]:
    return frozenset(pairs)


def src(
    slug: str, name: str, born: str | None, ms: list[tuple[int, str]], apps: set[tuple[str, str]] | None = None
) -> I.SourceProfile:
    return I.SourceProfile(
        url=URL.format(slug[0].upper(), slug), name=name, born=born, appearances=frozenset(apps or set()),
        memberships=frozenset(ms), in_census=True,
    )  # fmt: skip


def loc(
    key: str, name: str, born: str | None, ms: list[tuple[int, str]], apps: set[tuple[str, str]] | None = None, **kw
) -> I.LocalIdentityRec:  # type: ignore[no-untyped-def]
    return I.LocalIdentityRec(
        layer="snapshot", key=key, name=name, born=born, source_urls=tuple(kw.get("urls", ())),
        appearances=frozenset(apps or set()), memberships=frozenset(ms), identity_status=kw.get("status", "canonical"),
        canonical_id=kw.get("canonical"),
    )  # fmt: skip


def resolve(profiles: list[I.SourceProfile], locals_: list[I.LocalIdentityRec], overrides=()) -> I.IdentityResult:  # type: ignore[no-untyped-def]
    return I.resolve_identities({p.url: p for p in profiles}, locals_, tuple(overrides))


def test_name_and_full_dob_with_corroborating_membership_resolves() -> None:  # rule 3
    p = src("Ann_Able", "Ann Able", "1990-07-01", [(2020, "Alpha")])
    r = resolve([p], [loc("legacy:a", "Ann Able", "1990-07-01", [(2020, "Alpha")])])
    res = r.resolutions["legacy:a"]
    assert (res.status, res.url, res.rule) == ("resolved", p.url, "R3-name-dob-membership")
    assert r.profile_to_locals[p.url] == ["legacy:a"]


def test_identical_names_are_separated_by_dob_and_never_by_first_seen() -> None:  # T03
    a = src("Gary_Ablett0", "Gary Ablett", "1961-10-01", [(1984, "Geelong")])
    b = src("Gary_Ablett1", "Gary Ablett", "1984-05-14", [(2002, "Geelong")])
    r = resolve([a, b], [loc("legacy:sr", "Gary Ablett", "1961-10-01", [(1984, "Geelong")]),
                         loc("legacy:jr", "Gary Ablett", "1984-05-14", [(2002, "Geelong")])])  # fmt: skip
    assert r.resolutions["legacy:sr"].url == a.url and r.resolutions["legacy:jr"].url == b.url


def test_no_url_is_ever_constructed_from_a_name() -> None:  # T03
    r = resolve([], [loc("legacy:ann_able_01071990", "Ann Able", "1990-07-01", [(2020, "Alpha")])])
    assert (
        r.resolutions["legacy:ann_able_01071990"].status == "unresolved"
        and r.resolutions["legacy:ann_able_01071990"].url is None
    )


def test_conflicting_dob_with_a_verified_url_is_an_identity_conflict_not_a_comparison() -> None:  # T03
    p = src("Ann_Able", "Ann Able", "1990-07-01", [(2020, "Alpha")])
    r = resolve([p], [loc("legacy:a", "Ann Able", "1991-08-02", [(2020, "Alpha")], urls=[p.url])])
    res = r.resolutions["legacy:a"]
    assert res.status == "conflict" and res.url is None and "born" in res.detail


def test_verified_source_url_with_agreeing_evidence_resolves_by_rule_1() -> None:
    p = src("Ann_Able", "Ann Able", "1990-07-01", [(2020, "Alpha")])
    r = resolve([p], [loc("legacy:a", "Different Name", "1990-07-01", [(2020, "Alpha")], urls=[p.url])])
    assert (r.resolutions["legacy:a"].status, r.resolutions["legacy:a"].rule) == ("resolved", "R1-verified-url")


def test_verified_url_to_a_profile_with_a_totally_incompatible_career_is_a_conflict() -> None:
    p = src("Ann_Able", "Ann Able", None, [(1950, "Alpha")])
    r = resolve([p], [loc("legacy:a", "Ann Able", None, [(2020, "Beta")], urls=[p.url])])
    assert r.resolutions["legacy:a"].status == "conflict"


def test_multiword_surname_resolves_by_unique_dob_and_identical_memberships() -> None:  # rule 3b
    p = src("Ah_Chee0", "Brad Ah Chee", "1997-03-03", [(2018, "Adelaide"), (2019, "Adelaide")])
    r = resolve([p], [loc("legacy:chee_brad", "Brad Chee", "1997-03-03", [(2018, "Adelaide"), (2019, "Adelaide")])])
    res = r.resolutions["legacy:chee_brad"]
    assert (res.status, res.rule) == ("resolved", "R3b-dob-memberships") and any(
        f.category == "IDENTITY_VARIANCE" for f in r.findings
    )


def test_split_surname_with_only_partial_membership_overlap_stays_unresolved() -> None:  # rule 3b is set equality
    p = src("Ah_Chee0", "Brad Ah Chee", "1997-03-03", [(2018, "Adelaide"), (2019, "Adelaide")])
    r = resolve([p], [loc("legacy:chee_brad", "Brad Chee", "1997-03-03", [(2018, "Adelaide")])])
    assert r.resolutions["legacy:chee_brad"].status == "unresolved"


def test_missing_dob_is_resolved_only_by_exact_appearance_set_equality() -> None:  # rule 4
    apps = {("m1", "Alpha"), ("m2", "Alpha")}
    p = src("Kelly_Robinson", "Kelly Robinson", None, [(2020, "Alpha")], apps)
    ok = resolve([p], [loc("legacy:k", "K Robinson", None, [(2020, "Alpha")], apps)])
    assert (ok.resolutions["legacy:k"].status, ok.resolutions["legacy:k"].rule) == ("resolved", "R4-appearance-set")
    partial = resolve([p], [loc("legacy:k", "Kelly Robinson", None, [(2020, "Alpha")], {("m1", "Alpha")})])
    assert partial.resolutions["legacy:k"].status == "unresolved"  # overlap is never enough


def test_january_first_dob_is_low_precision_and_never_a_sole_key() -> None:  # rule 3 / 4
    p = src("Old_Timer", "Old Timer", "1890-01-01", [(1920, "Alpha")], {("m1", "Alpha")})
    q = src("Old_Timer0", "Old Timer", "1890-01-01", [(1925, "Beta")], {("m9", "Beta")})
    r = resolve([p, q], [loc("legacy:o", "Old Timer", "1890-01-01", [(1925, "Beta")], {("m9", "Beta")})])
    res = r.resolutions["legacy:o"]
    assert res.url == q.url and res.rule == "R4-appearance-set"  # set equality, not the 1 January date


def test_two_local_ids_claiming_one_profile_are_a_conflict_with_no_merged_counts() -> None:  # T04
    p = src("Ann_Able", "Ann Able", "1990-07-01", [(2020, "Alpha")])
    r = resolve([p], [loc("legacy:a1", "Ann Able", "1990-07-01", [(2020, "Alpha")]),
                      loc("legacy:a2", "Ann Able", "1990-07-01", [(2020, "Alpha")])])  # fmt: skip
    assert r.resolutions["legacy:a1"].status == r.resolutions["legacy:a2"].status == "conflict"
    assert r.profile_to_locals.get(p.url, []) == [] and any(f.category == "IDENTITY_CONFLICT" for f in r.findings)


def test_alias_chains_resolve_to_the_canonical_player_without_double_counting() -> None:  # T04
    p = src("Ann_Able", "Ann Able", "1990-07-01", [(2020, "Alpha")])
    canon = loc("legacy:a", "Ann Able", "1990-07-01", [(2020, "Alpha")])
    alias = loc("legacy:a_dup", "Ann Able", "1990-07-01", [(2020, "Alpha")], status="alias", canonical="legacy:a")
    r = resolve([p], [canon, alias])
    assert r.resolutions["legacy:a"].status == "resolved"
    assert r.resolutions["legacy:a_dup"].status == "alias" and r.resolutions["legacy:a_dup"].url == p.url
    assert r.profile_to_locals[p.url] == ["legacy:a"] and r.owner["legacy:a_dup"] == "legacy:a"


def test_alias_loop_and_dangling_alias_are_rejected() -> None:  # T04
    a = loc("legacy:a", "A", None, [], status="alias", canonical="legacy:b")
    b = loc("legacy:b", "B", None, [], status="alias", canonical="legacy:a")
    r = resolve([], [a, b])
    assert {r.resolutions["legacy:a"].status, r.resolutions["legacy:b"].status} == {"conflict"}
    dangling = resolve([], [loc("legacy:x", "X", None, [], status="alias", canonical="legacy:missing")])
    assert dangling.resolutions["legacy:x"].status == "conflict"


def test_an_override_resolves_identity_only_and_changes_nothing_else() -> None:
    p = src("Ann_Able", "Ann Able", "1990-07-01", [(2020, "Alpha")])
    ov = IdentityOverride("legacy:z", p.url, "abc", "page A row 3", "same career", "reviewer")
    r = resolve([p], [loc("legacy:z", "Zed", None, [])], [ov])
    res = r.resolutions["legacy:z"]
    assert (res.status, res.rule, res.url) == ("resolved", "R2-override", p.url)


def test_source_profiles_with_no_local_match_are_reported_missing() -> None:  # T01
    a = src("Ann_Able", "Ann Able", "1990-07-01", [(2020, "Alpha")])
    b = src("Bob_Baker", "Bob Baker", "1991-02-02", [(2020, "Beta")])
    r = resolve([a, b], [loc("legacy:a", "Ann Able", "1990-07-01", [(2020, "Alpha")])])
    assert r.source_unmatched == [b.url]


def test_normalisation_keeps_distinct_names_distinct() -> None:
    assert I.norm_name("  Tom   O’Halloran ") == I.norm_name("Tom O'Halloran")
    assert I.norm_name("Aaron Black") == I.norm_name("aaron BLACK")
    assert I.norm_name("Jean-Pierre Smith") != I.norm_name("Jean Pierre Smith")
    assert I.norm_name("Ah Chee") != I.norm_name("Chee")


# -- matches --------------------------------------------------------------------------------


def mh(
    url: str, season: int, stage: str, teams: tuple[str, str], date: str, *, drawn: bool = False
) -> I.SourceMatchHeader:
    return I.SourceMatchHeader(url=url, season=season, stage_text=stage, teams=frozenset(teams), date=date, drawn=drawn)


def lm(mid: str, season: int, stage: str, teams: tuple[str, str], date: str, replay: int = 0) -> I.LocalMatchRec:
    return I.LocalMatchRec(
        match_id=mid, season=season, stage_text=stage, teams=frozenset(teams), date=date, replay=replay
    )


def test_matches_align_by_season_stage_and_team_pair_regardless_of_home_away_order() -> None:
    m = I.map_matches(
        [lm("m1", 2026, "1", ("Alpha", "Beta"), "2026-03-05")], [mh("u1", 2026, "1", ("Beta", "Alpha"), "2026-03-05")]
    )
    assert m.local_to_source == {"m1": "u1"} and m.source_to_local == {"u1": "m1"} and m.ambiguous == []


def test_drawn_final_and_replay_are_distinguished_by_date_order_and_replay_ordinal() -> None:  # T08
    src_ms = [mh("u2", 2010, "Grand Final", ("Collingwood", "St Kilda"), "2010-10-02"),
              mh("u1", 2010, "Grand Final", ("Collingwood", "St Kilda"), "2010-09-25", drawn=True)]  # fmt: skip
    loc_ms = [lm("g0", 2010, "Grand Final", ("Collingwood", "St Kilda"), "2010-09-25", 0),
              lm("g1", 2010, "Grand Final", ("Collingwood", "St Kilda"), "2010-10-02", 1)]  # fmt: skip
    m = I.map_matches(loc_ms, src_ms)
    assert m.local_to_source == {"g0": "u1", "g1": "u2"}


def test_only_the_replay_present_locally_is_paired_by_ordinal_not_by_first_come() -> None:  # T08
    src_ms = [
        mh("u1", 2010, "Grand Final", ("A", "B"), "2010-09-25", drawn=True),
        mh("u2", 2010, "Grand Final", ("A", "B"), "2010-10-02"),
    ]
    m = I.map_matches([lm("g1", 2010, "Grand Final", ("A", "B"), "2010-10-02", 1)], src_ms)
    assert m.local_to_source == {"g1": "u2"} and m.source_only == ["u1"]


def test_replay_ordinal_that_disagrees_with_source_order_is_ambiguous_not_guessed() -> None:  # T08
    src_ms = [
        mh("u1", 2010, "Grand Final", ("A", "B"), "2010-09-25", drawn=True),
        mh("u2", 2010, "Grand Final", ("A", "B"), "2010-10-02"),
    ]
    m = I.map_matches([lm("g", 2010, "Grand Final", ("A", "B"), "2010-10-02", 0)], src_ms)
    assert m.local_to_source == {} and m.ambiguous == ["g"]


def test_source_only_and_local_only_matches_are_listed() -> None:  # T28
    m = I.map_matches([lm("only_local", 2026, "1", ("A", "B"), "2026-03-05")],
                      [mh("u9", 2026, "Grand Final", ("A", "B"), "2026-09-26")])  # fmt: skip
    assert m.source_only == ["u9"] and m.local_only == ["only_local"]
