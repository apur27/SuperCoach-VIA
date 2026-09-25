"""List-management views: drafts (national/rookie/DraftGuru), schools and contracts.

Stable IDs come from the canonical tables (``draft_event_id``/``observation_id``); rows are
ordered deterministically. Every view discloses source mode: contract observations are
point-in-time observations (``source_type`` live / manual_fixture / legacy) and are not
guaranteed current. Names that were not resolved to a player identity keep ``player_id=None``
and are listed by :func:`unresolved_names`; nothing is guessed.

Season scoping choice for :func:`lists_season`: drafts by draft ``season``, schools by
``draft_year``, contracts by ``contract_end == season`` (that season's out-of-contract class).
"""

from __future__ import annotations

# SQL is composed only from whitelisted identifiers (canonical stat/column names) and int-cast
# literals; every caller-supplied value is a bound parameter.
# ruff: noqa: S608
from collections import Counter
from typing import TYPE_CHECKING

from supercoach_via.storage.queries import SnapshotQuery

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from supercoach_via.publish.view_models import ContractRow, DraftRow, ListsSeason, SchoolRow

CONTRACT_DISCLOSURE = (
    "Contract observations are point-in-time records, not guaranteed current: 'manual_fixture' "
    "rows come from a verified fixture file (not a live feed), 'legacy' rows from the legacy CSV, "
    "'live' rows from a dated source fetch."
)


def _has(q: SnapshotQuery, name: str) -> bool:
    return name in q.manifest.tables and (q.tables is None or name in q.tables)


def _club_names(q: SnapshotQuery) -> dict[str, str]:
    return dict(q.rows("SELECT club_id, name FROM clubs")) if _has(q, "clubs") else {}


def draft_rows(q: SnapshotQuery, season: int | None = None) -> list[DraftRow]:
    from supercoach_via.publish.view_models import DraftRow

    if not _has(q, "draft_events"):
        return []
    names = _club_names(q)
    where, params = ("WHERE season = ?", [int(season)]) if season is not None else ("", [])
    rows = q.rows(
        f"""SELECT season, event_type, draft_round, pick, club_id, club_source_name, player_name,
                   player_id, recruited_from, grade
            FROM draft_events {where}
            ORDER BY season, event_type, draft_round NULLS LAST, pick NULLS LAST, draft_event_id""",
        params,
    )
    return [
        DraftRow(
            season=s, event_type=et, draft_round=rd, pick=pk,
            club=names.get(cid, src) if cid else src, player_name=name, player_id=pid,
            recruited_from=rf, grade=gr,
        )
        for s, et, rd, pk, cid, src, name, pid, rf, gr in rows
    ]


def contract_rows(q: SnapshotQuery, contract_end: int | None = None) -> list[ContractRow]:
    from supercoach_via.publish.view_models import ContractRow

    if not _has(q, "contract_observations"):
        return []
    names = _club_names(q)
    where, params = ("WHERE contract_end = ?", [int(contract_end)]) if contract_end is not None else ("", [])
    rows = q.rows(
        f"""SELECT player_name, player_id, club_id, contract_end, fa_category, observed_at, source_type, notes
            FROM contract_observations {where}
            ORDER BY contract_end NULLS LAST, club_id NULLS LAST, player_name, observation_id""",
        params,
    )
    return [
        ContractRow(player_name=n, player_id=pid, club=names.get(cid, cid) if cid else None,
                    contract_end=ce, fa_category=fa, observed_at=oa, source_type=st, notes=notes)
        for n, pid, cid, ce, fa, oa, st, notes in rows
    ]


def school_rows(q: SnapshotQuery, draft_year: int | None = None) -> list[SchoolRow]:
    from supercoach_via.publish.view_models import SchoolRow

    if not _has(q, "school_observations"):
        return []
    where, params = ("WHERE draft_year = ?", [int(draft_year)]) if draft_year is not None else ("", [])
    rows = q.rows(
        f"""SELECT draft_year, pick, player_name, player_id, school, school_type, confidence
            FROM school_observations {where}
            ORDER BY draft_year NULLS LAST, pick NULLS LAST, observation_id""",
        params,
    )
    return [
        SchoolRow(draft_year=y, pick=p, player_name=n, player_id=pid, school=s, school_type=t, confidence=c)
        for y, p, n, pid, s, t, c in rows
    ]


def lists_season(q: SnapshotQuery, season: int) -> ListsSeason:
    from supercoach_via.publish.view_models import ListsSeason, Source

    drafts = draft_rows(q, season)
    contracts = contract_rows(q, season)
    schools = school_rows(q, season)
    families: Counter[str] = Counter()
    if _has(q, "draft_events"):
        families.update(dict(q.rows("SELECT source_family, COUNT(*) FROM draft_events WHERE season = ? GROUP BY 1",
                                    [int(season)])))
    modes = Counter(c.source_type for c in contracts)
    if contracts:
        contract_note = "Contracts: " + ", ".join(f"{n} {m}" for m, n in sorted(modes.items())) + ". "
        contract_note += CONTRACT_DISCLOSURE
    else:
        contract_note = f"No contract observations with a {season} contract end."
    unresolved = sum(1 for d in drafts if d.player_id is None)
    unresolved += sum(1 for c in contracts if c.player_id is None)
    unresolved += sum(1 for s in schools if s.player_id is None)
    note = (
        f"{contract_note} Drafts: {len(drafts)} events"
        + (f" ({', '.join(f'{f}: {n}' for f, n in sorted(families.items()))})" if families else "")
        + f"; schools: {len(schools)} observations (classifier output, may be ambiguous). "
        f"{unresolved} names are not resolved to a player identity."
    )
    sources = [Source(label=f"Draft source: {f}", note=f"{n} rows") for f, n in sorted(families.items())]
    sources += [Source(label=f"Contract observations ({m})", note=CONTRACT_DISCLOSURE) for m in sorted(modes)]
    return ListsSeason(season=season, drafts=drafts, contracts=contracts, schools=schools,
                       source_note=note, sources=sources)


def unresolved_names(q: SnapshotQuery) -> pd.DataFrame:
    """Every list row whose name is not resolved to a player_id (table, record_id, name, season)."""
    import pandas as pd

    parts = []
    if _has(q, "draft_events"):
        parts.append("""SELECT 'draft_events' AS table_name, draft_event_id AS record_id, player_name,
                               season, COALESCE(club_id, club_source_name) AS club
                        FROM draft_events WHERE player_id IS NULL""")
    if _has(q, "contract_observations"):
        parts.append("""SELECT 'contract_observations', observation_id, player_name, contract_end, club_id
                        FROM contract_observations WHERE player_id IS NULL""")
    if _has(q, "school_observations"):
        parts.append("""SELECT 'school_observations', observation_id, player_name, draft_year, NULL
                        FROM school_observations WHERE player_id IS NULL""")
    if not parts:
        return pd.DataFrame(columns=["table_name", "record_id", "player_name", "season", "club"])
    return q.df(" UNION ALL ".join(parts) + " ORDER BY 1, 2")
