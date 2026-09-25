# Data contracts

Two contracts govern the rewrite. Both are generated from code, so this page only tells you
where each one lives and the rules around it. Nothing here should be copied as a
second, hand-maintained definition.

## Canonical dataset (operator side)

The source of truth is `supercoach_via.domain.schemas.TABLES`, with explicit Arrow types
and a `schema_version`. Tables are stored as immutable, content-addressed Parquet
fragments under `<data_root>/fragments/`. They are referenced by a snapshot manifest,
`<data_root>/snapshots/<sha256>.json`, whose ID is the SHA-256 of its semantic content.
`<data_root>/current.json` is the only accepted pointer. It moves only after
`validate_dataset` passes, and then atomically.

| Table | Key | Partition | Columns |
|---|---|---|---|
| `players` | player_id | - | 19 |
| `player_aliases` | player_id, alias | - | 5 |
| `clubs` | club_id | - | 6 |
| `club_aliases` | alias, valid_from_season | - | 5 |
| `venues` | venue_id | - | 4 |
| `seasons` | season | - | 8 |
| `matches` | match_id | season | 41 |
| `player_games` | match_id, player_id, club_id | season | 45 |
| `lineups` | match_id, club_id, player_id | season | 13 |
| `draft_events` | draft_event_id | - | 16 |
| `contract_observations` | observation_id | - | 16 |
| `school_observations` | observation_id | - | 14 |
| `live_snapshots` | source_game_id, payload_hash | - | 18 |
| `source_observations` | source_ref | - | 12 |
| `quality_issues` | issue_id | - | 11 |
| `quarantine` | quarantine_id | - | 10 |
| `source_files` | path | - | 9 |
| `legacy_predictions` | prediction_row_id | - | 16 |
| `legacy_rank_scores` | player_slug | - | 8 |
| `legacy_rank_yearly` | season, player_slug | - | 11 |
| `legacy_top100_bios` | serial_number | - | 8 |

Rules, each enforced by tests under `tests/scvia/`:

- **Nulls stay unknown.** A blank count is `null`, never `0`. A sum with no observed games
  is `null`. Means use observed denominators.
- **Dates carry their quality.** Legacy "March 1 + weeks" dates are `date_quality=inferred`
  and are never used to link rows. Dates resolved from a fixture are `fixture_verified`.
- **Every row keeps its provenance.** Rows carry `provenance`, `source_path`,
  `source_sha256` and `source_row`. A fetched row also carries `available_at`, the fetch
  instant. Legacy rows have `available_at = null` (unknown).
- **Quarantined rows keep their original cells in `raw`.** A row resolved by a later
  verified repair keeps that evidence under `resolved_by_repair:<reason>`. Validation
  accepts the resolution only when the linked row actually exists.
- **Updates create new snapshots.** `storage.snapshots.apply_upserts` writes a new
  candidate that replaces rows by table key. Only the partitions it touches are rewritten;
  the base snapshot is never changed.

## Public release (browser, reports, downloads)

The source of truth is `supercoach_via.publish.view_models` (`PUBLIC_MODELS`).

- `scvia schemas --out schemas` regenerates `schemas/*.schema.json`. The contract test
  `tests/scvia/contract/test_public_schemas.py` fails if the checked-in files are stale.
- `web/scripts/gen-types.mjs` generates the TypeScript types and ajv runtime validators from
  those schemas. `npm run gen:types:check` fails on drift.
- One release manifest (`release.json`) lists every resource with its relative path,
  SHA-256 and byte count. The browser fetches only resources listed there, from the same
  origin, and validates each one at runtime. JSON never contains NaN or Infinity.
- Payload shapes worth knowing:
  - match box scores (`MatchDetail`) and player-season game logs (`PlayerSeasonGames`)
    store `stats` as positional arrays aligned to `stat_columns`;
  - live snapshots (`LivePlayerRow`) store keyed stats limited to `reliable_fields`;
  - `null` always means "not recorded".
