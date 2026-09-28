# Independent recovery review — 2026-09-28

Model: Cursor Grok 4.7 High. Read-only fresh chat `5f051947-81a1-4dbc-a654-5569a4a02d2d`; completed at 10:58:25 UTC. This is the review before the two queued source corrections, not a final approval.

**BLOCK** for a local merge. Two source defects are still open. The grand-final rows, the sealed site, and the deployment limits are not the reason.

A scratch smoke is still running (`/tmp/scvia-scratch-smoke-season-20260928`, currently sealing `20260928T105232Z-5f3c67f392ee`). I did not edit the worktree, the main checkout, or that scratch tree.

## Merge defects

1. A successful refresh that checks a fixture but does not change match rows leaves `fixture_checked_at` stale.

`_with_season_aggregates` returns immediately when there are no match upserts, and otherwise rebuilds only seasons present in those upserts:

```133:139:src/supercoach_via/pipeline.py
    if not match_upserts:
        return merged
    ...
    seasons = sorted({int(row["season"]) for row in match_upserts})
```

The refresh call does pass `checked_seasons=set(plan.seasons)` (`pipeline.py:306-308`), but those seasons never reach the query unless they also have a match upsert. `season_aggregate_rows` itself is right for the rows it is given: counts and dates come from the merged matches, `schedule_complete` and `source_status` stay as recorded, and `fixture_checked_at` moves only when that season was checked (`refresh.py:360-396`). The pipeline never asks it to do that for an unchanged checked season. `test_pipeline.py` covers an added match and an unreachable source. It has no regression for a passing refresh whose only writes are source observations. The scratch copy still has the same early return at line 133, so this smoke cannot close the hole.

Partial failure is already correct: a non-verified refresh returns before any upsert (`pipeline.py:291-296`), and `test_unreachable_source_refresh_is_partial_and_never_moves_the_pointer` locks the accepted pointer.

2. Every match page, including the fetched grand final, is labeled as a legacy CSV import.

```198:198:src/supercoach_via/publish/resources.py
            sources=[Source(label="Legacy match/player CSV import", url=None, note=r.get("source_path"))],
```

On the sealed detail for `m:2026:gf:brisbane_lions:fremantle:0`, the label is that sentence and the note is `https://afltables.com/afl/stats/games/2026/081920260926.html`. The parent note already queues this until the scratch cycle exits. It is still a merge defect once that cycle exits.

## Grand final

This part agrees, on the release the parent named. I did not re-run `verify_grand_final.py`.

Release `20260928T104520Z-2561b2316c8a`, snapshot `sha256:aa836549e10e96a9039cc4642bbe563b94247e0bfaf95e745a0971cc0bb2b98f`. `validation.json` is PASS, seal `18418f5b…822268b4`. I did not recompute that seal. The public grand-final detail and the copy under `site/data/` are the same 6,973 bytes: 23 and 23 players, date 2026-09-26, stage `gf`, 218 matches in the 2026 index. Jye Amiss’s behinds and hitouts are JSON null. `brownlow_votes` is absent from the published columns because that column is blank for every player; the compact packer drops an all-null column. The parent’s after-fix report records 46 rows, 1,058 source cells including blanks, and matching public and site resources. The season row is 218 complete, last date 2026-09-26, `schedule_complete` null, `source_status` null.

`data/matches/matches_2026.csv` has 218 lines and no 26 September or Grand Final row. It is still the pre-final migration input.

`var/agent-runs/cursor-grok47-recovery/gf-season-growth.json` is the earlier site (`20260928T102101Z-d2e56565da24`, 276,811,441 bytes). The parent file and `gf-season-growth-corrected.json` both name `20260928T104520Z-2561b2316c8a` at 276,820,994 bytes. From that file, the 2021–2025 mean is 4,007,870 bytes. Adding three seasons at that mean plus 5 MiB gives 294,087,484 bytes (280.46 MiB), under 282 MiB and under 300 MiB. The growth file does not store that projection; I applied the same formula as the September 27 notes.

The body of `docs/rewrite/CURSOR_PROGRESS.md` and `docs/rewrite/IMPLEMENTATION_STATUS.md` still say the grand final is not in the snapshot. Those sentences are stale. The parent notes at the top of `CURSOR_PROGRESS.md` and the recovery reports above are the current grand-final evidence.

## What still holds

Publication still binds one validated inventory before `active_release()`, and a site without a seal is not published from `public/` (`release.py:738-796`). `docs/operations.md` tells the operator to seal, then preview with the base-aware Node server, and it keeps the knowledge-cutoff rule. `SWITCH_PLAN.md` still says the switch is not activated. `scripts/scvia_weekly.sh` is mode 775 and does not publish. Package config and report templates live under `src/supercoach_via/` and are checked by `test_packaged_config.py`. I did not rebuild the wheel.

## Not a merge defect, and not done

Production activation, a schedule, and a live publish remain off. Two genuine shadow cycles have not been run. The scratch smoke in progress is a captured-source replay: its log compares snapshot `sha256:55e295f1…`, the pre-final import, and it has not exited. It is not a live cycle and it is not evidence for this grand-final release.

I did not rerun pytest, Vitest, Playwright, mypy, pack/verify, a wheel install, or training. I did not recompute a source inventory for this tree. The September 27 inventory `fb871eda…` (857 files) is historical.
