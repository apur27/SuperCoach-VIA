---
name: afltables-reconciliation-structure
description: Verified AFL Tables page structure + capture facts for the all-player reconciliation (src/supercoach_via/reconciliation/); census/season/match/profile layouts, request counts, capture identity pinning
metadata:
  type: project
---

Measured live 2026-10-01 (pilot capture, `var/reconciliations/afltables/2026-10-01-pilotA`):
- robots.txt is a genuine HTTP 404 (custom "Broked!" HTML page). 26 letter pages `playersX_idx.html` (letter = SURNAME initial; profile URL folder = FIRST-name initial). Directory census = 13,364 profile links (local snapshot 13,368 players, legacy 13,367 files). All 130 season pages (1897-2026) parse with zero reader problems; 17,056 in-scope matches through 2026-09-30 (equals local `matches` row count).
- Season list comes from `stats_idx.html` year links (`YYYY.html` 1897-1964, `YYYYs.html` 1965+); fixtures from `/afl/seas/YYYY.html`; every game row has a `[Match stats]` link.
- Profile: summary table 0 = per-club-season TOTALS (Year Team # GM W-D-L + 22 stats, no %P; foot rows Totals/Averages), table 1 = per-season AVERAGES (same shape), then one game table per club-season (head rows: `Club - YYYY` colspan 28, then Gm Opponent Rd R # + 22 stats + %P; foot `Totals` = `N (W-D-L)` colspan 3 + stat totals). Every game row Rd cell carries exactly one match link. Some profiles have no `Born:` line.
- Match page: team Totals row never prints %P; pre-1965 Totals carry only goals; a drawn final and its replay share the `Round:` text and differ only by URL/date.
- Notes page: availability matrix (labels I5->IF, OP->1%; no DI column) + exceptions table ("Missing hitouts/behinds/free kicks for/against/all but goals", rows like `1975 R1-R13 | All games`, lineage club names e.g. Sydney=South Melbourne in 1975).
- Throughput: ~2.0 s per request at the policy spacing; ~30,560 requests remain after the 175-request pilot => ~17.4 h. HTML tokenizer (stdlib) parses ~4 MB/s.

Capture design decisions (why): capture identity pins ONLY capture-relevant files (urls/discover/capture/http/sourcepages + source policy); schema.py is append-only after launch; the long run imports code from an immutable snapshot dir (`var/reconciliations/afltables/code-snapshots/`) via PYTHONPATH so editing the worktree can never change a running capture; `Capture._check_code` refuses a resume if those hashes changed (then new plan + `--seed-from`). Plan has `plan_id` (everything) and `capture_identity` (resume identity) so comparison rules can be re-issued without re-capturing.

Local snapshot facts: pre-1984 Brownlow only populated 1931-34 (1,247 rows) in the candidate; quarantine holds 13 `replayed_draw_link_unresolved` player_games rows (drawn finals 1928/1946/1948/1972/1977/1990/2010) = real appearances missing from accepted rows.
