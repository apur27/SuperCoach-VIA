---
name: afl-data-source-urls
description: Scrapable vs SPA-shell URL map for AFL ladder/stats/FA reconciliation — afltables works, AFL.com.au hubs are empty SPAs
metadata:
  type: reference
---

URL map for ongoing AFL data reconciliation. Verified 2026-06-20 via WebFetch.

**afltables = the only reliable structured source. AFL.com.au data hubs are SPA shells (return nothing to a plain HTTP fetch).**

| Data type | Pattern | Status |
|---|---|---|
| Ladder (P/W/L/D/For/Against/%/Pts) | `https://afltables.com/afl/seas/<year>.html` | HTML, scrapable. Primary. Also has full round fixture below ladder. |
| Season per-player stat aggregates | `https://afltables.com/afl/stats/<year>.html` | HTML, scrapable. ~26 cols: GM KI MK HB DI GL BH HO TK CL FF/FA CP CM MI 1% %P SU etc. Season totals, NOT career. |
| Career game counts / career totals | `https://afltables.com/afl/stats/players/<idx>/<slug>.html` | Per-player career page; all-season Totals row = career. Slug format unverified in 2026-06 session — confirm against one live player before wiring. See [[reconciliation_source_afltables_player]]. |
| FA / contract status | `https://www.afl.com.au/news/<id>/<slug>` (+ `/trade`, `/trade/news`) | Server-rendered ARTICLE prose only. Players + restricted/unrestricted listed with inline `*`/`^` symbols. NOT structured — manual/LLM parse only. PARTIAL. |

**Dead ends (do NOT fetch — empty JS SPA shells, only nav/branding/Loading placeholder):**
- `https://www.afl.com.au/ladder` — no ladder rows in served HTML
- `https://www.afl.com.au/stats` — no current-season rows; only a static all-time record snippet

AFL.com.au is useful ONLY for its server-rendered news/article pages, never its data hubs.
`https://www.afl.com.au/news/trade-and-free-agency` is a 404 — correct trade roots are `/trade` and `/trade/news`.
