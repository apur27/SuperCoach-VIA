---
name: hof-refresh-pattern
description: Hall of Fame stat-leader docs refresh pattern — JSON source, docs touched, key gotchas
metadata:
  type: project
---

Ground truth for all [data] tags in hall-of-fame stat docs is `docs/hall-of-fame/_stat_leaders.json`.

Hub page: `docs/hall-of-fame-stat-leaders.md` — has council-pipeline stamp (DataSentinel + Gaffer + Skeptic) and "Last refreshed:" date. Update both on every refresh.

Sub-pages: `hall-of-fame-stat-{goals,games,disposals,marks,tackles,brownlow,...}.md` — have "Published:" date (no council stamp). Update date on every touch.

**Why:** JSON is recomputed from player CSVs; sub-page numbers lag until refreshed manually.

**How to apply:** On every stat-leaders refresh, read the JSON first, diff against each doc, update only [data]-tagged numbers that changed.

Last refresh: 2026-06-15 (batch 2 — clearances, contested, hitouts, kicks-handballs, goalassists, single-season). Changes in that run:
- Neale: 306→308 games; clearances 1955→1970, per_game 6.39→6.40; handballs 4626→4663, per_game 15.12→15.14; contested 3986→4013, rank #5→#3
- Dangerfield: 368→370 games; clearances 1891→1900; contested poss 4631→4655; handballs 3857→3878; goal assists 275→278
- Pendlebury: 434→435 games; clearances 1884→1887; contested poss 4396→4401, per_game 10.13→10.12; kicks 5472→5482, per_game 12.61→12.60; handballs 5494→5504, per_game 12.66→12.65; goal assists per_game 0.76→0.75
- Cripps: 242→243 games; clearances 1717→1720, per_game 7.10→7.08; contested poss 3520→3526, per_game 14.55→14.51, rank #10→#9; handballs 3850→3863, per_game 15.91→15.90, rank #15→#14
- Parker: 326→328 games; clearances 1579→1580, per_game 4.84→4.82; contested poss 3623→3634, per_game 11.11→11.08; handballs 3829→3843, per_game 11.75→11.72, rank #16→#17
- Bontempelli: clearances 1419→1432 games 270→272, rank #20→#18; goal assists 238→239 games 270→272
- Oliver (Clayton): clearances 1427→1435 games 217→218; contested poss 3208→3221 games 217→218; handballs rank #20 entry — new entrant (3696), replaced Daniel Cross (3687)
- Max Gawn: hitouts 8598→8673 games 259→261, per_game 33.20→33.23
- Brodie Grundy: hitouts 8022→8074 games 253→255, per_game 31.71→31.66
- Jarrod Witts: hitouts 7312→7328 games 214→215, per_game 34.17→34.08
- Wines: clearances games 281→283 total unchanged; handballs 3925→3951 games 281→283, per_game 13.97→13.96; contested poss 3280→3301 games 281→283, per_game 11.67→11.66
- Treloar: handballs 3906→3916 games 261→263, per_game 14.97→14.89
- Macrae: handballs 4055→4083 games 277→279, per_game 14.64→14.63
- Laird: handballs 3823→3850 games 278→280, rank #17→#16
- Walker: goal assists 240→241 games 308→309
- Greene (Toby): goal assists 215→217 games 273→274, rank #19→#18
- Simon Black: contested poss rank #9→#10 (overtaken by Cripps)
- Kennedy (Josh): contested poss rank #3→#4 (overtaken by Neale)
- Clearances single-season #10: Matt Priddis 2015/183 replaced by Tom Green 2025/183
- Dangerfield inside_50s career: 1754→1766 (mentioned in contested prose)

Previous refresh: 2026-06-02. Changes in that run:
- Pendlebury: 433→434 games, disposals 10933→10966, tackles 1997→1999, handballs 5477→5494, kicks 5456→5472, goal assists 325→328
- Neale: 305→306 games, disposals 8330→8353, clearances 1947→1955
- Sidebottom: 364→365 games, disposals 8296→8309
- Dangerfield: 367→368 games, disposals 8282→8295, contested poss 4627→4631
- Jack Steele: 213→214 games, tackles 1538→1540, per-game 7.22→7.20
- Jeremy Cameron: 290→291 games, goals 761→765
- Bontempelli: brownlow games 269→270
- Cripps: brownlow games 241→242
- Games doc rank 15/16: Sidebottom/Johnson tie split (Sidebottom now 365, Johnson 364)
- Games doc rank 20: Bruce Doull replaced by Paul Roos
- Tackles doc rank 20: Rory Sloane replaced by Marcus Bontempelli (1399 tackles)
