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

Last refresh: 2026-06-22 batch 2 (clearances, contested, hitouts, kicks-handballs, goalassists, single-season). Key changes:
- Neale clearances: 1,970→1,979 (308g, 6.43/g); contested poss: 4,013→4,028 (13.08/g)
- Dangerfield clearances: 370→371g, 1,900→1,916, 5.14→5.16; contested poss: 370→371g, 4,655→4,704, 12.58→12.68; handballs 370→371g, 3,878→3,920, 10.48→10.57; inside_50s career: 1,766→1,788; goal assists: 370→371g, 278→282, 0.75→0.76
- Pendlebury clearances: 1,887→1,891, 4.34→4.35; contested poss: 4,401→4,412, 10.12→10.14; handballs: 5,504→5,531, 12.65→12.71; kicks: 5,482→5,497, 12.60→12.64; inside_50s career: 1,548→1,551
- Cripps clearances: 243→244g, 1,720→1,727; contested poss: 243→244g, 3,526→3,542, 14.51→14.52
- Liberatore clearances: 261→262g, 1,646→1,651, 6.31→6.30; contested poss: 261→262g, 3,194→3,201, 12.24→12.22
- Parker clearances: 328→329g, 1,580→1,581, 4.82→4.81; contested poss: 328→329g, 3,634→3,640, 11.08→11.06; handballs: 328→329g, 3,843→3,855
- Oliver (Clayton) clearances: 218→219g, 1,435→1,440; contested poss: 218→219g, 3,221→3,233, 14.78→14.76; handballs: 218→219g, 3,696→3,712
- Bontempelli clearances: 272→273g, 1,432→1,445, 5.26→5.29; goal assists: 272→273g, 0.88/g
- Max Gawn hitouts (career): 261→262g, 8,673→8,703, 33.23→33.22
- Jarrod Witts hitouts (career): 215→216g, 7,328→7,471, 34.08→34.59
- Reilly O'Brien hitouts (career): 5,142→5,202, 34.98→35.39
- Toby Nankervis hitouts: 181→182g, 4,680→4,695, 25.86→25.80
- Sidebottom kicks: 365→366g, 4,769→4,799, 13.07→13.11
- Kicks #19: Gary Ablett jnr replaced by Dayne Zorko (311g, 4,706, 15.13); Ablett moves to #20
- Nathan Burke removed from kicks top-20 entirely
- Macrae handballs: 279→280g, 4,083→4,098, 14.63→14.64
- Cripps handballs: 243→244g, 3,863→3,882, 15.90→15.91
- Laird handballs: 280→281g, 3,850→3,905, 13.75→13.90
- Neale handballs: 4,663→4,690, 15.14→15.23
- Taylor Walker goal assists: 309→310g, 241→243
- Toby Greene goal assists: 274→275g, 217→218
- Single-season marks: Harris Andrews 221 (2025) enters at #9; Tredrea drops to #10; Tarrant removed
- Single-season tackles: Atkins 205→232; Dunkley 218 (2025) new at #2; Rowell 192→214 moves to #3; Selwood to #4; Priddis to #5; tied block now 6=; Swallow removed
- Single-season kicks: Zorko 2024/566→2025/571
- Single-season clearances: Rowell 190→205 (tied #2= with Neale 205); Serong 184→197 moves to #5; ranks cascade; full restructure
- Single-season inside-50s: Max Holmes 159 (2025) added at 7=; Jordan Dawson 156 (2025) at #10; Dustin Martin and Camporeale removed
- Single-season hit-outs: Witts 2019/1,007→2025/1,040 at #3; Grundy drops to #4; O'Brien 912→972 at #7; Dempsey to #8; Lloyd Meek 948 (2025) new at #9; Minson to #10; McInerney removed
- Single-season brownlow: Rowell 39 (2025) new at #2; Daicos drops to #3; ranks cascade; Neale (31) removed

Previous Last refresh: 2026-06-22 (batch 1 — hub, disposals, games, goals, tackles, marks, brownlow). Changes in that run:
- Pendlebury: disposals 10,986→11,028 (hub table); tackles 2,001→2,012 (hub table); handballs 5,504→5,531 (hub table + disposals tactical); kicks 5,482→5,497 (disposals tactical); Brownlow votes 223→225, rank "4"→"4=", per-game 0.51→0.52 (brownlow page)
- Dangerfield: Brownlow games 370→371, votes 251→259, per-game 0.68→0.70 (brownlow page); games 370→371 (games page + tactical); contested_poss hub 4,655→4,704
- Neale: clearances hub 1,970→1,979; Brownlow rank restored to "4=" with 225 votes (was 9/209 — stale); disposals rank 10→10 (was buried at 11), total 8,418→8,467, per_game 27.33→27.49; tactical per-game 27.33→27.49
- Sidebottom: games 365→366, rank "15"→"14=" (tied with Quinlan at 366); disposals games 365→366, total 8,309→8,367, per_game 22.76→22.86
- Jeremy Cameron: goals games 293→294, total 768→775, per-game 2.62→2.64
- Luke Parker: tackles games 328→329, total 1,580→1,582, per-game 4.82→4.81; disposals games 328→329, total 7,634→7,664, per-game 23.27→23.29
- Jack Steele: tackles games 216→217, total 1,553→1,555, per-game 7.19→7.17
- Dayne Zorko: tackles total 1,517→1,535, per-game 4.88→4.94; rank swapped with Ablett (now #8, Ablett #9)
- Tom Liberatore: tackles games 261→262, total 1,428→1,433
- Marcus Bontempelli: tackles games 272→273, total 1,410→1,417, per-game 5.18→5.19; Brownlow games 272→273; Brownlow tactical 0.71→0.70
- Patrick Cripps: Brownlow games 243→244, per-game 0.78→0.77; tactical games 243→244
- Pendlebury tackles per-game table: 4.62→4.63; tactical 4.62→4.63
- Disposals table reordered: Neale now #10, Dangerfield #11, Tuck #12 (was Tuck #10, Neale #11, Dangerfield #12)
- Disposals table: Macrae now #17 (280g/7,638), Shaw #18 (was Shaw #17, Macrae #18)
- Marks page: no changes (all values matched JSON)

Previous refresh: 2026-06-15 (batch 2 — clearances, contested, hitouts, kicks-handballs, goalassists, single-season). Changes in that run:
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
