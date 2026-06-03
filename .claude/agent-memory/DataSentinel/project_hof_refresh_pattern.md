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

Last refresh: 2026-06-02. Changes in that run:
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
