---
name: afltables-player-profile-url
description: afltables player profile URL uses FIRST-name initial (not last name) and needs requests+lxml, not bare read_html
type: reference
---

afltables player profile pages: `https://afltables.com/afl/stats/players/{initial}/{First}_{Last}.html`

- **initial = first letter of the player's FIRST name**, NOT the last name.
  - Scott Pendlebury -> `.../players/S/Scott_Pendlebury.html` (works). `.../players/P/...` 404s.
  - Marcus Bontempelli -> `.../players/M/Marcus_Bontempelli.html` (works). `.../players/B/...` 404s.
  - (The game_scraper task spec stated last-name initial; that is wrong - it 404s. Implementation in `_player_url_from_csv_path` uses first-name initial.)
- Our perf CSV filename is `<lastname>_<firstname>_<DDMMYYYY>_performance_details.csv`, so first=parts[1], last=parts[0], initial=first[0].upper().

**Fetching:** `pd.read_html(url)` 404s because pandas' default urllib User-Agent is blocked by afltables. Must fetch with `requests.get()` (the UA the rest of game_scraper uses works), then `pd.read_html(io.StringIO(resp.text), flavor='lxml')`. html5lib is NOT installed in the venv; lxml IS - pass `flavor='lxml'` explicitly.

**Totals row:** table 0 on the profile is the season-by-season block; its last row has first cell "Totals". Career-total columns: GM (games), DI (disposals), GL (goals), TK (tackles), CL (clearances), plus KI/MK/HB/HO/etc.

Reconciliation engine lives in `scrapers/game_scraper.py::audit_player_career_totals`, wired into `refresh_data.py` (audits only player files that grew this refresh). Games reconcile via max(games_played); counting stats via fillna(0).sum(). Era gate skips TK pre-1987 and CL pre-1998 (see [[data_stat_coverage_eras]]).
