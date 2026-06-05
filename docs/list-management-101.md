# List Management 101: Is Building Around Top-10 Draft Picks the Path to Premiership Dominance?

> Cross-references web-sourced draft data (Draftguru, AFL.com.au, Wikipedia) against repo-verified  
> premiership results and player career files. 26 premierships, 250+ top-10 draft selections, 2000–2025.

[← Back to main README](../README.md)

---

## The question

Every September, AFL clubs with wooden spoons console themselves with the same refrain: *"We'll get the top pick and rebuild."* The theory is elegant — load up on the best young talent year after year, develop them in parallel, and a dynasty assembles itself. The data across 26 premierships tells a more complicated, and more instructive, story.

---

## The dynasties that were built on top picks

The evidence that top-10 picks matter is real. Every sustained dynasty of the modern era has 3–5 top-10 selections as cornerstones. But the *way* those picks function is nothing like the naive theory.

**Hawthorn's 2004 draft: the gold standard.** The Hawks used picks 2, 5, and 7 in a single draft to select Jarryd Roughead, Lance Franklin, and Jordan Lewis — three Hall of Fame careers from the same 18-year-old cohort. Add Luke Hodge from the 2001 national draft (pick 1), and you have four top-10 picks who developed together, peaked together, and delivered 4 premierships **[data: data/matches/ ; round_num==Grand Final & team_1_team_name==Hawthorn OR team_2_team_name==Hawthorn ; year ; count — years 2008,2013,2014,2015]** in a span of eight seasons. Career games verified from this repo: Hodge 346 **[data: data/player_data/hodge_luke_15061984_performance_details.csv ; all ; - ; count]**, Roughead 283 **[data: data/player_data/roughead_jarryd_23011987_performance_details.csv ; all ; - ; count]**, Franklin 354 **[data: data/player_data/franklin_lance_30011987_performance_details.csv ; all ; - ; count]**, Lewis 319 **[data: data/player_data/lewis_jordan_24041986_performance_details.csv ; all ; - ; count]**.

**Richmond's two-draft nucleus.** Trent Cotchin (2007 national draft, pick 2) and Dustin Martin (2009 national draft, pick 3) — sourced from Draftguru — arrived two years apart, committed to one club, and jointly anchored three premierships in four seasons. Cotchin: 306 games at Richmond **[data: data/player_data/cotchin_trent_07041990_performance_details.csv ; all ; - ; count]**. Martin: 302 games **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; all ; - ; count]**.

**Brisbane's current rebuild.** The Lions' 2024–2025 back-to-back flags were built on Cameron Rayner (2017 pick 1, 175 games **[data: data/player_data/rayner_cam_21101999_performance_details.csv ; all ; - ; count]**) and Hugh McCluggage (2016 pick 3, 214 games **[data: data/player_data/mccluggage_hugh_03031998_performance_details.csv ; all ; - ; count]**), supplemented by Will Ashcroft (2022 pick 2 — web source: Draftguru) arriving just in time for the dynasty's peak.

**Geelong's sub-pick-5 dynasty.** Here is the first wrinkle: Geelong won 4 premierships **[data: data/matches/ ; round_num==Grand Final & (team_1_team_name==Geelong OR team_2_team_name==Geelong) ; year ; count — years 2007,2009,2011,2022]** and neither of their dynasty cornerstones was a top-5 pick. Joel Selwood came in at pick 7 in the 2006 draft; Jimmy Bartel at pick 8 in 2001 — both web-sourced from Draftguru. Selwood played 355 games **[data: data/player_data/selwood_joel_26051988_performance_details.csv ; all ; - ; count]**, Bartel 305 **[data: data/player_data/bartel_jimmy_04121983_performance_details.csv ; all ; - ; count]**. The Cats' dynasty was completed by mid-round picks — not pick-1 hauls.

---

## The paradox: most top picks, fewest flags

Here the theory collapses entirely.

**Carlton's three consecutive number-1s — and nothing.** Marc Murphy (2005 pick 1), Bryce Gibbs (2006 pick 1), Matthew Kreuzer (2007 pick 1) — all web-sourced from Draftguru. Three consecutive overall-first selections in a row. Murphy played 300 games **[data: data/player_data/murphy_marc_24011988_performance_details.csv ; all ; - ; count]**, Gibbs 268 **[data: data/player_data/gibbs_bryce_05051989_performance_details.csv ; all ; - ; count]**. Both good, long careers. Carlton's last premiership: 1995. Three #1 picks bought them nothing.

**The expansion-club experiment.** The AFL gifted Gold Coast and GWS early selections on a massive scale — effectively running a controlled experiment. Gold Coast drafted picks 1, 2, 3, 4, 7, 9, and 10 in 2010 alone (Swallow, Bennell, Day, Gaff, Caddy, Prestia, Gorringe — web source: Draftguru). GWS held 8 of the top 10 selections in 2011 (Patton, Coniglio, Tyson, Hoskin-Elliott, Buntine, Haynes, Tomlinson, Sumner — Draftguru). Add further top-3 picks in subsequent years and the Giants accumulated arguably the largest single-club stockpile of elite-range picks in AFL history. Premierships won by Gold Coast between 2011 and 2025: **0**. Premierships won by GWS across the same window: **0**.

---

## The single-draft-class effect

The most underrated insight in all of this: **a single exceptional draft class compounds.** Hawthorn didn't just collect top-10 picks — they collected three from the *same year*, so those players entered the system together, developed their chemistry across a decade, and peaked simultaneously. The 2004 Hawthorn draft may be the single most consequential 24 hours in premiership history.

Contrast that with Carlton's three consecutive #1 picks arriving in three separate cohorts, each needing to develop in an environment without the others, each peaking at a slightly different time. The picks were as good on paper. The cohort effect was absent.

---

## The five list management principles the data actually supports

| # | Principle | Evidence |
|---|---|---|
| 1 | **You need 3–5 top-10 picks as cornerstones** | Every modern dynasty has them; no dynasty was built without any |
| 2 | **One great single-year class beats five isolated #1 picks** | Hawthorn 2004 (3 picks, 4 flags) vs Carlton 2005–07 (3 picks, 0 flags) |
| 3 | **Pick 7 can win you more flags than pick 1** | Selwood P7: 4 flags; Bartel P8: 3 flags; Murphy P1: 0 flags |
| 4 | **Keep them** | Every dynasty retained their core; GWS and Gold Coast traded and lost key picks |
| 5 | **Top-10 picks are necessary, not sufficient** | GWS + Gold Coast prove quantity of selections alone does not build premierships |

---

## The verdict

Top-10 draft picks are a prerequisite for sustained premiership success — but only if you get the chemistry right. The clubs that have dominated since 2000 found 3–5 top-10 selections, concentrated them in 1–2 draft years, retained them through their development years, and supplemented with smart mid-round work. The clubs that treated top picks as an end in themselves — collecting them in isolation, trading them when patience ran out, never building a cohort — have nothing to show for the capital spent.

List management is not about hoarding picks. It's about timing, cohort density, and patience.

---

## Methodology

**Draft pick data (picks 1–10 per year, player names, clubs):** web-sourced from Draftguru (draftguru.com.au), AFL.com.au draft history, and Wikipedia AFL draft articles. This data is **not** present in this repo — draft pick numbers are not in the player CSV files.

**Premiership results:** derived from `data/matches/matches_YYYY.csv` files in this repo, filtering on `round_num == "Grand Final"`. The 2010 Grand Final original match ended in a draw (Collingwood 68, St Kilda 68); the replay result (Collingwood winners) is taken from historical record rather than the repo edge case.

**Career games:** counted as rows in each player's `data/player_data/<name>_performance_details.csv` file — one row per game played.

**Flag counts 2000–2025 (verified from repo):**
Brisbane Lions 5 **[data: data/matches/ ; round_num==Grand Final & winner==Brisbane Lions ; year ; count]** · Geelong 4 · Hawthorn 4 · Richmond 3 · Sydney 2 · West Coast 2 · Collingwood 2 · Melbourne 1 · Western Bulldogs 1 · Port Adelaide 1 · Essendon 1
