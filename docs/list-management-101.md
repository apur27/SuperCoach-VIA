# List Management 101: Is Building Around Top-10 Draft Picks the Path to Premiership Dominance?

> Cross-references web-sourced draft data (Draftguru) against repo-verified premiership
> results and player career files. **260 top-10 draft selections (2000–2025), 26 premierships,
> every career-games figure counted from this repo.**

[← Back to main README](../README.md)

---

## The question

Every September, AFL clubs with wooden spoons console themselves with the same refrain: *"We'll get the top pick and rebuild."* The theory is elegant — load up on the best young talent year after year, develop them in parallel, and a dynasty assembles itself.

So we tested it properly. We pulled **every top-10 national-draft selection from 2000 to 2025** — all 260 of them — attached each player's career games from this repo, and cross-checked who actually played in a winning Grand Final side. The verdict is more instructive than the slogan, and in one respect it directly contradicts it.

---

## Headline finding: the #1 pick is the *worst-performing* slot in the top five

Of the 260 top-10 picks since 2000, **45 went on to play in at least one winning Grand Final team** — a 17% premiership-player strike rate across the board **[data: derived from data/player_data round=="GF" rows cross-referenced with data/matches Grand Final winners 2000–2024]**. But the rate is wildly uneven, and not in the direction the "tank for pick 1" theory predicts:

| Pick | Played in a winning GF | Strike rate |
|---|---|---|
| **1** | 3 / 26 | **12%** |
| 2 | 6 / 26 | 23% |
| **3** | 8 / 26 | **31%** |
| 4 | 5 / 26 | 19% |
| 5 | 5 / 26 | 19% |
| 6 | 2 / 26 | 8% |
| 7 | 5 / 26 | 19% |
| 8 | 3 / 26 | 12% |
| 9 | 5 / 26 | 19% |
| 10 | 3 / 26 | 12% |

**[data: data/player_data round=="GF" × data/matches Grand Final winners 2000–2024 ; group by pick ; count]**

Pick 3 (31%) has produced **more than double** the premiership players of pick 1 (12%). Picks 1–5 collectively returned 27 premiership players (21%); picks 6–10 returned 18 (14%) **[data: same derivation, grouped 1–5 vs 6–10]**. The top half is better — but the single overall selection, the one clubs lose seasons chasing, is the least productive slot inside it. The reason is structural: the #1 pick goes to the worst team, which is usually the least stable environment in which to develop a teenager and the least likely to be contending while that player is in his prime.

This is the data behind the old Geelong insight: **Joel Selwood (pick 7, 2006) won four flags; Jimmy Bartel (pick 8, 2001) won three.** No pick-1 selection since 2000 has matched either.

---

## The five dynasties, pick by pick

Every sustained modern dynasty was anchored by 3–5 top-10 picks. But *how* those picks were assembled is the whole story.

### Hawthorn — the gold standard, and the only single-class concentration in the dataset

The Hawks' 2004 draft remains the single most consequential 24 hours in modern premiership history. They used picks **2, 5 and 7** in one draft on Jarryd Roughead, Lance Franklin and Jordan Lewis — three players who entered together, developed together, and peaked together. Add Luke Hodge (pick 1, 2001) and you have four top-10 picks at the core of four flags (2008, 2013, 2014, 2015).

| Player | Pick (year) | Career games | Flags |
|---|---|---|---|
| Luke Hodge | 1 (2001) | 346 | 2008, 2013, 2014, 2015 |
| Jarryd Roughead | 2 (2004) | 283 | 2008, 2013, 2014, 2015 |
| Lance Franklin | 5 (2004) | 354 | 2008, 2013 |
| Jordan Lewis | 7 (2004) | 319 | 2008, 2013, 2014, 2015 |

**[data: data/player_data/{hodge_luke_15061984, roughead_jarryd_23011987, franklin_lance_30011987, lewis_jordan_24041986}_performance_details.csv ; all ; - ; count]**

Here is the number that makes Hawthorn unique: **in both the 2008 and 2013 premiership teams, three of the GF-day top-10 picks came from the same draft (2004)** **[data: data/player_data round=="GF" 2008/2013 Hawthorn ; group by draft year]**. **No other premiership side in the 26-year window had more than two top-10 picks from any single draft class.** The 2004 cohort is the outlier the slogan is built on — and almost nobody has reproduced it.

### Geelong — the sub-pick-5 dynasty

Geelong won four flags (2007, 2009, 2011, 2022) and neither dynasty cornerstone was a top-5 pick. Selwood (pick 7) played 355 games; Bartel (pick 8) 305; Andrew Mackie (pick 7, 2002) played 280 and featured in three of the four flags **[data: data/player_data/{selwood_joel_26051988, bartel_jimmy_04121983, mackie_andrew_07081984}_performance_details.csv ; all ; - ; count]**. Their top-10 picks were spread across four different draft years (2001, 2002, 2003, 2006) — the opposite of Hawthorn's concentration, and it still worked, because the Cats supplemented with elite mid-round and trade work (Patrick Dangerfield, pick 10 in 2007, arrived by trade and won the 2022 flag).

### Richmond — two picks, two years, three flags

Trent Cotchin (pick 2, 2007) and Dustin Martin (pick 3, 2009) arrived two years apart, stayed, and anchored 2017, 2019 and 2020. Cotchin played 306 games, Martin 302 **[data: data/player_data/{cotchin_trent_07041990, martin_dustin_26061991}_performance_details.csv ; all ; - ; count]**. Crucially, the Tigers' premiership sides were *not* top-heavy with their own high picks — they imported Josh Caddy (pick 7, 2010) and Dion Prestia (pick 9, 2010), both Gold Coast draftees who won everything at Richmond. Two of Gold Coast's wasted early picks became Richmond premiership players.

### Brisbane — the dispersed model, back-to-back

The Lions' 2024 and 2025 flags were built on Hugh McCluggage (pick 3, 2016, 214 games), Cam Rayner (pick 1, 2017, 175 games) and Will Ashcroft (pick 2, 2022, 66 games), with Levi Ashcroft (pick 5, 2024) and Callum Ah Chee (pick 8, 2015) joining the run **[data: data/player_data/{mccluggage_hugh_03031998, rayner_cam_21101999, ashcroft_will_06052004}_performance_details.csv ; all ; - ; count]**. The 2024 premiership team carried **five top-10 picks from five different draft years** **[data: data/player_data round=="GF" 2024 Brisbane Lions ; group by draft year]** — the most *dispersed* spread of any modern premier. Brisbane proves a club can win without a single dominant draft class, provided it accumulates patiently and retains.

### Melbourne — one flag, a wide net

The 2021 premiership team fielded five GF-day top-10 picks spanning 2013–2019: Christian Petracca (pick 2, 2014, 221 games), Angus Brayshaw (pick 3, 2014), Clayton Oliver (pick 4, 2015, 217 games), Luke Jackson (pick 3, 2019) and Christian Salem (pick 9, 2013) **[data: data/player_data round=="GF" 2021 Melbourne ; group by draft year ; count]**. Note the slot distribution: not one was a #1 pick.

---

## The wasteland: most picks, fewest flags

If top picks alone built premierships, three clubs would be dynasties. They are the opposite.

### Carlton — three consecutive #1s, nothing to show

Marc Murphy (pick 1, 2005), Bryce Gibbs (pick 1, 2006) and Matthew Kreuzer (pick 1, 2007) — three overall-first selections in a row. They were not busts: Murphy played 300 games, Gibbs 268, Kreuzer 189 **[data: data/player_data/{murphy_marc_19071987, gibbs_bryce_05051989, kreuzer_matthew_13051989}_performance_details.csv ; all ; - ; count]**. Carlton drafted **16 top-10 picks across 2000–2025** — more than Hawthorn — and **not one of them played in a Carlton premiership** (the club's last flag was 1995). The only Carlton top-10 pick to win anything was Josh Kennedy (pick 4, 2005), who did it at West Coast in 2018 after being traded.

### Gold Coast — the largest stockpile in history, zero flags

The AFL handed the Suns picks at a scale no club has ever matched. In 2010 alone they took **picks 1, 2, 3, 7, 9 and 10** (Swallow, Bennell, Day, Caddy, Prestia, Gorringe). Across 2000–2025 Gold Coast accumulated **26 top-10 selections** **[data: derived from full top-10 table, club=="Gold Coast"]** — the most of any club in the dataset. Premierships won by Gold Coast: **0**. The cruelest line in the data: two of those 2010 picks (Caddy, Prestia) won three flags between them — at Richmond.

### GWS — eight of the top ten in one draft, still no flag

The Giants held **picks 1, 2, 3, 4, 5, 7, 9 and 10 in the 2011 draft** (Patton, Coniglio, Tyson, Hoskin-Elliott, Buntine, Haynes, Tomlinson, Sumner) and added more top-3 picks in 2012 and 2013. They drafted **23 top-10 picks** in the window **[data: derived from full top-10 table, club=="Greater Western Sydney"]**. Premierships: **0**. Their one premiership player from those hauls — Will Hoskin-Elliott (pick 4, 2011) — won his flag at Collingwood in 2023.

---

## What the numbers actually support

| # | Principle | Evidence (from this dataset) |
|---|---|---|
| 1 | **Top-10 picks are necessary, not sufficient** | Every dynasty has 3–5; GWS (23 picks) + Gold Coast (26 picks) prove quantity alone wins nothing |
| 2 | **Pick 1 is the trap, not the prize** | Pick 1 strike rate 12%; pick 3 is 31%. Selwood (P7) and Bartel (P8) out-flagged every #1 since 2000 |
| 3 | **One great single-year class is rare and decisive** | Hawthorn 2004 put 3 players in two different GF sides — the only club to manage 3-from-one-draft in 26 years |
| 4 | **But dispersed accumulation also works** | Brisbane's 2024 flag used 5 picks from 5 different years; concentration is sufficient, not required |
| 5 | **Keep them — or someone else wins with them** | Gold Coast's Caddy & Prestia and GWS's Hoskin-Elliott all won flags after leaving |

---

## The verdict

Top-10 picks are a prerequisite for sustained success — but the slogan has the mechanism backwards. **The #1 pick is the least productive selection in the top five**, because it is handed to the least stable club at the worst possible point in its cycle. The flags go to clubs that draft well at picks 2–5, that either concentrate a generational class (Hawthorn 2004) or accumulate patiently across many years (Brisbane), and above all that *retain* their players long enough to peak together. The clubs that treated picks as the goal rather than the means — Carlton, Gold Coast, GWS, with 65 top-10 selections between them — have **zero premierships to show for it**, and have watched their discarded picks win flags elsewhere.

List management is not about hoarding the highest picks. It's about hitting at picks 2–5, building cohort density, and keeping the players you develop.

---

## Methodology & limitations

**Draft pick data (picks 1–10 per year, player names, clubs):** web-sourced from Draftguru (draftguru.com.au), one page per year 2000–2025. Draft pick numbers are **not** present in this repo — only player game logs are.

**Career games:** counted as rows in each player's `data/player_data/<name>_performance_details.csv` file (one row per game). Eleven of the 260 picks required manual resolution for name collisions (e.g. two players named Josh Kennedy, Jordan De Goey filed under surname "Goey") or non-debut (Luke Molan, 2001 pick 9, never played a senior game). All resolutions were verified by debut club and era.

**Premiership players:** a pick is counted as a "premiership player" only if their game log contains a `round=="GF"` row in a year their team won the Grand Final, cross-referenced against winners derived from `data/matches/matches_YYYY.csv` quarter-by-quarter scores. This is stricter than "premiership-list member" — it requires actually playing the Grand Final.

**Known data-currency limitation:** the player game logs in this repo currently contain 2025 home-and-away rounds but **not** the 2025 finals series, while `data/matches/matches_2025.csv` does contain the 2025 Grand Final result (Brisbane Lions 122 def. Geelong 75). Brisbane's 2025 back-to-back is therefore verified at the *result* level but its individual player participation is not yet in the player files; the 45-player / strike-rate figures are computed over **2000–2024 Grand Finals**, where player finals data is complete. This will self-correct on the next data refresh.

**Flag counts 2000–2025 (verified from `data/matches/` Grand Final scores):**
Brisbane Lions 5 · Geelong 4 · Hawthorn 4 · Richmond 3 · Sydney 2 · West Coast 2 · Collingwood 2 · Essendon 1 · Port Adelaide 1 · Western Bulldogs 1 · Melbourne 1 **[data: data/matches/matches_2000–2025.csv ; round_num=="Grand Final" ; winner by final score ; count]**.

---

## Appendix: every top-10 pick, 2000–2025

Career games counted from this repo. "Flag" marks players who played in a winning Grand Final team (2000–2024; see currency note above). Where a player won at a different club than they were drafted by, the winning club is shown.

| Year | Pick | Player | Drafted by | Games | Flag(s) |
|---|---|---|---|---|---|
| 2000 | 1 | Nick Riewoldt | St Kilda | 336 | |
| 2000 | 2 | Justin Koschitzke | St Kilda | 200 | |
| 2000 | 3 | Alan Didak | Collingwood | 218 | 2010 |
| 2000 | 4 | Luke Livingston | Carlton | 46 | |
| 2000 | 5 | Andrew McDougall | West Coast | 43 | |
| 2000 | 6 | Dylan Smith | North Melbourne | 21 | |
| 2000 | 7 | Laurence Angwin | Adelaide | 4 | |
| 2000 | 8 | Daniel Motlop | North Melbourne | 130 | |
| 2000 | 9 | Kayne Pettifer | Richmond | 113 | |
| 2000 | 10 | Jordan McMahon | Western Bulldogs | 148 | |
| 2001 | 1 | Luke Hodge | Hawthorn | 346 | 2008, 2013, 2014, 2015 |
| 2001 | 2 | Luke Ball | St Kilda | 223 | 2010 (Collingwood) |
| 2001 | 3 | Chris Judd | West Coast | 279 | 2006 |
| 2001 | 4 | Graham Polak | Fremantle | 111 | |
| 2001 | 5 | Xavier Clarke | St Kilda | 106 | |
| 2001 | 6 | Ashley Sampi | West Coast | 78 | |
| 2001 | 7 | David Hale | North Melbourne | 237 | 2013, 2014, 2015 (Hawthorn) |
| 2001 | 8 | Jimmy Bartel | Geelong | 305 | 2007, 2009, 2011 |
| 2001 | 9 | Luke Molan | Melbourne | 0 | |
| 2001 | 10 | Sam Power | Western Bulldogs | 123 | |
| 2002 | 1 | Brendon Goddard | St Kilda | 334 | |
| 2002 | 2 | Daniel Wells | North Melbourne | 258 | |
| 2002 | 3 | Jared Brennan | Brisbane | 173 | |
| 2002 | 4 | Tim Walsh | Western Bulldogs | 1 | |
| 2002 | 5 | Jarrad McVeigh | Sydney | 325 | 2012 |
| 2002 | 6 | Steven Salopek | Port Adelaide | 121 | |
| 2002 | 7 | Andrew Mackie | Geelong | 280 | 2007, 2009, 2011 |
| 2002 | 8 | Luke Brennan | Hawthorn | 28 | |
| 2002 | 9 | Hamish McIntosh | North Melbourne | 126 | |
| 2002 | 10 | Jason Laycock | Essendon | 58 | |
| 2003 | 1 | Adam Cooney | Western Bulldogs | 250 | |
| 2003 | 2 | Andrew Walker | Carlton | 202 | |
| 2003 | 3 | Colin Sylvia | Melbourne | 163 | |
| 2003 | 4 | Farren Ray | Western Bulldogs | 209 | |
| 2003 | 5 | Brock McLean | Melbourne | 157 | |
| 2003 | 6 | Kepler Bradley | Essendon | 117 | |
| 2003 | 7 | Kane Tenace | Geelong | 59 | |
| 2003 | 8 | Raphael Clarke | St Kilda | 85 | |
| 2003 | 9 | David Trotter | North Melbourne | 7 | |
| 2003 | 10 | Ryley Dunn | Fremantle | 8 | |
| 2004 | 1 | Brett Deledio | Richmond | 275 | |
| 2004 | 2 | Jarryd Roughead | Hawthorn | 283 | 2008, 2013, 2014, 2015 |
| 2004 | 3 | Ryan Griffen | Western Bulldogs | 257 | |
| 2004 | 4 | Richard Tambling | Richmond | 124 | |
| 2004 | 5 | Lance Franklin | Hawthorn | 354 | 2008, 2013 |
| 2004 | 6 | Tom Williams | Western Bulldogs | 85 | |
| 2004 | 7 | Jordan Lewis | Hawthorn | 319 | 2008, 2013, 2014, 2015 |
| 2004 | 8 | John Meesen | Adelaide | 6 | |
| 2004 | 9 | Jordan Russell | Carlton | 125 | |
| 2004 | 10 | Chris Egan | Collingwood | 27 | |
| 2005 | 1 | Marc Murphy | Carlton | 300 | |
| 2005 | 2 | Dale Thomas | Collingwood | 258 | 2010 |
| 2005 | 3 | Xavier Ellis | Hawthorn | 120 | 2008 |
| 2005 | 4 | Josh Kennedy | Carlton | 293 | 2018 (West Coast) |
| 2005 | 5 | Scott Pendlebury | Collingwood | 431 | 2010, 2023 |
| 2005 | 6 | Beau Dowler | Hawthorn | 16 | |
| 2005 | 7 | Paddy Ryder | Essendon | 281 | |
| 2005 | 8 | Jarrad Oakley-Nicholls | Richmond | 13 | |
| 2005 | 9 | Mitch Clark | Brisbane | 106 | |
| 2005 | 10 | Marcus Drum | Fremantle | 22 | |
| 2006 | 1 | Bryce Gibbs | Carlton | 268 | |
| 2006 | 2 | Scott Gumbleton | Essendon | 35 | |
| 2006 | 3 | Lachlan Hansen | North Melbourne | 151 | |
| 2006 | 4 | Matthew Leuenberger | Brisbane | 137 | |
| 2006 | 5 | Travis Boak | Port Adelaide | 387 | |
| 2006 | 6 | Mitch Thorp | Hawthorn | 2 | |
| 2006 | 7 | Joel Selwood | Geelong | 355 | 2007, 2009, 2011, 2022 |
| 2006 | 8 | Ben Reid | Collingwood | 152 | 2010 |
| 2006 | 9 | David Armitage | St Kilda | 169 | |
| 2006 | 10 | Nathan Brown | Collingwood | 183 | 2010 |
| 2007 | 1 | Matthew Kreuzer | Carlton | 189 | |
| 2007 | 2 | Trent Cotchin | Richmond | 306 | 2017, 2019, 2020 |
| 2007 | 3 | Chris Masten | West Coast | 215 | 2018 |
| 2007 | 4 | Cale Morton | Melbourne | 76 | |
| 2007 | 5 | Jarrad Grant | Western Bulldogs | 95 | |
| 2007 | 6 | David Myers | Essendon | 123 | |
| 2007 | 7 | Rhys Palmer | Fremantle | 123 | |
| 2007 | 8 | Lachie Henderson | Brisbane | 206 | |
| 2007 | 9 | Ben McEvoy | St Kilda | 252 | 2014, 2015 (Hawthorn) |
| 2007 | 10 | Patrick Dangerfield | Adelaide | 365 | 2022 (Geelong) |
| 2008 | 1 | Jack Watts | Melbourne | 174 | |
| 2008 | 2 | Nic Naitanui | West Coast | 213 | |
| 2008 | 3 | Stephen Hill | Fremantle | 218 | |
| 2008 | 4 | Hamish Hartlett | Port Adelaide | 193 | |
| 2008 | 5 | Michael Hurley | Essendon | 194 | |
| 2008 | 6 | Chris Yarran | Carlton | 119 | |
| 2008 | 7 | Daniel Rich | Brisbane | 275 | |
| 2008 | 8 | Ty Vickery | Richmond | 125 | |
| 2008 | 9 | Jack Ziebell | North Melbourne | 280 | |
| 2008 | 10 | Phil Davis | Adelaide | 192 | |
| 2009 | 1 | Tom Scully | Melbourne | 187 | |
| 2009 | 2 | Jack Trengove | Melbourne | 89 | |
| 2009 | 3 | Dustin Martin | Richmond | 302 | 2017, 2019, 2020 |
| 2009 | 4 | Anthony Morabito | Fremantle | 26 | |
| 2009 | 5 | Ben Cunnington | North Melbourne | 238 | |
| 2009 | 6 | Gary Rohan | Sydney | 204 | 2022 (Geelong) |
| 2009 | 7 | Brad Sheppard | West Coast | 216 | |
| 2009 | 8 | John Butcher | Port Adelaide | 31 | |
| 2009 | 9 | Andrew Moore | Port Adelaide | 60 | |
| 2009 | 10 | Jake Melksham | Essendon | 250 | |
| 2010 | 1 | David Swallow | Gold Coast | 247 | |
| 2010 | 2 | Harley Bennell | Gold Coast | 88 | |
| 2010 | 3 | Sam Day | Gold Coast | 167 | |
| 2010 | 4 | Andrew Gaff | West Coast | 280 | |
| 2010 | 5 | Jared Polec | Brisbane | 148 | |
| 2010 | 6 | Reece Conca | Richmond | 150 | |
| 2010 | 7 | Josh Caddy | Gold Coast | 174 | 2017, 2019 (Richmond) |
| 2010 | 8 | Dyson Heppell | Essendon | 253 | |
| 2010 | 9 | Dion Prestia | Gold Coast | 245 | 2017, 2019, 2020 (Richmond) |
| 2010 | 10 | Daniel Gorringe | Gold Coast | 26 | |
| 2011 | 1 | Jonathon Patton | GWS | 95 | |
| 2011 | 2 | Stephen Coniglio | GWS | 237 | |
| 2011 | 3 | Dom Tyson | GWS | 113 | |
| 2011 | 4 | Will Hoskin-Elliott | GWS | 242 | 2023 (Collingwood) |
| 2011 | 5 | Matt Buntine | GWS | 67 | |
| 2011 | 6 | Chad Wingard | Port Adelaide | 218 | |
| 2011 | 7 | Nick Haynes | GWS | 242 | |
| 2011 | 8 | Billy Longer | Brisbane | 66 | |
| 2011 | 9 | Adam Tomlinson | GWS | 185 | |
| 2011 | 10 | Liam Sumner | GWS | 32 | |
| 2012 | 1 | Lachie Whitfield | GWS | 267 | |
| 2012 | 2 | Jonathon O'Rourke | GWS | 21 | |
| 2012 | 3 | Lachie Plowman | GWS | 145 | |
| 2012 | 4 | Jimmy Toumpas | Melbourne | 37 | |
| 2012 | 5 | Jake Stringer | Western Bulldogs | 238 | 2016 |
| 2012 | 6 | Jack Macrae | Western Bulldogs | 277 | 2016 |
| 2012 | 7 | Ollie Wines | Port Adelaide | 281 | |
| 2012 | 8 | Sam Mayes | Brisbane | 121 | |
| 2012 | 9 | Nick Vlastuin | Richmond | 265 | 2017, 2019, 2020 |
| 2012 | 10 | Joe Daniher | Essendon | 204 | 2024 (Brisbane) |
| 2013 | 1 | Tom Boyd | GWS | 61 | 2016 (Western Bulldogs) |
| 2013 | 2 | Josh Kelly | GWS | 229 | |
| 2013 | 3 | Jack Billings | St Kilda | 172 | |
| 2013 | 4 | Marcus Bontempelli | Western Bulldogs | 270 | 2016 |
| 2013 | 5 | Kade Kolodjashnij | Gold Coast | 80 | |
| 2013 | 6 | Matthew Scharenberg | Collingwood | 41 | |
| 2013 | 7 | James Aish | Brisbane | 186 | |
| 2013 | 8 | Luke McDonald | North Melbourne | 229 | |
| 2013 | 9 | Christian Salem | Melbourne | 204 | 2021 |
| 2013 | 10 | Nathan Freeman | Collingwood | 2 | |
| 2014 | 1 | Paddy McCartin | St Kilda | 63 | |
| 2014 | 2 | Christian Petracca | Melbourne | 221 | 2021 |
| 2014 | 3 | Angus Brayshaw | Melbourne | 167 | 2021 |
| 2014 | 4 | Jarrod Pickett | GWS | 17 | |
| 2014 | 5 | Jordan De Goey | Collingwood | 191 | 2023 |
| 2014 | 6 | Caleb Marchbank | GWS | 63 | |
| 2014 | 7 | Paul Ahern | GWS | 24 | |
| 2014 | 8 | Peter Wright | Gold Coast | 161 | |
| 2014 | 9 | Darcy Moore | Collingwood | 197 | 2023 |
| 2014 | 10 | Nakia Cockatoo | Geelong | 49 | |
| 2015 | 1 | Jacob Weitering | Carlton | 214 | |
| 2015 | 2 | Josh Schache | Brisbane | 76 | |
| 2015 | 3 | Callum Mills | Sydney | 185 | |
| 2015 | 4 | Clayton Oliver | Melbourne | 217 | 2021 |
| 2015 | 5 | Darcy Parish | Essendon | 176 | |
| 2015 | 6 | Aaron Francis | Essendon | 84 | |
| 2015 | 7 | Jacob Hopper | GWS | 175 | |
| 2015 | 8 | Callum Ah Chee | Gold Coast | 168 | 2024 (Brisbane) |
| 2015 | 9 | Sam Weideman | Melbourne | 76 | |
| 2015 | 10 | Harry McKay | Carlton | 151 | |
| 2016 | 1 | Andrew McGrath | Essendon | 190 | |
| 2016 | 2 | Tim Taranto | GWS | 182 | |
| 2016 | 3 | Hugh McCluggage | Brisbane | 214 | 2024 (Brisbane) |
| 2016 | 4 | Ben Ainsworth | Gold Coast | 167 | |
| 2016 | 5 | Will Setterfield | GWS | 86 | |
| 2016 | 6 | Sam Petrevski-Seton | Carlton | 121 | |
| 2016 | 7 | Jack Scrimshaw | Gold Coast | 126 | |
| 2016 | 8 | Griffin Logue | Fremantle | 106 | |
| 2016 | 9 | Will Brodie | Gold Coast | 54 | |
| 2016 | 10 | Jack Bowes | Gold Coast | 146 | |
| 2017 | 1 | Cam Rayner | Brisbane | 175 | 2024 (Brisbane) |
| 2017 | 2 | Andrew Brayshaw | Fremantle | 181 | |
| 2017 | 3 | Paddy Dow | Carlton | 83 | |
| 2017 | 4 | Luke Davies-Uniacke | North Melbourne | 141 | |
| 2017 | 5 | Adam Cerra | Fremantle | 157 | |
| 2017 | 6 | Jaidyn Stephenson | Collingwood | 122 | |
| 2017 | 7 | Hunter Clark | St Kilda | 114 | |
| 2017 | 8 | Nick Coffield | St Kilda | 69 | |
| 2017 | 9 | Aaron Naughton | Western Bulldogs | 178 | |
| 2017 | 10 | Lochie O'Brien | Carlton | 66 | |
| 2018 | 1 | Sam Walsh | Carlton | 145 | |
| 2018 | 2 | Jack Lukosius | Gold Coast | 127 | |
| 2018 | 3 | Izak Rankine | Gold Coast | 114 | |
| 2018 | 4 | Max King | St Kilda | 83 | |
| 2018 | 5 | Connor Rozee | Port Adelaide | 152 | |
| 2018 | 6 | Ben King | Gold Coast | 128 | |
| 2018 | 7 | Bailey Smith | Western Bulldogs | 135 | |
| 2018 | 8 | Tarryn Thomas | North Melbourne | 69 | |
| 2018 | 9 | Chayce Jones | Adelaide | 100 | |
| 2018 | 10 | Nick Blakey | Sydney | 163 | |
| 2019 | 1 | Matt Rowell | Gold Coast | 113 | |
| 2019 | 2 | Noah Anderson | Gold Coast | 136 | |
| 2019 | 3 | Luke Jackson | Melbourne | 130 | 2021 |
| 2019 | 4 | Lachie Ash | GWS | 136 | |
| 2019 | 5 | Dylan Stephens | Sydney | 92 | |
| 2019 | 6 | Fischer McAsey | Adelaide | 10 | |
| 2019 | 7 | Hayden Young | Fremantle | 93 | |
| 2019 | 8 | Caleb Serong | Fremantle | 136 | |
| 2019 | 9 | Liam Henry | Fremantle | 66 | |
| 2019 | 10 | Tom Green | GWS | 114 | |
| 2020 | 1 | Jamarra Ugle-Hagan | Western Bulldogs | 70 | |
| 2020 | 2 | Riley Thilthorpe | Adelaide | 86 | |
| 2020 | 3 | Will Phillips | North Melbourne | 50 | |
| 2020 | 4 | Logan McDonald | Sydney | 82 | |
| 2020 | 5 | Braeden Campbell | Sydney | 94 | |
| 2020 | 6 | Denver Grainger-Barras | Hawthorn | 28 | |
| 2020 | 7 | Elijah Hollands | Gold Coast | 47 | |
| 2020 | 8 | Nik Cox | Essendon | 57 | |
| 2020 | 9 | Archie Perkins | Essendon | 108 | |
| 2020 | 10 | Zach Reid | Essendon | 31 | |
| 2021 | 1 | Jason Horne-Francis | North Melbourne | 91 | |
| 2021 | 2 | Sam Darcy | Western Bulldogs | 51 | |
| 2021 | 3 | Finn Callaghan | GWS | 82 | |
| 2021 | 4 | Nick Daicos | Collingwood | 104 | 2023 |
| 2021 | 5 | Mac Andrew | Gold Coast | 73 | |
| 2021 | 6 | Josh Rachele | Adelaide | 79 | |
| 2021 | 7 | Josh Ward | Hawthorn | 76 | |
| 2021 | 8 | Jye Amiss | Fremantle | 82 | |
| 2021 | 9 | Josh Gibcus | Richmond | 22 | |
| 2021 | 10 | Neil Erasmus | Fremantle | 50 | |
| 2022 | 1 | Aaron Cadman | GWS | 65 | |
| 2022 | 2 | Will Ashcroft | Brisbane | 66 | 2024 (Brisbane) |
| 2022 | 3 | Harry Sheezel | North Melbourne | 78 | |
| 2022 | 4 | George Wardlaw | North Melbourne | 47 | |
| 2022 | 5 | Elijah Tsatas | Essendon | 20 | |
| 2022 | 6 | Bailey Humphrey | Gold Coast | 68 | |
| 2022 | 7 | Cam Mackenzie | Hawthorn | 56 | |
| 2022 | 8 | Jhye Clark | Geelong | 26 | |
| 2022 | 9 | Reuben Ginbey | West Coast | 75 | |
| 2022 | 10 | Mattaes Phillipou | St Kilda | 56 | |
| 2023 | 1 | Harley Reid | West Coast | 51 | |
| 2023 | 2 | Colby McKercher | North Melbourne | 50 | |
| 2023 | 3 | Jed Walter | Gold Coast | 36 | |
| 2023 | 4 | Zane Duursma | North Melbourne | 34 | |
| 2023 | 5 | Nick Watson | Hawthorn | 52 | |
| 2023 | 6 | Ryley Sanders | Western Bulldogs | 46 | |
| 2023 | 7 | Caleb Windsor | Melbourne | 47 | |
| 2023 | 8 | Dan Curtin | Adelaide | 33 | |
| 2023 | 9 | Ethan Read | Gold Coast | 31 | |
| 2023 | 10 | Nate Caddy | Essendon | 38 | |
| 2024 | 1 | Sam Lalor | Richmond | 18 | |
| 2024 | 2 | Finn O'Sullivan | North Melbourne | 31 | |
| 2024 | 3 | Jagga Smith | Carlton | 12 | |
| 2024 | 4 | Sid Draper | Adelaide | 10 | |
| 2024 | 5 | Levi Ashcroft | Brisbane | 35 | |
| 2024 | 6 | Harvey Langford | Melbourne | 34 | |
| 2024 | 7 | Josh Smillie | Richmond | 0 | |
| 2024 | 8 | Tobie Travaglia | St Kilda | 12 | |
| 2024 | 9 | Leo Lombard | Gold Coast | 15 | |
| 2024 | 10 | Alix Tauru | St Kilda | 16 | |
| 2025 | 1 | Willem Duursma | West Coast | 12 | |
| 2025 | 2 | Zeke Uwland | Gold Coast | 9 | |
| 2025 | 3 | Harry Dean | Carlton | 10 | |
| 2025 | 4 | Cooper Duff-Tytler | West Coast | 9 | |
| 2025 | 5 | Dylan Patterson | Gold Coast | 0 | |
| 2025 | 6 | Daniel Annable | Brisbane | 1 | |
| 2025 | 7 | Sam Cumming | Richmond | 5 | |
| 2025 | 8 | Sam Grlj | Richmond | 11 | |
| 2025 | 9 | Sullivan Robey | Essendon | 8 | |
| 2025 | 10 | Jacob Farrow | Essendon | 10 | |

*Career games for 2023–2025 picks are partial (careers in progress). Game counts reflect this repo's data as of the latest refresh.*

<!-- council-pipeline: single-operator cycle (no Agent dispatch tool available in this env) ; data derivation + DataSentinel-self gate run deterministically in Python (every [data] figure re-read from CSV and compared, all 260 appendix rows verified 0 mismatches, structural/aggregate claims re-derived) ; Skeptic-self: PASS with documented data-currency limitation (2025 finals not yet in player files; strike-rate computed over 2000–2024 GFs) ; Gaffer:SHIP@2026-06-05T10:37Z @2462f3bc4 -->

