# Round 20 cheat sheet - 2026 (experimental v0)

> [← Back to fan landing page](../start-here-no-code.md) | [← Back to main README](../../README.md)

> **Status: experimental v0.** Predictions are model output, not certainties - typical error is ±4 disposals. Always cross-check team lists before lockout.

*Source: `data/prediction/next_round_20_prediction_20260714_0730.csv` (413 player rows, mean predicted disposals 15.97, range 8.0-30.0).*

---

## Top 30 predicted disposal leaders - Round 20, 2026

| Rank | Player | Team | Predicted disposals |
|------|--------|------|--------------------:|
| 1 | Sheezel Harry | North Melbourne | 30.0 |
| 2 | Wanganeen-Milera Nasiah | St Kilda | 29.0 |
| 3 | Callaghan Finn | Greater Western Sydney | 29.0 |
| 4 | Ash Lachie | Greater Western Sydney | 28.0 |
| 5 | Daicos Nick | Collingwood | 27.0 |
| 6 | Smith Bailey | Geelong | 27.0 |
| 7 | Oliver Clayton | Greater Western Sydney | 27.0 |
| 8 | Ashcroft Will | Brisbane Lions | 27.0 |
| 9 | Holmes Max | Geelong | 27.0 |
| 10 | Walsh Sam | Carlton | 27.0 |
| 11 | Dale Bailey | Western Bulldogs | 26.0 |
| 12 | Brayshaw Andrew | Fremantle | 26.0 |
| 13 | Daniel Caleb | North Melbourne | 26.0 |
| 14 | Neale Lachie | Brisbane Lions | 26.0 |
| 15 | Anderson Noah | Gold Coast | 26.0 |
| 16 | Merrett Zach | Essendon | 26.0 |
| 17 | Pendlebury Scott | Collingwood | 25.0 |
| 18 | Richards Ed | Western Bulldogs | 25.0 |
| 19 | Daicos Josh | Collingwood | 25.0 |
| 20 | Parker Luke | North Melbourne | 25.0 |
| 21 | Roberts Archie | Essendon | 25.0 |
| 22 | Dawson Jordan | Adelaide | 25.0 |
| 23 | Kennedy Matthew | Western Bulldogs | 25.0 |
| 24 | Hill Bradley | St Kilda | 25.0 |
| 25 | Heeney Isaac | Sydney | 25.0 |
| 26 | Noble John | Gold Coast | 25.0 |
| 27 | Warner Chad | Sydney | 24.0 |
| 28 | Uwland Bodhi | Gold Coast | 24.0 |
| 29 | Mills Callum | Sydney | 24.0 |
| 30 | Blakey Nick | Sydney | 24.0 |

---

## Top 3 per club

The model's three highest-predicted players from each club. Useful for trade-target sanity checks.

### Adelaide
1. Dawson Jordan - 25.0
2. Laird Rory - 24.0
3. Milera Wayne - 23.0

### Brisbane Lions
1. Ashcroft Will - 27.0
2. Neale Lachie - 26.0
3. Zorko Dayne - 22.0

### Carlton
1. Walsh Sam - 27.0
2. Smith Jagga - 24.0
3. Newman Nic - 23.0

### Collingwood
1. Daicos Nick - 27.0
2. Pendlebury Scott - 25.0
3. Daicos Josh - 25.0

### Essendon
1. Merrett Zach - 26.0
2. Roberts Archie - 25.0
3. Parish Darcy - 23.0

### Fremantle
1. Brayshaw Andrew - 26.0
2. Reid Murphy - 23.0
3. Bolton Shai - 23.0

### Geelong
1. Smith Bailey - 27.0
2. Holmes Max - 27.0
3. Stewart Tom - 21.0

### Gold Coast
1. Anderson Noah - 26.0
2. Noble John - 25.0
3. Uwland Bodhi - 24.0

### Greater Western Sydney
1. Callaghan Finn - 29.0
2. Ash Lachie - 28.0
3. Oliver Clayton - 27.0

### Hawthorn
1. Newcombe Jai - 23.0
2. Ward Josh - 22.0
3. Amon Karl - 22.0

### Melbourne
1. Bowey Jake - 23.0
2. Pickett Kysaiah - 22.0
3. Fitzgerald Joel - 21.0

### North Melbourne
1. Sheezel Harry - 30.0
2. Daniel Caleb - 26.0
3. Parker Luke - 25.0

### Port Adelaide
1. Butters Zak - 24.0
2. Burgoyne Jase - 20.0
3. Byrne-Jones Darcy - 20.0

### Richmond
1. Taranto Tim - 22.0
2. Prestia Dion - 20.0
3. Banks Sam - 18.0

### St Kilda
1. Wanganeen-Milera Nasiah - 29.0
2. Hill Bradley - 25.0
3. Hall Max - 24.0

### Sydney
1. Heeney Isaac - 25.0
2. Warner Chad - 24.0
3. Mills Callum - 24.0

### West Coast
1. Mccarthy Tom - 23.0
2. Kelly Tim - 23.0
3. Reid Harley - 21.0

### Western Bulldogs
1. Dale Bailey - 26.0
2. Richards Ed - 25.0
3. Kennedy Matthew - 25.0

---

## Reading this cheat sheet

- **Predictions are disposals only**, not SuperCoach fantasy points.
- **Typical error: ±4 disposals** (see [backtest results](../afl-backtest-2026.md)).
- **The model is slow on role changes and tag jobs.**
- **Late outs are not handled.** Always check team lists.

For the full honest version, see [How to use this for SuperCoach](../how-to-use-this-for-supercoach.md).

---

## How this page is generated

Run `python scripts/generate_weekly_cheat_sheet.py` from the repo root. The script reads the most recent prediction CSV and writes this markdown.

*Last generated: 2026-07-13T21:36:43+00:00*
