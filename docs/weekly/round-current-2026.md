# Round 9 cheat sheet - 2026 (experimental v0)

> [← Back to fan landing page](../start-here-no-code.md) | [← Back to main README](../../README.md)

> **Status: experimental v0.** Predictions are model output, not certainties - typical error is ±4 disposals. Always cross-check team lists before lockout.

*Source: `data/prediction/next_round_9_prediction_20260430_2322.csv` (411 player rows, mean predicted disposals 15.88, range 8.4-29.4).*

---

## Top 30 predicted disposal leaders - Round 9, 2026

| Rank | Player | Team | Predicted disposals |
|------|--------|------|--------------------:|
| 1 | Daicos Nick | Collingwood | 29.4 |
| 2 | Callaghan Finn | Greater Western Sydney | 29.2 |
| 3 | Sheezel Harry | North Melbourne | 28.9 |
| 4 | Sinclair Jack | St Kilda | 28.7 |
| 5 | Roberts Archie | Essendon | 28.4 |
| 6 | Butters Zak | Port Adelaide | 28.4 |
| 7 | Smith Bailey | Geelong | 28.3 |
| 8 | Ash Lachie | Greater Western Sydney | 28.3 |
| 9 | Oliver Clayton | Greater Western Sydney | 28.0 |
| 10 | Holmes Max | Geelong | 27.3 |
| 11 | Whitfield Lachie | Greater Western Sydney | 27.2 |
| 12 | Walsh Sam | Carlton | 26.8 |
| 13 | Daicos Josh | Collingwood | 26.2 |
| 14 | Merrett Zach | Essendon | 26.2 |
| 15 | Miller Touk | Gold Coast | 25.8 |
| 16 | Neale Lachie | Brisbane Lions | 25.7 |
| 17 | Richards Ed | Western Bulldogs | 25.4 |
| 18 | Ashcroft Will | Brisbane Lions | 25.3 |
| 19 | Bontempelli Marcus | Western Bulldogs | 25.3 |
| 20 | Wanganeen-Milera Nasiah | St Kilda | 25.2 |
| 21 | Petracca Christian | Gold Coast | 25.2 |
| 22 | Mcinerney Justin | Sydney | 25.2 |
| 23 | Kennedy Matthew | Western Bulldogs | 25.1 |
| 24 | Davies-Uniacke Luke | North Melbourne | 25.1 |
| 25 | Serong Caleb | Fremantle | 25.1 |
| 26 | Wilkie Callum | St Kilda | 25.1 |
| 27 | Sanders Ryley | Western Bulldogs | 24.7 |
| 28 | Flanders Sam | St Kilda | 24.4 |
| 29 | Newcombe Jai | Hawthorn | 24.4 |
| 30 | Worrell Josh | Adelaide | 24.3 |

---

## Top 3 per club

The model's three highest-predicted players from each club. Useful for trade-target sanity checks.

### Adelaide
1. Worrell Josh - 24.3
2. Milera Wayne - 24.1
3. Dawson Jordan - 23.6

### Brisbane Lions
1. Neale Lachie - 25.7
2. Ashcroft Will - 25.3
3. Wilmot Darcy - 20.4

### Carlton
1. Walsh Sam - 26.8
2. Cripps Patrick - 23.1
3. Smith Jagga - 22.0

### Collingwood
1. Daicos Nick - 29.4
2. Daicos Josh - 26.2
3. Houston Dan - 23.9

### Essendon
1. Roberts Archie - 28.4
2. Merrett Zach - 26.2
3. Tsatas Elijah - 24.2

### Fremantle
1. Serong Caleb - 25.1
2. Clark Jordan - 23.6
3. Reid Murphy - 22.7

### Geelong
1. Smith Bailey - 28.3
2. Holmes Max - 27.3
3. Bruhn Tanner - 22.3

### Gold Coast
1. Miller Touk - 25.8
2. Petracca Christian - 25.2
3. Uwland Bodhi - 23.6

### Greater Western Sydney
1. Callaghan Finn - 29.2
2. Ash Lachie - 28.3
3. Oliver Clayton - 28.0

### Hawthorn
1. Newcombe Jai - 24.4
2. Impey Jarman - 22.0
3. Ward Josh - 20.9

### Melbourne
1. Steele Jack - 23.9
2. Pickett Kysaiah - 23.3
3. Gawn Max - 18.8

### North Melbourne
1. Sheezel Harry - 28.9
2. Davies-Uniacke Luke - 25.1
3. Daniel Caleb - 23.8

### Port Adelaide
1. Butters Zak - 28.4
2. Horne-Francis Jason - 21.2
3. Farrell Kane - 20.0

### Richmond
1. Ross Jack - 21.5
2. Hopper Jacob - 19.6
3. Prestia Dion - 19.6

### St Kilda
1. Sinclair Jack - 28.7
2. Wanganeen-Milera Nasiah - 25.2
3. Wilkie Callum - 25.1

### Sydney
1. Mcinerney Justin - 25.2
2. Blakey Nick - 23.4
3. Sheldrick Angus - 22.4

### West Coast
1. Kelly Tim - 22.4
2. Reid Harley - 21.9
3. Mccarthy Tom - 21.7

### Western Bulldogs
1. Richards Ed - 25.4
2. Bontempelli Marcus - 25.3
3. Kennedy Matthew - 25.1

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

*Last generated: 2026-05-10T00:19:39+00:00*
