---
name: blank-counting-stat-means
description: Blank/NaN in a counting stat (tackles, marks, hit-outs) means "zero in a played game", not "data not recorded" — within the stat's recorded era. Use fill-zero for season means.
metadata:
  type: project
---

When computing a player's season mean for a counting stat (tackles, marks, hit-outs, contested marks, goals), a blank/NaN row in the modern recording era means **zero in a game the player actually played**, not "data not recorded" and not "did not play".

**Why:**
- In 2026, counting-stat columns NEVER contain an explicit `0` — they hold positive integers or NaN, nothing in between. Verified for tackles, marks, kicks, handballs, hit_outs across 336+ sampled 2026 rows. A genuine zero game has no other encoding available, so it is stored as NaN. This is a uniform feed convention: blank = zero.
- NaN-tackle rows are games that were played: across 600 files (1987+), of 2,976 NaN-tackle rows, 2,928 had disposals>0; only 48 were true absences (all skill stats NaN).
- `percentage_of_game_played` on 2026 blank-tackle rows: median 76%, mean 69%, IQR 64–86. Full games, not cameos or non-appearances. (A true non-appearance has NO row at all.)
- Example bias: Impey 2026 tackles — dropna mean 2.333 vs fill-zero 1.750 (33% overstatement). Sinclair 1.818 vs 1.667.

**How to apply:**
- For a counting stat WITHIN its recorded era, season mean = `df[stat].fillna(0).mean()` over the rows that exist (played games). NOT `dropna` — dropna inflates per-game averages for low-frequency stats by discarding the quietest games.
- Do NOT fill-zero outside the stat's recorded era — there NaN means "not collected league-wide" and zeros would be fabricated. Tackles pre-1987, clearances/contested-poss pre-1998: exclude the era instead. See [[data_stat_coverage_eras]].
- A true non-appearance is an absent row, so it never enters either denominator.
- Caveat: this is the feed's de-facto convention inferred from data, not a documented schema. Re-check if the upstream source starts emitting explicit zeros.
