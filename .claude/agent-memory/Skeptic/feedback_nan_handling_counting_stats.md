---
name: nan-handling-counting-stats
description: For per-game counting-stat means (tackles/marks), dropna beats fill-zero when blanks are recording gaps on games-played; demand denominator disclosure
metadata:
  type: feedback
---

When auditing a published per-game mean of a counting stat (tackles, marks, hit-outs) where some rows are blank/NaN, do not accept fill-zero as a safe default.

**Why:** In player_data CSVs, blank counting-stat rows are frequently genuine recording gaps on games the player demonstrably PLAYED, not did-not-play rows. Evidence: in `data/player_data/sinclair_jack_12021995_performance_details.csv`, all 20 blank-tackle rows had 8-34 disposals and 66-91% TOG — confirmed games played. fill-zero would assert a false "0 tackles" observation for each, systematically understating high-possession players. dropna only over-counts mildly when the blank fraction is small (1/12 ≈ 8% upward bias).

**The deciding input:** `.claude/agent-memory/Scientist/snapshot_data_quality.md` confirms tackles is a RELIABLE FanFooty field (zero mismatches in R10 2026 verification). So blank tackle rows are post-game recording gaps, NOT live-feed defaults-to-zero. This rules out the only scenario that would justify fill-zero.

**How to apply:**
- Counting-stat blank on a present row (disposals > 0, TOG > 0) → treat as missing, use dropna. fill-zero here is a BLOCK-worthy false observation.
- fill-zero is only correct when a blank provably means DNP / no opportunity (usually an absent row, not a present-row blank).
- The real safeguard under EITHER convention: require the denominator be disclosed inline — "X.X per game (N of M games recorded)" — and flag small-sample when N/M < 0.75 or N < 5. Both conventions otherwise hide the denominator (dropna shrinks it; fill-zero fakes the numerator).
- Tackles/marks are observable live (reliable) and post-game (post-1987), so tackle-based tripwires pass observability — phrase them as rolling/season aggregates over recorded games.
