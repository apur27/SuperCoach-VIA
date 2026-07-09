---
name: live-stamped-doc-figures-need-asof-freeze
description: Active-player figures in stamped docs move every round; re-deriving them to "live" without an as-of freeze re-strands them within ~1 round
metadata:
  type: project
---

Any [data] figure about an **active** player/club in a published/stamped doc is a moving target
and cannot be finalized to "live" without a machine-readable as-of-round freeze (the F02a
`<!-- verify-asof: round=N -->` directive + visible badge, a Gaffer-owned build).

**Why:** During F02 (2026-07-07, unstranding 5 stamped docs), the working tree already carried a
prior session's "live" corrections — and they were ALREADY ~1 round stale against current data
(R18). Concrete: dustin-martin peer table had Dangerfield at 371 games / 377 goals (data: 372 /
382), Neale 309 (data: 310); forgotten-heroes had Hewett 213 games / 60 goals (data: 214 / 62);
free-agency had Butters 154 games (data: 155). Every active-player figure had advanced. Re-deriving
to "live" now, without an as-of freeze, just recreates the same staleness next round — the exact
strand the fix was meant to close.

**How to apply:**
- Classify every disputed figure retired/frozen-round (STABLE) vs active/live (MOVING) before
  editing. Retired players (Martin, Ablett, Selwood, Fyfe post-2025) and past-round snapshots
  (a doc re-based to end-of-R17) are stable and safe to finalize. Active-player "live" figures
  should be written in the SAME change that adds the as-of badge, else they re-strand.
- A doc re-based to a PAST round (D1 grand-final-strategy → end-of-R17) is the clean case: R17 is
  settled, the ladder/margins won't move. Re-basing R15→R17 materially REORDERS the ladder though
  (2026: North Melbourne 11th→8th, Brisbane 8th→10th, Collingwood 12th→9th, GWS 13th→15th), which
  cascades into any position-based narrative tiering — that is a narrative rework, not a find-
  replace.
- 2026 ladder-through-R17 recipe: `data/matches/matches_2026.csv`, filter `round_num<=17`, score
  = final_goals*6+final_behinds (final_* are cumulative match totals), W/L/D from pf vs pa, sort by
  pts then pct. Teams have played 13-15 games each through R17 (byes/uneven fixture).
- Related: [[dropna_denominator_coverage_bias]] (the D3 rarity count is one such live figure: 17
  live, was expected 16).
