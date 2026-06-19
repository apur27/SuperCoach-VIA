---
name: feedback-collision-gate-layering
description: A per-cell [data] number gate cannot catch a wrong-player row whose number is real — need cross-club identical-row scan + Skeptic semantic check
metadata:
  type: feedback
---

When a doc derives rows by mapping an identifier (jersey number, name key) to a player, DataSentinel's per-cell number check is NOT sufficient: a wrong-player row whose number happens to be a real figure for *some* player passes every cell check. Verify the PERSON, not just the number.

**Why:** 2026-06-19 squad-union rebuild of the list-quality article. DataSentinel Pass 1+2 passed 67/67 sampled cells, yet Skeptic caught a jersey-map surname collision — the Bulldogs' Bailey Williams row (2015 #48, 184g) was cloned onto West Coast. The Scientist sweep then found FIVE such collisions (Bailey Williams, Will Hayes, Sam Butler, Tom Lynch, Callum Brown), all from an unscoped `name -> pick/games/grade` lookup matching same-name players across clubs/eras. Root cause: team-scope the lookup (resolve to the player whose 2026 perf file is for THAT club + debut-era).

**How to apply:** For any rebuild that resolves IDs to people, instruct DataSentinel to ALSO run a deterministic cross-club duplicate-row scan (no two clubs share an identical (name, year, pick, grade, games) row) and instruct Skeptic to semantically check named individuals against their club/career. Treat "same number, wrong person" as its own defect class, distinct from "wrong number". Related: [[project_footystrategy_name_hallucination]], [[squad-union-rebuild]], [[feedback_parallel_council_commits]].
