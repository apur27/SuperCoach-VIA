---
name: batch-sweep-over-serial-rounds
description: When the same defect class appears twice in one session, recommend one repo-wide deterministic sweep instead of serialized fix-and-re-audit rounds
metadata:
  type: feedback
---

When a pattern class (esp. class 1 staleness drift) recurs in a second document within one engagement, stop serialized find→fix→re-audit rounds. Recommend a single deterministic repo-wide sweep (script comparing all published docs against source artifacts) and route remaining instances to backlog.

**Why:** 2026-07-25 session found staleness drift in four docs across four serial rounds (HOF gate → README → run-report → afl-backtest-2026.md), each triggering a new fix cycle plus three overlapping full audit passes. Net value positive but ~2x the time disciplined batching would have cost; find-severity declined each round (CRITICAL → live-wrong → stale prose) without anyone noticing the gradient. User flagged the session length directly.

**How to apply:** In any survey or advisory role, track defect class per finding. Second instance of the same class = switch recommendation from "fix this doc" to "commission the sweep, backlog the rest." Also: bundle verification into one consolidated pass after fixes land, not one pass per round. Related: [[measurement-vintage-crossing]].
