---
name: coach-name-false-positives
description: config/coach_names.txt carries bare surnames (Hardwick, Bolton, Lyon, Nicks, Voss) that collide with active AFL players — verify first name before blocking
metadata:
  type: reference
---

`config/coach_names.txt` is the canonical gate list, but several entries are bare
surnames that collide with current players. Confirmed collisions seen in published
docs: **Hardwick** (Blake Hardwick, Hawthorn), **Bolton** (Shai Bolton). Others on the
list with the same risk profile: Lyon, Nicks, Voss, Roos.

**Do not block on a surname hit without checking the first name.** The config's own
scope note is explicit: "Player names are NOT coach names and are always allowed
(Cripps, Walsh, Milera, ...)", and the rationale memory
(`.claude/agent-memory/FootyStrategy/coach_anonymity_lint.md`) says the same. The rule
is coach-specific; players are normal football vocabulary.

**How to apply:** grep the list, then for every hit read the surrounding cell — a first
name plus a stat line in an auto-generated table is a player, not a coach reference.
Report unavoidable collisions as gate hygiene against the *config*, not as a doc
defect. Auto-generated tables (backtest misses, brief tracking lists) will keep
tripping these, and a gate that cries wolf trains reviewers to wave through the real
violation.
