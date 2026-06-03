---
name: flaky-output-channel
description: This session's Bash/Read output truncated+duplicated; one-value-per-call and grep -c were reliable; confirm surprising bulk results before believing
metadata:
  type: feedback
---

When tool output renders unreliably (lines truncated mid-string, duplicated, or
only the first line of multi-line output appears), STOP batching output.

**Why:** During the 2026-05-31 docs audit, both Bash and Read repeatedly returned
garbled multi-line text and a parallel batch got fully cancelled by one error in
the group. It nearly produced two FALSE conclusions: "all docs wiped" (they were
fine, intact in HEAD) and "many forgotten-heroes have no data file" (a regex
artifact — those tables store stats as inline `[data]:` prose, not pipe-table
columns, so the row-regex matched nothing). The files and exit codes were always
correct; only rendering/parsing was the problem.

**How to apply:**
- Verify one fact per Bash call, emitting a SINGLE short line.
- Use `grep -c` / boolean counts and only drill into failures.
- Write computed results to /tmp files, confirm via short summary lines.
- Treat a surprising bulk result (everything fails / everything empty) as a
  likely glitch first; confirm 1-2 cases standalone before believing it.
- Don't batch many independent Bash calls in one block — one error cancels them all.
- The 26,656-file data/player_data dir makes glob() flaky under load; resolve a
  player's exact filename, read THAT file.

See [[feedback_parallel_council_commits]] for "verify by content, not command success".
