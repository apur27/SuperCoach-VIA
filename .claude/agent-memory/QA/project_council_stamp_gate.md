---
name: project_council_stamp_gate
description: How scripts/check-council-stamp.sh matches files and enforces the audit-record gate — verified behavior as of 2026-07-03
metadata:
  type: project
---

`scripts/check-council-stamp.sh` only evaluates files whose **path** matches
`docs/news/*.md` or `docs/hall-of-fame-stat-*.md` (see script lines ~114,
131-139). Anything else passed to it is silently SKIPPED, not FAILED — so a
naive test with a stamped file at an arbitrary path (e.g. in /tmp) will report
"0 checked, 1 skipped, 0 failed" and exit 0, which looks like a pass but proves
nothing. To test the failure path, the file must live at a matching path.

`AUDIT_ENFORCE=1` is the default (not opt-in). A doc carrying a
`<!-- council-pipeline: ... PASS ... -->` stamp with no matching content-hash
audit record in `.claude/audit/sentinel-<hash>-*.json` hard-fails (exit 1) with
"no DataSentinel audit record exists for it." A doc whose stamp DOES match an
audit record passes (exit 0) with per-file "OK: stamp verified against audit
record (content hash ...)" output.

`.claude/audit/` is gitignored (`.claude/audit/*.jsonl` and
`.claude/audit/sentinel-*.json` per `.gitignore`), so audit records are
operational/local-only and never show up in `git status`.

**Why:** saved so a future QA cycle testing this gate doesn't waste a cycle
re-discovering the path-matching rule or mistaking a SKIP for a PASS.

**How to apply:** when asked to verify this gate fails correctly, copy a real
badged doc to a temp file **under `docs/hall-of-fame-stat-*.md` or
`docs/news/*.md`** (e.g. `docs/hall-of-fame-stat-QATEST.md`) with content
altered enough to change its hash, run the script, then delete the temp file
immediately — never leave it in the working tree since it will show up in
`git status`.
