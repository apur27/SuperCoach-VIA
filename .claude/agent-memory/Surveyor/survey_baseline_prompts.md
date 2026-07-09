---
name: survey-baseline-prompts
description: sha256 (first 12) of every agent prompt and skill file as of the 2026-07-09 STANDARD survey — diff against these to detect prompt drift on the next survey
metadata:
  type: project
---

Baseline taken 2026-07-09 (STANDARD survey, HEAD 0b07b3048). All changes since the
2026-07-07 baseline were verified as routed-finding closures (F1/F02a/F6/F7/F9/F11/
F13/F17/A-07 commits) — zero unexplained drift. NOTE: BriefBuilder.md hash below
includes an UNCOMMITTED one-line F06 fix (commit-message year de-hardcode); if that
fix is reverted rather than committed, the hash regresses to cf35b7208e0e.

```
596c84dec550 .claude/agents/BriefBuilder.md   (working tree; incl. uncommitted F06)
df07a9621c41 .claude/agents/Chronicler.md
87e0acafac79 .claude/agents/DataSentinel.md
ce7055ce1e46 .claude/agents/FootyStrategy.md
c566311a075d .claude/agents/Gaffer.md
f76d24c61bb4 .claude/agents/QA.md
8ccb6c235227 .claude/agents/Scientist.md
0bf49c0b0826 .claude/agents/Skeptic.md
05034b03cf82 .claude/agents/Surveyor.md
0573d48f9fcc .claude/skills/council-brief.md
b53fc3c931fb .claude/skills/weekly-cycle.md
```

**How to apply:** unchanged hash on a file with an open routed finding = finding not
yet actioned; re-raise it. Changed hash on a file with no finding = unexplained drift;
read the diff. See [[survey-open-findings]].
