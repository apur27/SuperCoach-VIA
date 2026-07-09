---
name: survey-baseline-prompts
description: sha256 (first 12) of every agent prompt and skill file as of the 2026-07-07 DEEP survey — diff against these to detect prompt drift on the next survey
metadata:
  type: project
---

Baseline taken 2026-07-07 during the DEEP agent-architecture survey. On the next
STANDARD/DEEP survey, re-run `sha256sum .claude/agents/*.md .claude/skills/*.md`
and diff: any changed hash means the file needs a fresh prompt-hygiene read.

```
cf35b7208e0e .claude/agents/BriefBuilder.md
df07a9621c41 .claude/agents/Chronicler.md
088215fce93a .claude/agents/DataSentinel.md
59a1a3b9fc68 .claude/agents/FootyStrategy.md
86f2fa26e7ec .claude/agents/Gaffer.md
cbfefba1c204 .claude/agents/QA.md
611c3c232ac3 .claude/agents/Scientist.md
09dddb00db8d .claude/agents/Skeptic.md
05034b03cf82 .claude/agents/Surveyor.md
97b835c4f1ef .claude/skills/council-brief.md
068b20781018 .claude/skills/weekly-cycle.md
```

Expected changes (routed findings; a changed hash here is GOOD if it closes the finding):
BriefBuilder (F1 Bash tool), QA (F11 JSON schema), FootyStrategy (F6/F7 pipeline
contract + description), Scientist (F17 repo invariants, F18 stale memory sentence),
Skeptic (F9 paths cleanup), Gaffer (F13 Surveyor consult point), weekly-cycle
(F21 stale round sentence, F22 gitignored allowlist entry). See [[survey-open-findings]].

**How to apply:** unchanged hash on a file with an open routed finding = finding not
yet actioned; re-raise it. Changed hash on a file with no finding = unexplained drift;
read the diff.
