---
name: coach-names-grep-f-gotcha
description: config/coach_names.txt has header comments and blank lines that must be stripped before use as a grep -f pattern list, or every line of the doc false-positive-matches.
metadata:
  type: feedback
---

`config/coach_names.txt` (the read-only gate config for step-7 coach-name scanning) starts with a
multi-line `#`-comment header and has blank separator lines between name-group sections. Passed
directly as `grep -i -f config/coach_names.txt <doc>`, the blank lines act as empty regex patterns
that match every line unconditionally — producing a false coach-name hit on the doc's title, every
section heading, effectively the whole file.

**Why:** discovered 2026-08-29 verifying docs/afl-insights.md — raw `grep -f` flagged lines 1, 7,
20, 30 (headings, unrelated prose) as coach-name violations, none of which contained an actual
coach surname.

**How to apply:** always sanitize before using the config as a pattern list:
```bash
grep -v '^#' config/coach_names.txt | grep -v '^\s*$' > /tmp/coach_names_clean.txt
grep -n -i -w -f /tmp/coach_names_clean.txt <doc>
```
Use `-w` (word boundary) too — the config header itself says matching is "case-insensitive substring
on word boundary," and without `-w` a short surname fragment can match inside an unrelated word.
</content>
