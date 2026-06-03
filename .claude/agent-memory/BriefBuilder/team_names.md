---
name: team-name-canonicalisation
description: Canonical AFL team name forms as used in data/matches/ CSV files
metadata:
  type: reference
---

Canonical team names from `data/matches/matches_2026.csv`:

- St Kilda (not "St. Kilda", not "stkilda" — slug form is "stkilda" but data value is "St Kilda")
- Hawthorn
- Greater Western Sydney (not "GWS", not "Greater Western Sydney Giants")
- Brisbane Lions (not "Brisbane" alone)
- Western Bulldogs (not "Bulldogs", not "Footscray")
- Port Adelaide
- Gold Coast
- North Melbourne

Slug rule (for filenames): lowercase, hyphen-separated, remove spaces and dots.
- "St Kilda" → "stkilda"
- "Greater Western Sydney" → "greater-western-sydney"
- "West Coast" → "west-coast" (not "west-coast-eagles")
- "Essendon" → "essendon" (not "essendon-bombers")
