---
name: player-disambiguation-roberts
description: Two roberts_archie performance files exist — 1910 DOB is historical, 2005 DOB is current Essendon player
metadata:
  type: reference
---

Two files exist in `data/player_data/` named `roberts_archie_*_performance_details.csv`:
- `roberts_archie_16071910_performance_details.csv` — historical player (born 1910)
- `roberts_archie_18112005_performance_details.csv` — current Essendon midfielder (born 2005)

When prediction CSV lists "Roberts Archie, Essendon", always use the **18112005** file.

**How to apply:** Whenever searching for Roberts Archie in player data, glob for both and confirm the 2005 DOB file is used for current-season analysis.
