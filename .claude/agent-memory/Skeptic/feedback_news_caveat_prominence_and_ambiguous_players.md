---
name: news-caveat-prominence-and-ambiguous-players
description: News-article reviews — watch lede/subtitle overclaim vs end-of-doc caveats, and inconsistent handling of contract-CSV "ambiguous repo match (2 files)" players
metadata:
  type: feedback
---

For `docs/news/` articles built on `data/contracts/afl_2026_contracts.csv`, two recurring soft-spots to probe (neither is usually a BLOCK, both are PASS_WITH_CONCERNS material):

1. **Caveat prominence, not just presence.** These articles park honest caveats in an end "Limitations and methodology" section (often self-labelled "not buried"). The caveat being present is necessary but not sufficient — check whether the *lede/subtitle* overclaims relative to it. Classic tell: subtitle says "every contract status sourced" / "every stat verified" while the fixture-snapshot-not-live caveat and the no-repo/ambiguous-player exceptions live only at the very end. A reader who stops early gets a stronger impression than the data supports.
   **Why:** prominence-smoothing is the news-doc analogue of lens-tension smoothing — the disclosure exists but is positioned so the cleaner story lands first.
   **How to apply:** read the subtitle/lede claims, then grep the end-section caveats, and flag any lede claim the end-section walks back.

2. **"ambiguous repo match (2 files)" players must be handled consistently.** The contract CSV flags some players (e.g. Bailey Williams, Tom Campbell, Jack Carroll, Tom Lynch) as ambiguous with blank games. Articles should either decline to cite a figure (the safe move) or disclose the resolution. Watch for asymmetry: one ambiguous player cited with a figure while an identically-flagged one is left uncited. If the brief pre-clears a specific case (e.g. "Tom Lynch DOB quirk flagged"), that one is acceptable — but surface the asymmetry so the operator sees the differing standard.
   **Why:** identical source-flags handled differently is a defensibility gap a sharp reader will catch.
   **How to apply:** grep the CSV for "ambiguous repo match", then confirm each such player either carries no `[data]` games figure in the article or has an in-line disclosure.
