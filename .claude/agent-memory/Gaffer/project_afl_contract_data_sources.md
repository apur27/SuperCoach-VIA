---
name: afl-contract-data-sources
description: Where 2026 AFL contract/free-agency data actually comes from — afl.com.au + zerohanger WebFetch OK, other contract sites permission-blocked; live Python scrape hits bot-fronted SPAs so scraper falls back to verified fixtures
metadata:
  type: reference
---

For AFL contract / free-agency / off-contract data (not in repo `data/`), established in the 2026-06-19 cycle:

**Sources that WORK (server-rendered, WebFetch-permitted):**
- `https://www.afl.com.au/news/<id>/2026afl-free-agentslist` — the OFFICIAL AFL free-agents list with Restricted/Unrestricted status per club. Authoritative for FA classification. Tag `[contract source: AFL.com.au]`.
- `https://www.zerohanger.com/afl/players/off-contract-<year>/` — broader off-contract pool per club (all expiring contracts, not just 8-year FAs). Tag `[contract source: ZeroHanger]`.

**Sources BLOCKED in this env:** WebFetch is permission-denied on `aflcontracts.com.au` and `theaflcontracttracker.com.au`; the footywire `ft_player_contracts` path 404s. Don't burn turns retrying them — go straight to afl.com.au + zerohanger.

**Scraper reality (important):** a live Python `requests`/bs4 fetch of afl.com.au and zerohanger returns HTTP 200 but JS/Cloudflare SHELL bodies that parse to 0 rows. The working pattern: scraper attempts live fetch, detects "200 but 0 rows", and falls back to a persisted verified-ground-truth FIXTURE (`scrapers/fixtures/*.html`), logging `source: fixture`. This is honest as long as the fixture content is hand-verified and the doc's methodology says "verified fixture, not live feed." Artifact lives at `data/contracts/afl_<year>_contracts.csv` (Scientist owns the write; Gaffer never writes under data/).

**Honesty contract for these:** contract/FA status is `[contract source: ...]`, NEVER `[data]`. Salary-cap room has NO public/structured source — it is inference only, zero dollar figures. Career stats (games/age) still come from repo `data/player_data/` as `[data]`. See [[footystrategy-name-hallucination]].
