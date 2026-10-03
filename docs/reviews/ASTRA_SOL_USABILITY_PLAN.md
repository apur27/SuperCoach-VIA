# Astra design review and ordered implementation plan

Reviewed 3 October 2026. Read-only review of an isolated checkout at `8df4dab4a` and the sealed preview release `20261003T105359Z-67fb4988599a`. Read `CLAUDE.md`, README introduction/data-status sections, shared shell/styles, homepage, history explorer and player tables. No source, dataset, paused worktree or Claude files changed.

## Evidence and diagnosis

Captured 32 page/viewport/theme states: homepage, player search, history rankings and a player profile, at 320/390/768/1440px in light/dark. Open-menu screenshot is additional. Inspected actual homepage screenshots at 320, 390, 768 and 1440px; mobile history, player and expanded navigation screenshots. All sampled states had zero document horizontal overflow and no browser page errors. This establishes a usable baseline, not a full regression or accessibility verdict.

- The homepage says “Season 2026 overview” without explaining the product. Its subtitle promises upcoming forecasts even when none exists. The first major section then repeats the missing-forecast information already in a warning.
- Navigation sits against the viewport edge on desktop while brand/body start at the centered content gutter. Eight links plus More wrap awkwardly at 768px. On phones, brand, preference selectors and Menu occupy three or four rows. Header height is 169px at 320 and 137px at 390.
- “Status Current validation PASS” appears immediately below “known audit failures.” The facts describe different checks, but the interface leaves readers to infer that distinction.
- The main heading begins at y=475px at 320 and y=398px at 390. The provisional warning must remain conspicuous; reclaim space from shell and routine metadata instead.
- At 390px, the results table is 1,077px wide inside a 356px region. The player season table is 4,650px wide and the game log 3,090px. Regions scroll, but there is no visible instruction, and row identity disappears during horizontal scrolling.
- History hides Clubs, Seasons and Games with data on small screens through `.col-optional`, without an available reveal control. The player names and main value are separated by wide columns on desktop.
- The homepage season-leaders table has 40 rows while its neighbouring form table has five. The asymmetry leaves a large empty area and makes a landing page feel like a report. At 768px even the two-column form panel is too narrow for its own data.
- Article excerpts expose internal publishing text (“Council chain…”). Titles and dates are useful; those excerpts need not be on the homepage.
- Existing colours, contrast, restrained borders and light/dark themes provide a sound visual base. Keep them; use clearer grouping and consistent spacing instead of a new visual identity.

Evidence: `astra-baseline.json`, `screens/before-home-390-light.png`, `screens/before-home-1440-dark.png`, `screens/before-home-768-light.png`, `screens/before-menu-390-light.png`, `screens/before-history-390-light.png`, `screens/before-player-390-dark.png`. Local evidence is under `var/ui-review/20261003/` (outside Git).

## Ordered work for Sol

### 1. Make the global header compact and navigation predictable

Files: `web/src/layouts/AppLayout.astro`, `web/src/lib/nav.ts`, shared CSS; shell only if necessary.

- Keep brand and Menu on the first row on narrow screens. Put Theme and Times in a single intentional secondary row with their labels; both controls remain accessible. Prefer a grid/flex layout with explicit areas, not incidental wrapping. Primary controls and Menu retain 44px touch targets.
- Use the mobile disclosure at widths where the complete desktop navigation cannot fit, including 768px. Desktop navigation must share the same centered max-width and 16px inner gutter as the brand and main content; repair the `.site-nav ul` override that currently removes those margins/padding.
- Put the most common exploration routes first: Home, Players, Matches, Rankings, Predictions, then More. “Home” maps to the existing overview key and route; “Rankings” maps to the existing history key and route. Move Teams, Accuracy and Articles into More alongside existing secondary links. Preserve all current routes, footer access, `aria-current`, keyboard/Escape behaviour and no-JS navigation. This is a label/order change, not new routing.
- Give the current primary link a subtle filled background plus the existing underline. Make More visually recognisable as a disclosure (small chevron is sufficient).

Acceptance: brand and Menu occupy the same row at 320 and 390; header height at 390 is at most 112px with menu closed. At 768 use the disclosure, with no stranded More row. At 1440 nav and content left edges align. Menu opens/closes with pointer and keyboard; Escape restores focus; all links remain reachable without JS. Preferences persist.

### 2. Explain status accurately and reduce routine metadata height

Files: `Freshness.astro`, shared CSS; preserve `ProvisionalStatus.astro` content/logic and robots logic.

- Keep the provisional banner, snapshot/build dates and Audit details link visible before main content on every affected page. Do not collapse, weaken or condition it on viewport/JS.
- Label freshness as “Freshness at build” rather than generic “Status.” Replace generic “validation PASS” with “Snapshot selection: PASS” (use the actual field value). Put this value and source-checked timestamp inside the existing Release details disclosure if needed for compactness. Keep season, coverage, freshness, Data status and Methodology visible.
- Inside details explain that snapshot selection records whether the build used the locally selected dataset, and link Data status for source-audit findings. Parent code review established that `freshness.validation_state` only tests the selected-snapshot pointer (`builder.py` `_is_current`); it is not a release-consistency result. This corrects the initial plan’s overly strong interpretation. Separately run integrity evidence may establish release consistency, but this field cannot.
- Keep release IDs and generated/published timestamps in details. Use readable type; do not solve height by shrinking text further.

Acceptance: the compact provenance strip stays within the existing 170px limit at 320px for both DEMO fixtures and the real candidate, in all tested time-zone choices. The local real candidate currently measures 133px; the recorded CI failure is 194px on a fixture against a 170px limit, so local candidate success alone does not close it. Do not raise the limit. Preserve all date precision/time-zone semantics, stale warning behaviour, snapshot-specific audit matching and noindex.

### 3. Give the homepage a clear purpose and useful first actions

File: `pages/index.astro`, modest page/shared styles.

Use this reader-facing opening, based on README:

- H1: “Explore AFL players, matches and history”
- Intro: “SuperCoach VIA brings together player and match history, comparisons, rankings and downloads.”
- Supporting line: “Check the snapshot date and data status before using a total, ranking or prediction.” Link Data status.
- Keep the season as a small contextual label near the season content, derived from `ov.season`.
- Place the existing player search directly after the introduction. Full-width search input on phones, clear submit button. Retain form GET semantics and query encoding.
- Add a compact group of direct destination links immediately after search: Rankings (`history/`), Matches (`matches/`), Compare players (`compare/`), Downloads (`downloads/`). Use the same button/link-card treatment and spacing, with no invented counts.
- Keep public warnings intact, visible near the introduction/actions. Rename their heading “Snapshot notes” if desired; do not suppress raw warning records or substitute invented audit conclusions.

Then order the page: Recent results; Forecast status; Form and season leaders; Latest articles. Keep ForecastStatus and its actual reason, future fixtures, forecast highlights and model details supported. When upcoming is empty, present the existing empty state compactly rather than a large vacant subsection. Preserve current warning and reason information even where it repeats; avoid brittle string filters against generated warnings.

Acceptance: a newcomer can tell what the site provides and can reach player search, rankings and matches without interpreting release machinery. At 390px search begins within the first 900px viewport with notices visible. Search for a real player reaches a correct result; direct links work under both `/` and `/SuperCoach-VIA/`. Available-future-fixture DEMO state still renders its forecast/highlights.

### 4. Bound homepage highlights and keep all records available

Files: homepage, `LeaderTable.astro`, shared CSS if needed.

- Show at most five season-leader rows initially. Label explicitly “Season leaders preview.” Keep the release order and existing values; do not recalculate or reinterpret rankings.
- Place the full season-leaders table in a native details disclosure directly beneath, labelled “View all season leaders in this snapshot.” Render the complete supplied rows there. This avoids linking to a different historical table that may use another scope/method.
- Form table remains complete. Stack the two panels until each is wide enough to be useful (at least until a 1024px viewport); avoid two cramped tables at 768px.
- Keep article titles, dates/category and links, with an “All articles” link. Omit all homepage excerpts consistently rather than filtering specific council phrases. Article pages/data remain untouched.

Acceptance: closed homepage no longer contains a 40-row visible leaders block; complete supplied leaders can be reached with one keyboard-accessible disclosure, also without JS. No new numbers or rewritten player facts. Homepage sections use consistent spacing and the article section is discoverable without scrolling through all leaders.

### 5. Make table overflow visible and preserve orientation

Files: shared styles, `HistoryExplorer.tsx`, `PlayerView.tsx`, `common/StatTable.tsx`, `MatchTable.astro`, `LeaderTable.astro` and any small shared helper actually justified.

- Add a visible “Scroll sideways for more columns” hint for narrow/overflowing tables. It must appear outside the scrolled content, remain readable, and associate with the region through `aria-describedby` where practical. For static components a responsive hint is sufficient; do not introduce global mutation observers merely to detect overflow.
- Retain native semantic tables, captions, header scopes and keyboard-focusable labelled scroll regions. Keep numeric columns right aligned and tabular.
- Use scoped sticky row-identity cells for the reviewed tables: player/statistic in leaders/career; season in season summary; match/date in results/game log. For history use the player-name column (including its header) so names remain visible as values scroll; do not blindly sticky every first cell, which would pin Rank instead.
- Sticky cells need an opaque light/dark background, clear separating border/shadow, correct stacking and a bounded width. At 320px they must leave space for at least one useful value column. Permit text wrapping in long identity labels instead of making the pinned column wider than the viewport.
- History: remove hiding of Clubs/Seasons/Games with data for this table and keep those columns available by horizontal scrolling. Put Rank, Player, Value, Coverage before supporting columns so useful numbers come earlier. Keep all information and sort controls accessible at every width. This is simpler than adding a second column-toggle interaction.
- Player tables: use human-readable display labels for underscore-separated stat names while retaining raw keys for data access. Keep column ordering and values intact in this pass; advanced column selection/pagination is unnecessary scope.
- On player profile add a short local link row to Career statistics, Seasons and Game log using existing section IDs. This helps users reach tables below the long profile/coverage information.

Acceptance: at 320/390px, scroll every representative table to its far right using pointer and keyboard; identity remains legible, rightmost data is reachable, sticky cells do not overlap the caption, and document overflow stays zero. History coverage/denominators and hidden supporting columns are available on mobile. Sorting still preserves URL state and numeric order; links still open the correct player/match. At 200% zoom, no clipped controls or unreadable pinned-column collision. Do not remove data or reduce table type below the existing size.

### 6. Apply one restrained visual treatment

Use the existing palette/tokens. Give homepage exploration links and section groups consistent surface, border, radius and padding. Maintain clear hierarchy: intro/search first, section headings second, captions third, metadata muted. Reset profile `dl` default `dd` indentation in a scoped rule so labels/values align cleanly. Avoid broad colour changes, decorative imagery, new dependencies, charts redesign or animation. Verify both themes visually, especially opaque sticky cells, disclosure triangles, focus rings and selected navigation.

## Verification and final Astra review

Implement with focused tests as required by CLAUDE.md. Add behaviour assertions for compact header/provenance, homepage destinations and complete leaders disclosure, visible table guidance/mobile data access, plus sticky identity/scroll reachability. Avoid screenshot pixel tests that duplicate CSS values.

Parent/Sol runs web typecheck/lint/unit/browser suites at both base paths, normal payload/build/seal checks for the fresh output, and real-candidate smoke flows. Keep the existing provenance height threshold. The separate `/usr/bin/node` CI portability issue is outside this presentation plan; report it independently unless separately assigned.

Required final browser evidence:

1. Homepage screenshots at 320, 390, 768 and 1440 in light/dark; mobile closed/open menu.
2. History and player tables at 320/390 light/dark, including scrolled-right states; one desktop state each. Expand full leaders once and record complete row access.
3. Search → player; rankings sort → player; match → player → back; mobile menu/Escape; theme persistence; section anchor navigation. Check no unexpected page/console/request errors.
4. No document overflow, one H1, visible focus, axe WCAG A/AA on homepage/history/player in both themes; no-JS navigation, provisional warning, robots, links and leaders disclosure remain usable.
5. Source-audit FAIL and release-consistency PASS remain distinct. Candidate data, snapshot identity and player values are unchanged. No edits to Claude work or numeric pipeline.

Astra should inspect the new actual screenshots and representative interactions after Sol finishes, then record any remaining presentation defects separately from existing source-data limitations. No constraint questions are needed before implementation; all choices above stay within the requested presentation scope.
