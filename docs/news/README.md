# AFL news - data-grounded commentary

> [← Back to main README](../../README.md) | [← Back to AFL insights](../afl-insights.md)

This is the news desk for SuperCoach-VIA. Each entry takes a current AFL story - a coach change, a club crisis, a list decision, a finals run, a controversy - and grounds it in the actual data this repo carries. No vibes. No quoted-without-source numbers. Every claim that is verifiable against the dataset is tagged `**[data]**` and reproducible from the CSV files in `data/`.

The format is a two-layer collaboration:

- **Scientist** writes the data layer: season records, career stats, ladder positions, statistical comparisons across eras. Numbers come from the repo, not from memory.
- **FootyStrategy** writes the tactical and cultural layer: what the data means on the ground, what a coach actually does about it, what the structural read is.

The two layers are visible in every entry. The data tables and `**[data]**` tags are Scientist's work; the prose interpretation between the tables is FootyStrategy's.

## Why a news section in a data repo

Most AFL commentary moves fast and cites little. The numbers in the columns of a Monday-morning think piece are typically remembered, not looked up. This repo has 125+ years of structured match data and a per-game player table going back decades - so for any current story where the question is "is that true?" or "how does this compare to the last time it happened?", the answer is sitting in a CSV file two directory levels away.

The news section is the place to turn that into reading. Slower than a tweet. Smaller than a podcast. Built so the numbers survive a click.

## Entries (most recent first)

| Date | Headline | Topic | Status |
|---|---|---|---|
| 2026-06-19 | [AFL 2026–2030: Five-Year Grand Final Strategy — All 18 Clubs](2026-06-19-afl-2026-5yr-grand-final-strategy.md) | All 18 clubs' five-year path to a Grand Final, read from each club's partial-2026 snapshot (ladder through ~Round 15) rather than its reputation: competitive tiers, structural gaps, off-contract exposure, and the one acquisition each club most needs. Dedicated read on Restricted free agent Zak Butters (152 games). Salary-cap reads are figure-free inference. | Complete (Scientist + FootyStrategy + DataSentinel + Skeptic layers; team_1=home convention on Collingwood sub-figures authorised by repo owner) |
| 2026-06-19 | [AFL 2026 — Free Agency and the Trade Window: What Every Club Needs and Who They Should Chase](2026-06-19-afl-2026-free-agency-trade-window.md) | All 18 clubs' trade-window needs + specific targets, built on scraped 2026 contract/free-agency data (AFL.com.au official FA list + ZeroHanger off-contract pool, 124 players) cross-referenced with repo career stats. Dedicated read on marquee restricted free agent Zak Butters. Figure-free, inference-only salary-cap reads. | Complete (Scientist + FootyStrategy + DataSentinel + Skeptic layers; genuine council chain) |
| 2026-06-17 | [AFL 2026 — List Quality and Draft Pipeline: Where Every Club Stands](2026-06-17-afl-2026-list-quality-draft-pipeline.md) | All 18 clubs: 2026 list identity + full selected-22 squad annotated with draft pick/grade/games + 22-year draft efficiency per club (1,529 picks, 2004–2025). Combines team list analysis with National Draft data pipeline. | Complete (Scientist + FootyStrategy layers) |
| 2026-06-16 | [The AFL National Draft: Error, Structure, and the Shape of Talent](../articles/afl-draft-analysis-2025.md) | 7-part comprehensive analysis: error taxonomy; school pipeline; brand-currency thesis; data decision framework (pick cliff at 16, club efficiency 3-to-1); Richmond case study (0/32 post-dynasty void); Geelong model (average drafting + late-extraction + trade = 4 premierships); 7-rule data playbook. 1,538 National picks, 2004–2025 | Complete (Scientist + FootyStrategy layers) |
| 2026-06-05 | [List Management 101 — Is the Top-10 Draft Pick Strategy a Path to Premiership Dominance?](../list-management-101.md) | Strategic analysis across 127 premierships and 13,329 player careers: premiership cores are built on retained, experienced players (17-21 of the top 22 with 150+ game careers), not on draft position; data carries no draft-pick numbers so the proxy is career longevity | Complete (single-operator cycle; deterministic DataSentinel PASS + Skeptic PASS_WITH_CONCERNS, no full council dispatch available) |
| 2026-05-30 | [Meet Gaffer — the council's new delivery lead and editor-in-chief](../council-intro-gaffer.md) | Internal / team announcement — introduces the orchestration + presentation role that commissions the council chain and decides "ready to ship" on PASS; boss of process, not of truth | Internal / team announcement (not a council-reviewed article) |
| 2026-05-29 | [Greg Williams — The Possession Engine](2026-05-29-greg-williams-possession-engine.md) | Career tribute to the dual-Brownlow Carlton/Sydney/Geelong midfielder; the handball-out-of-traffic signature behind the two medals eight years apart | Complete (all six council layers) |
| 2026-05-29 | [Jonathan Brown — The Fist of God](2026-05-29-jonathan-brown-fist-of-god.md) | Career tribute to the Brisbane Lions full-forward; the 2004 Grand Final as the defining image of refusing to be moved | Complete (all six council layers) |
| 2026-05-27 | [James Hird and the Essendon vacancy — the case for Australia's most contested second chance](2026-05-27-hird-essendon-coach.md) | Argued case for James Hird as next Essendon senior coach; engages the supplements-era counterarguments | Complete (data + FootyStrategy layers) |
| 2026-05-25 | [Neale Daniher — Why Not](2026-05-25-neale-daniher-tribute.md) | Tribute on the passing of the Essendon Hall of Fame footballer, Melbourne coach, and FightMND founder — aged 65 | Complete (all six council layers) |
| 2026-05-19 | [Scott Pendlebury — The StormRider](2026-05-19-pendlebury-stormrider.md) | All-time games record milestone (tied at 432) | Complete (data + FootyStrategy layers) |
| 2026-05-15 | [Who should be the next Carlton coach?](2026-05-15-carlton-next-coach.md) | Carlton coaching succession | Complete (data + FootyStrategy + Codex layers) |
| 2026-05-15 | [Richmond vs St Kilda, Round 11, 2026: Milera out, Tigers in](2026-05-15-richmond-vs-stkilda-r11.md) | Match preview / injury impact | Complete (data + FootyStrategy + Codex layers) |
| 2026-05-13 | [Michael Voss steps down: what went wrong at Carlton, and why it doesn't diminish a legend](2026-05-13-voss-carlton.md) | Carlton coaching change | Complete (data + FootyStrategy layers) |

## House rules for entries

Every entry follows these rules. Both agents are bound by them.

1. **Every specific number is tagged.** Games, wins, losses, points, votes, percentages - if it comes from the data, it carries a `**[data]**` tag. If it cannot be verified in the data and is sourced from public record (e.g., a Brownlow Medal year), it is tagged `**[historical record]**`.
2. **The data source is named at the bottom.** A "Methodology and caveats" section at the end of every entry lists which files were read.
3. **Associational vs. causal language is explicit.** A coach's record under the coach's tenure is associational, not causal. If the entry implies a causal claim ("Voss's tactics caused the decline"), it must be flagged or supported with a comparison.
4. **No betting tips.** The repo is a public data and modelling project, not a tipping service.
5. **No business decisions disguised as analysis.** "Carlton should appoint coach X" is a club decision, not a data decision. The entry can lay out the trade-offs; it cannot make the call.
6. **The FootyStrategy layer is marked.** Placeholder comments (`<!-- FOOTYSTRATEGY INSERT: ... -->`) sit in the draft until the tactical layer is filled in. Entries are not published-final until both layers are in.
7. **Entries are dated.** Filename format: `YYYY-MM-DD-shortslug.md`.

## How an entry gets built

The workflow is:

1. A story breaks (coach sacking, retirement, controversy, finals upset, list move).
2. **Scientist** pulls every number in the repo that bears on the story: season records, career stats, comparable historical cases. Writes the data layer with `**[data]**` tags on every figure.
3. The draft has clearly-marked placeholders for the tactical interpretation.
4. **FootyStrategy** reads the data layer and fills in the placeholders with the football-coach interpretation: what the numbers mean structurally, what a club typically does about this kind of situation, what the cultural reading is.
5. The two-layer file is reviewed once for tone consistency and then committed.

This is the same pattern used in the [coaches strategy corner](../coaches-strategy-corner/README.md): Scientist provides the data; FootyStrategy provides the football meaning; the user/editor signs off on the final piece.

## Glossary of tags

- `**[data]**` - verified against a specific file in `data/` at the time the entry was written. Reproducible.
- `**[historical record]**` - public-record fact (e.g., Brownlow winner of a given year) that is not in this repo's data files but is uncontroversial.
- `**[historical record - unverified in data]**` - public-record fact that cannot be cross-checked here; reader should treat with the same trust they'd give a tweet from a credentialled source.
- *(no tag)* - commentary, opinion, interpretation, or a sentence built on top of the tagged numbers above it.
