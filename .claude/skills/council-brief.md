---
name: council-brief
description: Run the full council chain to produce and ship a pre-match brief. Invoke as "/council-brief <home> vs <away>, round <N>". Encodes every handoff contract, both DataSentinel passes, the Skeptic and QA gates, the Gaffer ship step, and Chronicler. Use whenever the user asks for a coaches-strategy-corner brief for a specific fixture and round.
---

# /council-brief — pre-match brief pipeline

Invoked by the coordinator as `/council-brief <home> vs <away>, round <N>`.
Parse the three inputs: home team, away team, round number. The season year is the current year unless stated.

This skill encodes the ONE canonical council chain. Run the steps in order. Each gate can halt the ship; a halt routes the specific finding back to the owning agent — never to the user as an unresolved problem.

```
BriefBuilder → DataSentinel(Pass 1) → FootyStrategy → DataSentinel(Pass 2) → Skeptic → QA → Gaffer(SHIP) → Chronicler
```

## Standing rules that break this chain (read first)

- **A Pass-1 PASS is NOT final clearance.** The ship is gated on Pass 2. Pass 1 only clears the data skeleton so FootyStrategy can safely build on it.
- **DataSentinel runs TWICE.** Pass 1 gates the data skeleton before FootyStrategy sees it; Pass 2 gates the full doc — including all interpretation-layer prose — before Skeptic.
- **Skeptic BLOCK routes back to FootyStrategy**, not to the user. Re-run only the FootyStrategy → DataSentinel(Pass 2) → Skeptic segment after the fix.
- **DataSentinel must enumerate tags deterministically** via the CLI: `python scripts/tag_vocabulary.py <doc>`. Do not eyeball the tag set.
- **No coach names in published docs.** Checked against `config/coach_names.txt`. A match is a DataSentinel FAIL / Skeptic BLOCK, not a stylistic note.
- **Stage by explicit filename, never `git add .`** Only Gaffer commits, via `scripts/git_commit_safe.sh`.
- Briefs are **user-initiated only** — this skill runs because the user asked for this fixture. Never trigger it autonomously.
- Gaffer never authors or edits a `[data]`-tagged number. Only BriefBuilder/Scientist derive numbers from `data/`.

## Step 1 — BriefBuilder

- **Invoke:** BriefBuilder agent.
- **Pass:** home team, away team, round number, season year.
- **It produces:** the data skeleton at `docs/coaches-strategy-corner/<slug>.md` — season form, H2H ledger, model predictions, top-5-per-side tracking list. Every number carries a `[data]` tag with its source-file annotation. Interpretation is left as `<!-- FOOTYSTRATEGY INSERT -->` placeholders.
- **Expected output:** a draft file path. No prose interpretation yet.
- **Failure:** if BriefBuilder cannot source a required table (missing prediction CSV, missing player file), it says so explicitly rather than inventing a number. Route the gap back to BriefBuilder or escalate the missing data — do not proceed with a hole.

## Step 1b — Scientist review of bias markers (before Pass 1)

- BriefBuilder may leave `<!-- SCIENTIST REVIEW: … -->` markers where a derived figure looks anomalous (e.g. a model bias flag with `|bias| > 0.5`, a suspicious H2H gap, a coverage-era boundary case).
- **Invoke:** Scientist to resolve every such marker before DataSentinel Pass 1. Scientist either corrects the figure (re-derived from `data/`, quoted) or annotates why it stands.
- **No `<!-- SCIENTIST REVIEW -->` marker may survive into Pass 1.** An unresolved bias flag shipping to the reader is exactly what this step exists to stop.
- If there are no markers, this step is a no-op — proceed to Pass 1.

## Step 2 — DataSentinel Pass 1 (data skeleton gate)

- **Invoke:** DataSentinel agent on the BriefBuilder output.
- **Pass:** the draft file path, and note this is **Pass 1 (data skeleton)**.
- **It does:** enumerates every `[data]` tag via `python scripts/tag_vocabulary.py <doc>`, re-reads each figure from the cited CSV, flags untagged stat-shaped numbers, coach-name violations (`config/coach_names.txt`), FanFooty schema violations (`config/fanfooty_schema.yaml`), and era-coverage violations (`config/stat_coverage_eras.yaml`).
- **Verdict:** machine-readable JSON. `PASS` iff tags_failed == 0 && untagged_numbers_flagged == 0 && coach_names_flagged == 0 && schema_violations == 0.
- **On PASS:** proceed to FootyStrategy. Remember: this is NOT final clearance.
- **On FAIL:** halt. Route the specific failing tags back to BriefBuilder, re-run only Step 1 → Step 2. Never soften or re-run-until-green.

## Step 3 — FootyStrategy

- **Invoke:** FootyStrategy agent on the Pass-1-cleared skeleton.
- **Pass:** the cleared draft file path. FootyStrategy fills the `<!-- FOOTYSTRATEGY INSERT -->` placeholders with lens analysis, tiered recommendation, tripwires, and caveat propagation.
- **It produces:** the completed brief — interpretation layer added, structure/tier/tripwire format per its output contract.
- **Failure:** FootyStrategy must not invent data or exceed upstream caveats, and must not name coaches. If it cannot write a defensible interpretation from the skeleton, it flags Insufficient rather than overselling.

## Step 4 — DataSentinel Pass 2 (full-doc gate)

- **Invoke:** DataSentinel agent on the full completed doc.
- **Pass:** the completed draft path, and note this is **Pass 2 (full doc)** — verify ALL `[data]` tags including any FootyStrategy added, plus re-run the coach-name and untagged-number sweep across the new prose.
- **Verdict:** JSON, same PASS rule as Pass 1.
- **On PASS:** proceed to Skeptic. This is the clearance the ship is gated on.
- **On FAIL:** halt. If the failure is a data tag, route to whoever wrote it (FootyStrategy for interpretation-prose numbers, BriefBuilder for skeleton numbers). Re-run only the affected segment through Pass 2.

## Step 5 — Skeptic

- **Invoke:** Skeptic agent on the Pass-2-cleared doc.
- **Pass:** the doc path. Skeptic probes tripwire observability in this repo's data, caveat-hierarchy fidelity vs upstream findings, and lens-tension smoothing.
- **Verdict:** `PASS` / `PASS_WITH_CONCERNS` / `BLOCK`. Skeptic never silently edits the doc.
- **On PASS or PASS_WITH_CONCERNS:** proceed to QA. Log concerns in the retro.
- **On BLOCK:** halt, and route by the *nature* of the finding — not always to FootyStrategy. A BLOCK never reaches the user as an open problem, and is never overridden.

  | BLOCK cause | Route to | Re-run segment |
  |-------------|----------|----------------|
  | Interpretation error — lens-tension smoothing, caveat exceeded, unfalsifiable claim, manufactured certainty | FootyStrategy (Step 3) | Step 3 → Pass 2 → Skeptic |
  | Data error — CRITICAL smoke-test mismatch, a `[data]` number wrong at source, a skeleton figure | DataSentinel / BriefBuilder (Step 1–2) | Step 1 → Pass 1 → Step 3 → Pass 2 → Skeptic |
  | Missing/limited data the prose leaned on | BriefBuilder or escalate the data gap | from Step 1 |

  A data-layer BLOCK routed to FootyStrategy would be unfixable there (FootyStrategy cannot correct a skeleton number) — always route a data-error BLOCK back to the data layer.

## Step 6 — QA

- **Invoke:** QA agent on the full doc plus pipeline state.
- **It does:** runs the test suite, validates output schema, checks the doc is well-formed and all mandatory artifacts exist.
- **Verdict:** `PASS` / `PASS_WITH_WARNINGS` / `FAIL`.
- **On PASS or PASS_WITH_WARNINGS:** proceed to ship. Log warnings in the retro.
- **On FAIL:** halt with the same authority as a DataSentinel FAIL. Route each failure to its owning agent, fix, re-run QA.

## Step 7 — Gaffer SHIP

Only after Pass 2 PASS + Skeptic PASS/PASS_WITH_CONCERNS + QA PASS/PASS_WITH_WARNINGS:

1. **Trust badge:** `python scripts/inject_trust_badge.py <doc> --date <ship-date>`. It counts the doc's genuine `[data]` tags and writes the verification line under the H1. (A doc with zero tagged stats gets no badge.)
2. **Provenance stamp** — append, on PASS only:
   ```
   <!-- council-pipeline:
     BriefBuilder@<sha>, Scientist@<sha>, FootyStrategy@<sha>,
     DataSentinel:PASS(pass1)@<ts>, DataSentinel:PASS(pass2)@<ts>,
     Skeptic:PASS@<ts>, QA:PASS@<ts>, Gaffer:SHIP@<ts>
   -->
   ```
   Refuse to advance any doc whose stamp is missing a tier or shows a non-PASS.
3. **Commit:** stage the brief by explicit filename, then `scripts/git_commit_safe.sh commit -m "<message>"`. Never `git add .`.
4. **Push:** `git push origin main`. Never `--force`.

## Step 8 — Chronicler

- **Invoke:** Chronicler agent after the push.
- **Pass:** the commit hash and cycle type `"brief-publish"`.
- **It produces:** the run report and top-3 expansion recommendations.

## Retro

Log one line to Gaffer memory: what broke, the top Chronicler recommendation, what to add to the backlog.
