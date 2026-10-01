# AFL Tables reconciliation architecture task for Gaffer and Surveyor

Run as the existing **Gaffer** agent, with **Surveyor** providing independent architecture
review. Read their actual `.claude/agents/*.md` definitions and relevant project memories.
Use **Opus 5.5** (`claude-opus-5-5`) for Gaffer and Surveyor on this task. These explicit
user model choices override their defaults for these invocations only. Confirm each
selected model from session metadata. Report an access
restriction instead of silently substituting another model. Record requested and resolved
model identifiers; if the provider does not expose the resolved ID, say so explicitly.

Read `CLAUDE.md`, any applicable local instructions, and
`docs/rewrite/afltables-reconciliation/DESIGN.md` in full. This payload is an instruction
to review a concrete design and produce a usable decision, not merely summarize it.
Use the existing Claude/Grok/Sol work already in main.

## Role ownership

Use registered agents by their existing names. Do not define new architect/engineer
agents or copy their system prompts into a CLI registry. This document supplies the
assignment; `.claude/agents/` remains the authority for roles, tools and memories.

- **Surveyor:** commission a DEEP survey of the proposed reconciliation and its failure
  paths, explicitly passing Opus 5.5 as the invocation model. It writes its own survey
  under `.claude/surveys/` and routes evidence-backed findings. It does not edit this
  design, prompts, production code or other review documents. Its findings are advisory.
- **Gaffer:** package the design decision and task review documents, make design/prompt
  edits within your remit, and sequence owners' work. Do not author statistical figures.
  Only you commit/push, using the safe wrapper after writers have stopped.
- **Scientist:** consult for source semantics, numeric evidence and engineering feasibility;
  any such consultation uses the user's Sonnet 5.5 selection. It owns implementation and
  later fixes. Design review must not start bulk capture or production implementation.
- **QA:** independently verifies applicable tests/contracts in final acceptance.
  DataSentinel verifies tagged numeric claims when required by existing gates. Neither
  Surveyor's opinion nor Gaffer's summary substitutes for their real verdicts.

After each Surveyor run, Gaffer preserves an exact, hashed copy of its report under
`docs/reviews/afltables-reconciliation/` with a unique run/phase name before another
survey can reuse the date-based path. Keep design and acceptance evidence separately.
Record QA's actual interpreter and commands: its older examples contain a machine-specific
Python path and counter-based legacy checks. Use the working locked environment for this
task, and validate the new auditor against this design's appearance/coverage contracts.
Do not change the shared agent definitions or treat a legacy spot-check as full coverage.

Read `.claude/agent-memory/Gaffer/project_no_agent_dispatch_tool.md`, then verify the
current session's actual dispatch tools and available agents. Do not assume availability
from memory. If dispatch is unavailable, give the exact separate-session command using
the existing agent and record the missing review; do not claim that agent ran. This task
requires an actual independent review before acceptance. Do not enable bypass permissions.

## Design review mode

1. Record HEAD, working-tree status, applicable instructions and active pipeline state.
   Hash the participating `.claude/agents/` definitions and identify memories consulted.
   Preserve unrelated changes. Inspect the files named in design section 3 and their
   tests. Do not assume dated review conclusions still match the code.
2. Commission Surveyor to inspect representative source HTML using bounded read-only requests and archived
   fixtures. Cover an early career, modern career, identical names, a club transfer,
   substitution markers, unavailable historical statistics, and a drawn/replayed final.
   This is a parser/identity feasibility review, not the full network census.
3. Have Surveyor challenge the design against every T01–T30 case. Particularly investigate source
   directory completeness, profile identity, game links, missing DOBs, rowspans, source
   averages, missing-stat exceptions, unplayed substitutes and conflicting source pages.
   Confirm what can and cannot be established from each page type.
4. Check transport reuse carefully: current URL policy lacks census/notes paths; current
   retries cap Retry-After; current archives contain mutable validator files; current
   player-source check is optional and only discovers snapshot observations. The design
   must address these rather than claiming the existing code already does the full job.
5. Gaffer resolves routine architecture choices using the review and owning-agent input,
   then edits the design. Surveyor keeps its read-only role. Preserve the design's
   complete population requirement, deterministic comparison, input immutability and
   strict treatment of missing evidence. Record material changes and reasons. A smaller
   implementation is welcome if it satisfies every requirement.
6. Write `docs/reviews/AFLTABLES_RECONCILIATION_DESIGN_REVIEW.md` and a machine-readable
   `docs/reviews/afltables-reconciliation-design-review.json`. The review must contain:
   - `decision`: `APPROVED` or `CHANGES_REQUIRED`;
   - requested/resolved models per agent, reviewed HEAD and relevant code-path/hash inventory;
   - exact agent-definition hashes, consulted memory paths and Surveyor report hash;
   - SHA-256 of the final approved `DESIGN.md`, and hashes of the two role payloads;
   - actual files/source samples inspected and source URLs/capture hashes where available;
   - findings with severity, exact location, reason, disposition and required test;
   - all T01–T30 cases mapped to proposed test names/modules;
   - explicit decisions for scope, identity fallback, blank evidence, source conflicts,
     performance targets, CLI contracts and completeness accounting;
   - `blocking_findings`: an empty list is required for APPROVED.
7. If approved, provide a short implementation order and the exact Scientist/Sonnet launch command
   from README.md. Do not launch Sonnet or begin bulk acquisition in this review session.
   The user has requested an architect-confirmed design before engineering execution.

Code inspection and short probes are allowed. Keep production code, data, releases,
source observations, default harness, hooks and schedules unchanged in design review.
Gaffer may edit review documents and the design; Surveyor writes only its own survey and
memory. Gaffer uses the required safe commit wrapper if committing documentation.
Approval is a model review, not owner consent to modify
statistics or activate production.

Do not ask the user to decide ordinary implementation details. If a necessary policy
cannot be determined from evidence, leave a precise blocking finding, explain its impact,
and distinguish it from a routine question you can resolve yourself.

## Final acceptance mode

When launched with "final acceptance", commission Surveyor for a fresh independent
review and QA for the applicable test/schema/artifact checks. Read the approved design,
architecture review, Scientist's implementation diff, test results and full-run artifacts. Do not rely on the
engineer's prose or a prior PASS summary alone.

Check D01–D18 individually. Recompute report/stream hashes and coverage identities. Read
representative findings back to raw source cells and local rows. Inspect failure paths,
cache keys, source and local input drift, output aliases, partial census accounting and
the separation between source logic and production import logic. Run bounded adversarial
tests and an offline replay when practical; reuse the frozen corpus rather than starting
another mass fetch.

Gaffer writes `docs/reviews/AFLTABLES_RECONCILIATION_ACCEPTANCE.md` plus
`docs/reviews/afltables-reconciliation-acceptance.json`, bound to implementation commit,
code hashes, approved design hash, plan and capture manifests, output hashes and actual
Surveyor/QA evidence. Use Scientist's verified numeric outputs rather than authoring new
statistical claims. Run DataSentinel on tagged documents when its gate applies. Do not
run the football-publication chain for an engineering plan or invent a council stamp.
Use one of `ACCEPTED`, `CHANGES_REQUIRED`, `EXECUTION_INCOMPLETE` for delivery, and retain
the independent data verdicts `PASS`, `FAIL`, `UNKNOWN`. A correctly detected data FAIL
can coexist with accepted software. A partial source capture cannot be called a completed
all-player audit.

Do not repair statistics as part of reviewing them. If engineering changes are needed,
route exact findings and failing examples to Scientist on Sonnet 5.5. Routine corrections inside the approved
design do not require a new architecture cycle. Material scope or semantic changes do.

After acceptance and applicable gates, Gaffer commits via `scripts/git_commit_safe.sh`,
delivers the verified code to main and removes the merged task branch, preserving dirty
or divergent work and necessary audit artifacts. A data FAIL discovered by correct audit
software is separate from QA FAIL; it does not authorize a statistics correction or a
production publish. Preserve the failing report and state the data status honestly.

## Final answer format

State the mode and decision first. Link the review. Give blocking findings, if any, and
the evidence that closes resolved findings. Say what was actually executed, what remains
unverified, and the next role's concrete task. Do not claim a running agent exists unless
you started it and verified its process/session.
