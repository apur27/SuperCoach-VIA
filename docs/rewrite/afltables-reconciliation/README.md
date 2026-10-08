# Launch the AFL Tables reconciliation task

This package uses the existing agents in `.claude/agents/`. **Opus 5.5** handles
architecture and acceptance; **Sonnet 5.5** handles implementation and the full
reconciliation run. It builds on the existing integrity checker and agent memories.
The launch package was prepared before implementation. Its current execution state is
recorded below; the original launch instructions follow for reference.

## Status — 7 October 2026

The reconciler and corrections are accepted with conditions and merged into `main`
(delivery `1add8cdc5`, corrections `67217df40`). The corrected data verdict remains
**UNKNOWN: zero confirmed discrepancies and 63 unresolved source cells per layer**.
Acceptance of code does not establish a global data PASS or activate the new pipeline.

The full parent report is `81ac70ba8aa25110119a91779d2c2df91088b2f0f87408db3546de8109a4509c`,
for snapshot `sha256:f1abd8c2b7f8d6b3812e73f91a4403add1844f3ad9037139e7dc373a0ed91703`.
The candidate's later **2026-only** report is
`6fecac8e1c8eb804774eb70a5f9d03bd5448ee86d43d1253472ca0f1ea7d9c2a`, for snapshot
`sha256:b9830cbf1093544e234f26159440a2cc84eb834a223da13d05e931ae8f5815c4`.
The [composite source-coverage record](../../reviews/evidence/afltables-candidate-source-coverage-20261007.json)
retains their actual scopes and identities and verifies unchanged historical fragments.
It is not a new full audit of the child; the unresolved historical cells remain open.
Local source evidence is under `var/reconciliations/afltables/2026-10-05-corrected/`
and `var/reconciliations/afltables/2026-10-06-gate-live/` in the main checkout.

The hosted preview still uses `3de65975…` and its retained FAIL report `58eff517…`.
The [provisional Hall of Fame tables](../../hall-of-fame/provisional/README.md) now
use corrected candidate `b9830cbf…` and its pinned UNKNOWN composite source coverage;
regeneration is not a new full audit. Corrected candidate publication needs its own
release, input-consistency report and sealed site; pointer
promotion and pipeline activation remain separate owner decisions.

The [run record](../../reviews/AFLTABLES_RECONCILIATION_RUN.md) and
[acceptance record](../../reviews/AFLTABLES_RECONCILIATION_ACCEPTANCE.md) preserve the
original review plus dated follow-up. Gate skip/warn persistence, re-gates after fixes,
pending-season retries and smoke diff hashes landed in `238cacf46`; the source-backed
fix path passed a scratch smoke, and the 6 October weekly cycle completed. The fast
hook tier passed 2,127 tests in 65.7 s on four workers on 7 October; documented targets
remain missed and unchanged pending the owner's budget decision.

Preserve the original Claude worktree, its unmerged memory and local evidence until
custody is checked. The original launch instructions below are historical task briefs,
not instructions to restart completed acquisition or acceptance.

Read [the design](DESIGN.md), especially section 14's definition of done.
The [architect payload](ARCHITECT_PROMPT.md) and [engineer payload](ENGINEER_PROMPT.md)
give the existing agents task-specific deliverables. They are task briefs, not replacement
agent definitions. There is no additional agent registry to install.

## Existing roles

| Agent | Assignment for this task |
|---|---|
| [Gaffer](../../../.claude/agents/Gaffer.md) | Coordinates the architecture review and final acceptance on Opus 5.5; owns design edits, review packaging, serialized commits and delivery to main |
| [Surveyor](../../../.claude/agents/Surveyor.md) | Independently reviews architecture and final evidence on Opus 5.5; writes its own survey and routes findings to owners |
| [Scientist](../../../.claude/agents/Scientist.md) | Implements, tests and executes the deterministic Python audit on Sonnet 5.5; owns data methodology and numeric findings |
| [QA](../../../.claude/agents/QA.md) | Runs applicable test, schema and artifact checks independently before delivery |
| [DataSentinel](../../../.claude/agents/DataSentinel.md) | Verifies tagged statistical claims when required by the existing document gates |

Surveyor remains advisory and does not edit the implementation or task design. Gaffer
records the architecture decision using the actual Surveyor findings; it does not author
player statistics or invent another agent's verdict. Scientist hands code and findings
back to Gaffer for commits. Existing gates retain their current scopes.

The commands below were checked against local Claude Code **2.1.284** help. Exact model
availability still depends on the account/provider and must be verified at launch.
The explicit model choices below are the user's overrides **for this task only**.
For example, Scientist's stored default is `opus` and Surveyor's is `fable`; this task
requests Sonnet 5.5 and Opus 5.5 respectively. Do not edit those defaults or shadow their
definitions with `--agents`. Gaffer must also pin and verify the model when dispatching
Surveyor: a parent model selection alone does not override a child's configured model.
Anthropic documents session agent selection and per-invocation model selection.
[Model configuration](https://code.claude.com/docs/en/model-config),
[custom agents](https://code.claude.com/docs/en/sub-agents).

## Step 1 Opus design review

Run from the repository root:

```bash
cd /home/abhi/git/SuperCoach-VIA
claude --agent Gaffer --model claude-opus-5-5 --effort high \
  "Follow docs/rewrite/afltables-reconciliation/ARCHITECT_PROMPT.md in design review mode. Use the existing Surveyor agent on Opus 5.5 for independent architecture review, resolve findings through their existing owners and write the approval artifacts. Do not launch engineering."
```

Opus must write an APPROVED review with no blocking findings and the final design hash.
If it requests changes, resolve them and repeat its review before starting Sonnet. This
is the architect-confirmation step requested for this task.

## Step 2 Sonnet implementation and full execution

After the design is approved:

```bash
cd /home/abhi/git/SuperCoach-VIA
claude --agent Scientist --model claude-sonnet-5-5 --effort high \
  "Follow docs/rewrite/afltables-reconciliation/ENGINEER_PROMPT.md. Verify Opus approval, implement the approved design with tests, and execute the full all-player reconciliation. Produce reproducible reports and an honest PASS, FAIL or UNKNOWN verdict."
```

This launches a real implementation/execution task. Full acquisition can take hours;
the design requires checkpoints, progress, bounded requests and an exact resume command.
Scientist should use one task worktree, preserve existing data and complete the offline
comparison even when the data has genuine failures. No model evaluates individual
statistics at runtime; the implemented Python rules decide the result.

## Step 3 Opus final acceptance

Run this from the implementation worktree Scientist reports, so Gaffer and its reviewers
see the final code and artifacts.

```bash
claude --agent Gaffer --model claude-opus-5-5 --effort high \
  "Follow docs/rewrite/afltables-reconciliation/ARCHITECT_PROMPT.md in final acceptance mode. Commission the existing Surveyor on Opus 5.5 and QA for independent checks, route fixes to Scientist on Sonnet 5.5, and deliver accepted work to main through the existing gates. Separate software acceptance from the data verdict."
```

Gaffer owns the accepted code's main-branch delivery and cleanup. No parallel commits.
Keep the original failing data reports. Accepted software may correctly report data FAIL;
a partial run must never become an all-player PASS through wording or exclusions.

## What the finished audit must establish

- Every player in the independent source census and every local player is accounted for.
- Appearance membership and supported statistics are compared game by game, as well as
  season/career totals, observed denominators and applicable averages.
- Identity ambiguity, unavailable historical statistics, network failures and conflicting
  source pages remain visible and prevent an unsupported integrity claim.
- The captured source and pinned local inputs reproduce identical reports offline across
  cold, warm, changed-since and different worker-count runs.
- Primary app data, legacy CSVs and the sealed release receive separate verdicts.

The current local candidate and retained reference are different inputs; their verified
paths and IDs are in design section 11. The legacy CSVs are older migration inputs.
Do not infer that an old local preview or deployed site contains the candidate being audited.
