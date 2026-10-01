# Launch the AFL Tables reconciliation task

This package uses the existing agents in `.claude/agents/`. **Opus 5.5** handles
architecture and acceptance; **Sonnet 5.5** handles implementation and the full
reconciliation run. It builds on the existing integrity checker and agent memories.
No Claude agent or full source capture was launched while preparing it.

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
