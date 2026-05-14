# The Crumb - Phase 2: Production-Grade Multi-Agent Architecture

> [← Back to main README](../README.md) · [← Phase 1 build guide](footy-ai-chatbot-setup.md) · [← AI system architecture](ai-architecture.md)

**Who is this page for?** Engineers extending The Crumb beyond the Phase 1 prototype, and anyone using SuperCoach-VIA as a reference for production multi-agent design. Phase 1 (the 13-agent coaching staff) is documented in [footy-ai-chatbot-setup.md](footy-ai-chatbot-setup.md); read that first if you have not. This doc assumes you know how Phase 1 works and explains what Phase 2 changes, why, and how.

---

## 1. Executive summary

Phase 1 of The Crumb is a working 13-agent AFL coaching-staff simulation: a Senior Coach orchestrator on `claude-opus-4-7`, line coaches and analysts on `claude-sonnet-4-6`, a Data Steward on `claude-haiku-4-5`, all reasoning over the SuperCoach-VIA CSV dataset. It works. You ask a question, the Senior Coach calls the right specialists, merges their answers, and returns a data-grounded reply.

But Phase 1 works the way a well-run training session works - through cooperation, not enforcement. Agents stay in their lane because their system prompt *tells* them to. Agents hand each other free-text answers with no schema contract. The Data Steward runs arbitrary Python. There is no automated check that the system as a whole still behaves after a change. None of this is wrong for a prototype; all of it is a liability at production scale.

**Phase 2 makes the structure load-bearing.** It applies three patterns from real multi-agent deployments - the **Planner-Executor split**, the **role-based crew with IAM-style isolation**, and the **supervisor-worker graph** - through five concrete changes:

1. Split the Senior Coach into a **Planner**, **Executors**, and a data-blind **Synthesis** node.
2. Replace arbitrary code execution with **parameterised query templates**.
3. Wrap every agent-to-agent handoff in a validated **`FootyFinding` Pydantic envelope**.
4. Route low-confidence answers to a **human-in-the-loop (HITL)** review queue.
5. Add a nightly **evaluation harness** across five test categories.

The result keeps the legible markdown-in-repo character of the project while giving the same guarantees a cloud-IAM deployment would: an agent can only touch what its role allows, every handoff is schema-checked, and a regression is caught by a test rather than by a user.

This document is the full technical spec. The same patterns generalise far beyond football - see [section 9](#9-what-this-teaches-beyond-football).

---

## 2. The three João Moura patterns and why they apply

João Moura (CrewAI) has described a set of patterns from production multi-agent work. Three of them map directly onto The Crumb's gaps. Each also maps onto a football idea, which is not a gimmick - the football framing is how the non-engineer stakeholders of this repo reason about the design.

### 2.1 Planner-Executor split

**The pattern:** separate the component that decides *what to do* from the components that *do it*. The planner reads the goal and emits a structured plan; executors run plan steps; the planner never touches data, the executors never decide strategy.

**Why it applies here:** Phase 1's Senior Coach does both jobs in one Opus call - it decides which specialists to call *and* synthesises their answers *and* owns the final framing. When the answer is wrong, you cannot tell whether the *plan* was wrong (called the wrong specialists) or the *synthesis* was wrong (merged good findings badly). Splitting them makes each independently testable and independently debuggable.

**Footy analogy:** the match committee picks the plan on Thursday; the players execute it on Saturday. You do not want the player running the plan to also be rewriting it mid-quarter - that is how structure collapses into improvisation.

### 2.2 Role-based crew with IAM isolation

**The pattern:** each agent gets exactly the access its role requires - tools, credentials, data paths - and nothing more, enforced by infrastructure rather than instruction.

**Why it applies here:** Phase 1 enforces roles through the system prompt. The List Manager is *told* not to write tactical recommendations; nothing stops it. The Data Steward is *told* to be read-only; it runs arbitrary `subprocess` calls. Prompt-scoped boundaries are flexible and cheap, but they are honour-system boundaries. Phase 2 makes the boundary real: read-only mounts, a fixed tool surface per role, and a Synthesis node with no data access at all.

**Footy analogy:** the goal umpire can signal a goal; they physically cannot also bounce the ball at the centre. The role *is* the permission set - you do not rely on the goal umpire choosing not to bounce the ball.

### 2.3 Supervisor-worker graph

**The pattern:** model the orchestration as an explicit graph - declared nodes, declared edges, durable persisted state - so that a failed step can be replayed from its last good state instead of re-running the whole workflow.

**Why it applies here:** Phase 1's orchestration is implicit in an Opus reasoning chain. If the Senior Coach turn fails at specialist call four of six, the whole query restarts - all prior work is lost, and the failure leaves no inspectable state. A graph makes the handoffs durable: each node's output is persisted, a crash resumes from the last node, and the graph itself is the documented orchestration contract.

**Footy analogy:** a structured game plan with named phases and triggers, not a verbal "see how we go". When a phase breaks down you reset to the last set position - you do not restart the game from the first bounce.

---

## 3. Change 1 - Planner-Executor split of the Senior Coach

### 3.1 Phase 1 today

The Senior Coach (`claude-opus-4-7`) receives the user question, decides which of the 11 staff agents to call, dispatches them (in parallel where independent), receives free-text answers, and synthesises a final reply. One agent, three responsibilities, one opaque reasoning chain.

### 3.2 Phase 2 design

Three distinct nodes replace the monolith:

| Node | Model | Responsibility | Data access |
|------|-------|----------------|-------------|
| **Planner** | `claude-opus-4-7` | Parse the user question, emit a structured execution plan: which agents, what each is asked, dependency order | **None** |
| **Executor** (per agent) | `claude-sonnet-4-6` | Run one plan step - call one specialist agent with its scoped request | Via parameterised templates only |
| **Synthesis** | `claude-opus-4-7` | Merge validated `FootyFinding` envelopes into one coherent answer; own final framing | **None** |

The Planner produces a plan; it cannot see data, so it cannot smuggle a number into the plan. The Synthesis node combines findings; it cannot see data either, so it physically cannot fabricate a figure - it can only restate and reconcile what the Executors verified. The Executors are the only nodes with a path to data, and that path is the parameterised templates of [section 4](#4-change-2---parameterised-query-templates).

### 3.3 Sample Planner output

The Planner emits JSON, not prose. This is the contract the Executors consume:

```json
{
  "query_id": "q-20260515-0093",
  "user_question": "The coach wants to drop player B from the midfield rotation - should he?",
  "plan": [
    {
      "step": 1,
      "agent": "list_manager",
      "request": "Player B last 6 games: disposals, SC score, lineup status for the upcoming round",
      "depends_on": []
    },
    {
      "step": 2,
      "agent": "midfield_coach",
      "request": "Player B last 6 games: clearances, contested possessions, centre bounce attendance",
      "depends_on": []
    },
    {
      "step": 3,
      "agent": "opposition_analyst",
      "request": "Upcoming opponent: rank for opposition midfielder disposals allowed, last 4 rounds",
      "depends_on": []
    },
    {
      "step": 4,
      "agent": "stats_methodology",
      "request": "Is a 6-game window sufficient to call player B's form stable? Check sample size, leakage, effect size of any decline.",
      "depends_on": [1, 2]
    }
  ],
  "synthesis_instruction": "Answer contradicts likely coach intuition - require stats_methodology sign-off before Synthesis returns a recommendation. If confidence below threshold, route to HITL.",
  "parallel_groups": [[1, 2, 3], [4]]
}
```

The `parallel_groups` field tells the executor layer that steps 1-3 are independent and 4 depends on them. The `synthesis_instruction` is the Planner flagging, up front, that this is a challenge-the-coach answer and must clear the methodology gate.

### 3.4 Acceptance criteria

- Planner emits valid plan JSON for 100% of a 30-question test set; no plan step references a non-existent agent.
- Planner makes zero data-access tool calls (verified by the role-isolation eval category).
- Synthesis node makes zero data-access tool calls; its output cites only findings present in the envelopes it received.
- A crashed Executor step is replayable from persisted plan state without re-running completed steps.

---

## 4. Change 2 - Parameterised query templates

### 4.1 Phase 1 today

The Data Steward owns a `query_data` tool that executes an arbitrary pandas snippet via `subprocess` against the CSVs. It is read-only by convention (the agent is told not to write) and by a 30-second timeout - but the agent can author any code path it likes.

### 4.2 Phase 2 design

Replace `query_data` with a fixed registry of **parameterised query templates**. The agent fills parameters; it cannot author code. Each template is a named, schema-validated, audited function over the data layer.

```python
QUERY_TEMPLATES = {
    "player_form": {
        "params": {"player_id": "str", "n_games": "int (1-20)"},
        "returns": "DataFrame: last n_games rows for player, with file citation",
    },
    "player_career_aggregate": {
        "params": {"player_id": "str", "stat": "enum[disposals,goals,tackles,...]",
                   "from_year": "int", "to_year": "int"},
        "returns": "scalar aggregate + era-coverage flag",
    },
    "team_round_results": {
        "params": {"team": "str", "year": "int", "round": "int"},
        "returns": "DataFrame: match result rows + file citation",
    },
    "team_form": {
        "params": {"team": "str", "n_rounds": "int (1-15)"},
        "returns": "DataFrame: recent results + derived form markers",
    },
    "round_predictions": {
        "params": {"round": "int", "year": "int"},
        "returns": "DataFrame: model predictions for the round + MAE context",
    },
    "lineup_status": {
        "params": {"team": "str", "year": "int"},
        "returns": "DataFrame: named lineup for the current round",
    },
}
```

Each template:
- validates its parameters against the declared schema before touching disk;
- applies the era-coverage rules centrally (tackles from 1987, clearances and contested possessions from 1998, the 2017 hit-out recording change) - a request for an out-of-era stat is *refused at the template*, not left to the agent;
- returns its result already carrying the file path(s) read;
- emits a structured audit record (template name, parameters, caller agent, timestamp, row count).

There is no general code-execution path left. The Data Steward role shrinks to "pick the right template and fill its parameters" - and the entire class of arbitrary-code failures and injection surface is gone.

### 4.3 Acceptance criteria

- Zero arbitrary code-execution tool calls remain in the agent surface.
- Every template rejects out-of-schema parameters before any file read.
- An out-of-era stat request (e.g. tackles in 1985) is refused with a clear reason, verified by the era-coverage eval category.
- Every template call produces an audit record.

---

## 5. Change 3 - the `FootyFinding` Pydantic envelope

### 5.1 Phase 1 today

Agents hand each other free text. The Midfield Coach returns a paragraph; the Senior Coach reads the paragraph. There is no schema, so a missing citation or a dropped caveat is not caught - it is simply absorbed into the next step and surfaces, if at all, as a wrong final answer.

### 5.2 Phase 2 design

Every agent-to-agent handoff is a validated `FootyFinding` object. A handoff that does not validate is rejected at the boundary.

```python
from datetime import datetime
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


class ConfidenceTier(str, Enum):
    SETTLED = "settled"               # converging evidence, robust data
    PROBATIONARY = "probationary"     # converging evidence, exploratory data
    CONTESTED = "contested"           # genuine disagreement across lenses/agents
    INSUFFICIENT = "insufficient"     # neither data nor consensus support a finding


class DataCitation(BaseModel):
    file_path: str = Field(..., description="Exact repo-relative path read")
    rows_used: int = Field(..., ge=0)
    query_template: str = Field(..., description="Which parameterised template produced this")


class EraFlag(BaseModel):
    stat: str
    earliest_valid_year: int
    note: str = Field(..., description="e.g. 'tackles recorded from 1987 only'")


class FootyFinding(BaseModel):
    finding_id: str
    produced_by: str = Field(..., description="Agent role that authored this finding")
    query_id: str = Field(..., description="Links back to the Planner's plan")

    claim: str = Field(..., description="The finding itself, one or two sentences")
    confidence: ConfidenceTier

    citations: list[DataCitation] = Field(
        ..., min_length=1,
        description="At least one citation required - no uncited numeric claim",
    )
    era_flags: list[EraFlag] = Field(default_factory=list)
    caveats: list[str] = Field(
        default_factory=list,
        description="Upstream caveats propagate here unchanged",
    )

    tripwire: str | None = Field(
        None,
        description="Required for SETTLED/PROBATIONARY: an observable that would overturn the claim",
    )
    causal_language: Literal["associational", "causal"] = "associational"
    out_of_scope: str | None = Field(
        None, description="What this finding deliberately does not address",
    )

    produced_at: datetime

    def model_post_init(self, __context) -> None:
        # A high-confidence finding with no tripwire is automatically downgraded.
        if self.confidence in (ConfidenceTier.SETTLED, ConfidenceTier.PROBATIONARY):
            if not self.tripwire:
                self.confidence = ConfidenceTier.CONTESTED
                self.caveats.append("Auto-downgraded: no tripwire supplied.")
```

The envelope enforces the project's existing discipline as a *type*, not a guideline:

- `citations` has `min_length=1` - a finding with no data citation cannot be constructed.
- A `SETTLED` or `PROBATIONARY` finding with no `tripwire` is auto-downgraded to `CONTESTED` - the FootyStrategy rule, now mechanical.
- `caveats` and `era_flags` are lists that propagate forward unchanged - the Synthesis node receives every upstream caveat, it cannot quietly drop one.
- `causal_language` defaults to `"associational"` - a causal claim must be set explicitly and is visible to every downstream consumer.

### 5.3 Acceptance criteria

- 100% of agent handoffs are `FootyFinding` instances; any non-conforming handoff raises at the boundary.
- A finding with a numeric claim and no citation cannot be constructed (validated by a unit test).
- A `SETTLED` finding submitted without a tripwire is observed as `CONTESTED` downstream.
- Upstream caveats appear unchanged in the final synthesised answer (citation-precision eval category).

---

## 6. Change 4 - HITL routing on confidence threshold

### 6.1 Phase 1 today

Every answer is returned directly to the user, regardless of how shaky the evidence is. A `Contested` or `Insufficient Evidence` answer reaches the user with the same delivery path as a `Settled` one.

### 6.2 Phase 2 design

After the Synthesis node produces a final answer, a **routing gate** inspects the aggregate confidence:

```
                 ┌───────────────────┐
   findings ───▶ │  Synthesis node   │ ───▶  final answer + aggregate confidence
                 └───────────────────┘
                          │
                          ▼
                 ┌───────────────────┐
                 │  Routing gate     │
                 └─────────┬─────────┘
              confidence   │   confidence
              >= threshold │   < threshold
                  ▼        │        ▼
        ┌──────────────┐   │   ┌──────────────────┐
        │ Return to    │   │   │ HITL review queue │
        │ user directly│   │   │ (human signs off  │
        └──────────────┘   │   │  or revises)      │
                           │   └──────────────────┘
```

Aggregate confidence is the floor of the contributing findings' tiers, adjusted down further if findings disagree. The threshold is configurable. Routing triggers:

- aggregate tier is `CONTESTED` or `INSUFFICIENT`;
- any contributing finding carries an era-coverage refusal;
- the Planner's `synthesis_instruction` explicitly flagged the query as challenge-the-coach and the methodology gate did not return `SETTLED`.

A queued answer is held with its full `FootyFinding` chain attached, so the human reviewer sees exactly what evidence produced it. The reviewer signs off, revises, or rejects; the decision and reviewer identity are written to the audit log. High-confidence answers flow straight through - HITL is a gate on uncertainty, not a tax on every query.

### 6.3 Acceptance criteria

- An answer built from a `CONTESTED` finding is routed to the queue, not returned directly.
- An era-coverage refusal anywhere in the chain forces routing.
- A queued item carries its complete `FootyFinding` chain for reviewer inspection.
- Reviewer decision and identity are persisted to the audit log.

---

## 7. Change 5 - the evaluation harness

### 7.1 Phase 1 today

The ML model has a walk-forward backtest. The agent layer has nothing - LLM output quality is checked by ad-hoc human review after the fact.

### 7.2 Phase 2 design

A nightly regression suite over five test categories. Each category has a fixed set of cases with known-correct behaviour; a run scores pass/fail per case and the suite fails if any category regresses below its threshold.

| # | Category | What it checks | Example test case | Pass condition |
|---|----------|----------------|-------------------|----------------|
| 1 | **Citation precision** | Every number in the final answer traces to a real file via a real template call | "What is Marcus Bontempelli's disposal average over his last 6 games?" - then assert every figure in the answer maps to a `DataCitation` whose `file_path` exists | 100% of numeric claims cited; 0 fabricated figures |
| 2 | **Era-coverage refusal** | The system refuses stats that did not exist in the requested era | "Compare tackle counts for a 1975 player vs a 2020 player" | System refuses the 1975 side with a clear era note; does not invent a number |
| 3 | **Role isolation** | The Planner and Synthesis nodes never touch data; specialists stay in scope | Inspect the tool-call log for a full query run | Planner and Synthesis make 0 data-access calls; no specialist calls a template outside its declared set |
| 4 | **Calibration** | `SETTLED` answers hold up more often than `CONTESTED` ones, against later actuals | Take 20 prior `SETTLED` and 20 `CONTESTED` disposal-direction findings; score against the actual round results | `SETTLED` accuracy materially exceeds `CONTESTED` accuracy; if not, the tiers are not informative |
| 5 | **Falsifiability** | Every `SETTLED`/`PROBATIONARY` recommendation carries a real, observable tripwire | Parse all such findings from a query batch | 100% carry a non-empty tripwire that names a concrete observable, not a restatement of the claim |

Categories 1-3 and 5 are deterministic structural checks - cheap to run nightly. Category 4 is the slow one: it needs later actuals to score against, so it runs as a rolling check, scoring older findings as new round results land. The suite output is persisted as CSV alongside the ML backtest output, so agent-layer quality has the same diffable history the model already has.

### 7.3 Acceptance criteria

- All five categories run nightly (category 4 on a rolling basis) and persist results.
- A regression in any category fails the suite and is visible in the persisted history.
- The suite runs without a human in the loop.

---

## 8. Data isolation architecture - the local adaptation

João's role-based isolation pattern assumes cloud IAM: the platform issues role-scoped credentials and enforces them. SuperCoach-VIA runs locally on one machine, so Phase 2 reproduces the *guarantee* with three local mechanisms instead of cloud machinery.

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                          THE CRUMB - Phase 2                          │
  │                                                                        │
  │   ┌───────────┐         ┌───────────────┐         ┌────────────────┐  │
  │   │  Planner  │         │   Executors   │         │   Synthesis    │  │
  │   │           │         │  (per agent)  │         │                │  │
  │   │ NO DATA   │ ──plan──▶│               │─findings▶│   NO DATA      │  │
  │   │ ACCESS    │         │  data access  │         │   ACCESS       │  │
  │   └───────────┘         │  ONLY via     │         └────────────────┘  │
  │        ▲                │  templates    │                 │           │
  │        │                └───────┬───────┘                 │           │
  │        │                        │                         ▼           │
  │   user question                 │                  ┌──────────────┐   │
  │                                  │                  │ Routing gate │   │
  │                                  ▼                  └──────┬───────┘   │
  │                    ┌──────────────────────────┐            │           │
  │                    │  Parameterised query     │      pass  │  hold     │
  │                    │  template registry       │       ◀────┴────▶      │
  │                    │  - schema-validated      │     user      HITL     │
  │                    │  - era-coverage enforced │              queue     │
  │                    │  - audit-logged          │                        │
  │                    └────────────┬─────────────┘                        │
  └─────────────────────────────────┼──────────────────────────────────────┘
                                    │  READ-ONLY MOUNT
                                    │  (OS-enforced, not prompt-enforced)
                                    ▼
                    ┌──────────────────────────────────┐
                    │        DATA LAYER (read-only)     │
                    │  data/player_data/*.csv           │
                    │  data/matches/*.csv               │
                    │  data/lineups/*.csv               │
                    │  data/prediction/*.csv            │
                    └──────────────────────────────────┘
```

**The three mechanisms:**

1. **Read-only mount.** The data directory is mounted read-only to the agent process. The operating system enforces it. No agent - regardless of what its prompt says or what code it tries to run - can write to the data layer.

2. **Parameterised templates as the only data path.** There is no general code-execution tool. An agent reaches data exclusively through the fixed template registry, and the templates apply schema validation, era-coverage rules, and audit logging centrally. An agent cannot author a new way to touch disk.

3. **Data-blind Planner and Synthesis nodes.** The two nodes that decide *what to ask* and *how to frame the answer* have no data access at all. The Planner cannot smuggle a number into the plan; the Synthesis node cannot fabricate one into the answer. They can only route, and combine, what the Executors verified through templates.

Together these give the same property cloud IAM would: an agent can only touch what its role allows, and the boundary is enforced by the system, not by the agent's good behaviour.

---

## 9. What this teaches beyond football

The football is the demo. The architecture is the deliverable.

Every Phase 2 change is a general multi-agent production pattern, and the football instance is just the most legible way to show it working:

- **Planner-Executor split** - separating "decide what to do" from "do it" applies to any agent system where you need to debug *which* half failed: a research assistant, a customer-support crew, a code-review pipeline. The split is what makes each half independently testable.

- **Role-scoped data isolation** - read-only mounts, a fixed tool surface per role, and data-blind orchestration nodes give you cloud-IAM guarantees without cloud IAM. Any local or air-gapped multi-agent deployment needs this, and most reach for it too late.

- **Schema-validated handoffs (`FootyFinding`)** - a typed envelope on every agent-to-agent message turns "the agent forgot to cite its source" from a silent wrong answer into a validation error at the boundary. This is the single highest-leverage change for any multi-agent system handing structured findings between agents.

- **HITL routing on a confidence threshold** - gating *only the uncertain outputs* to a human, with the full evidence chain attached, is how you get human oversight without making a human review every response. The threshold is the dial between throughput and safety.

- **Nightly eval regression on the agent layer** - the ML world has had backtests for decades; the agent layer usually has nothing. Five deterministic structural checks plus one rolling calibration check give the agent layer the same diffable quality history a model already has.

SuperCoach-VIA is small enough to read in an afternoon and complete enough to show all five patterns interacting. That is the point of the repo. If you are building a multi-agent system and you want a reference implementation of these patterns that you can actually read end to end - this is it. The disposal predictions are just what makes it fun to look at.

---

## Related

- [Building The Crumb (Phase 1)](footy-ai-chatbot-setup.md) - the 13-agent staff this builds on
- [AI system architecture](ai-architecture.md) - the full system architecture, RAG, tool router, eval harness, sovereign deployment
- [How this repo uses Claude](how-this-repo-uses-claude.md) - policy-as-code, multi-agent orchestration, feedback governance
- [Main README](../README.md) - project overview, current eval results
