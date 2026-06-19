---
name: no-agent-dispatch-tool
description: Agent/council dispatch availability is SESSION-DEPENDENT — sometimes the Agent tool + full council are live, sometimes absent; check the system-reminder agent-types list each session before deciding mode
metadata:
  type: project
---

The Agent/council-dispatch capability is NOT stable across sessions. Check the `<system-reminder>` "Available agent types" block at the start of EACH session before choosing operating mode.

**Two observed states:**
- **Council LIVE (e.g. 2026-06-19 free-agency cycle):** the Agent tool is exposed and lists the full council — BriefBuilder, DataSentinel, FootyStrategy, Skeptic, Scientist, plus Explore/Plan/general-purpose. Run the GENUINE chain: commission Scientist for engineering+data derivation, FootyStrategy for prose, DataSentinel to gate, Skeptic for adversarial review. Stamp honestly as a real multi-agent chain.
- **Council ABSENT (earlier sessions):** no Task/Agent dispatch tool at all — `select:Task` and "agent dispatch" return nothing. Then run a SINGLE-OPERATOR cycle preserving the substance of every gate (deterministic Python number-matching = DataSentinel-self, adversarial control test = Skeptic-self) and stamp honestly as "single-operator cycle (no Agent dispatch tool available)" — NEVER fake a 7-agent chain.

**Why this matters:** the provenance stamp must reflect what actually happened. Faking the chain (claiming agents ran when they didn't, or vice versa) violates the ONE RULE. The mode is dictated by the tooling actually present, not by the contract's assumption that Agent always exists.

**How to apply:** First action each cycle — read the agent-types list. If the council is there, dispatch it. If not, single-operator with honest stamp. Never assume from a prior session. See [[council-stamp-gate-scope]] and [[enforcement-substrate-state]].
