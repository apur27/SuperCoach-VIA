# Using AI Agents (Claude Code + Scientist)

> [← Back to main README](../README.md)

<!-- This file is part of the SuperCoach-VIA documentation. See README.md for the project overview. -->

**Who is this section for?** Anyone curious about Claude Code or this project's specialised Scientist agent - how to install, how to use it day-to-day, and how to use the Scientist for proper analytical work without burning your token budget.

This entire project is built and maintained using Claude. The two pages below cover everything from a fresh Ubuntu laptop to running statistical analyses on the prediction model.

## Pages in this section

- **[Setting up Claude Code on Ubuntu](claude-code-setup.md)** - Steps 1–7 (Node.js, install, login, Python venv, project open, verify, default-model setup) plus common troubleshooting.
- **[Using Claude and the Scientist agent](scientist-agent.md)** - the day-to-day playbook. When to use plain Claude, when to invoke the Scientist, the cost disclaimer, the improvement loop after every backtest.

> **Cost reminder:** the Scientist agent runs on **Claude Opus** and burns tokens fast. Plain Claude (no `@"Scientist (agent)"` prefix) handles 80% of work for a fraction of the cost. Read the [STOP. READ THIS FIRST.](scientist-agent.md#stop-read-this-first-do-not-waste-the-scientist) section before invoking the Scientist.

## FootyStrategy agent

The FootyStrategy agent is a tactical AFL brainstorming specialist. Where the Scientist reads data and runs analysis, FootyStrategy answers the question the numbers raise but cannot answer: *what does a coach actually do about this?*

**Invoke with:** `@"FootyStrategy (agent)"` in a Claude Code session.

**What it does:**
- Tactical pattern analysis: zone defences, tagging conventions, ruck-rotation, half-time structural resets
- List-construction analysis: evaluates each AFL team's list by quality tier and draft pick pedigree (pick 1, father-son, rookie, trade, etc.)
- Cross-team vulnerability mapping: where is a team one injury away from crisis? How does list depth explain tactical identity?
- Post-match brainstorm: converts data anomalies into coaching questions and answers them from football knowledge

**Best used:**
- After a Scientist post-match analysis to interpret structural findings tactically
- When planning matchup zones against an upcoming opponent
- When you want to know why a team plays the way it does (list-driven tactical identity)
- To generate the "what does the opposition coach actually do at half-time" scenarios the data can't surface

**Worked examples:**
- Tactical brainstorm with Scientist: [Richmond vs Adelaide R9 full-time verdict](coaches-strategy-corner/richmond-vs-adelaide-round-9-2026-full-time-verdict.md) - see the "Strategic brainstorm - Scientist x FootyStrategy" section
- All-18-club list analysis: [AFL 2026 list quality and draft pipeline](news/2026-06-17-afl-2026-list-quality-draft-pipeline.md)

**Data limitations:** Same as Scientist - no GPS, no video, no positional data. FootyStrategy brings football knowledge, not additional data sources.

## Related

- [How this repo uses Claude](how-this-repo-uses-claude.md) - custom agent design, policy-as-code, feedback governance, multi-agent orchestration (portfolio/showcase doc)
- [Installation & first-time setup](installation.md)
- [Running predictions & backtests](usage.md)
- [Live AFL insights](afl-insights.md)
