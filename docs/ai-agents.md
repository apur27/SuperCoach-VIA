# Using AI Agents (Claude Code + Scientist)

> [← Back to main README](../README.md)

<!-- This file is part of the SuperCoach-VIA documentation. See README.md for the project overview. -->

**Who is this section for?** Anyone curious about Claude Code or this project's specialised Scientist agent - how to install, how to use it day-to-day, and how to use the Scientist for proper analytical work without burning your token budget.

This entire project is built and maintained using Claude. The two pages below cover everything from a fresh Ubuntu laptop to running statistical analyses on the prediction model.

## Pages in this section

- **[Setting up Claude Code on Ubuntu](claude-code-setup.md)** - Steps 1–7 (Node.js, install, login, Python venv, project open, verify, default-model setup) plus common troubleshooting.
- **[Using Claude and the Scientist agent](scientist-agent.md)** - the day-to-day playbook. When to use plain Claude, when to invoke the Scientist, the cost disclaimer, the improvement loop after every backtest.

> **Cost reminder:** the Scientist agent runs on **Claude Opus** and burns tokens fast. Plain Claude (no `@"Scientist (agent)"` prefix) handles 80% of work for a fraction of the cost. Read the [STOP. READ THIS FIRST.](scientist-agent.md#stop-read-this-first-do-not-waste-the-scientist) section before invoking the Scientist.

## Related

- [Installation & first-time setup](installation.md)
- [Running predictions & backtests](usage.md)
- [Live AFL insights](afl-insights.md)
