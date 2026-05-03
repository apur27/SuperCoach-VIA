# Troubleshooting and common questions

> [← Back to main README](../README.md)

<!-- This file is part of the SuperCoach-VIA documentation. See README.md for the project overview. -->

**Who is this page for?** Anyone who has run into an error or has a "wait, can it do X?" question. Quick fixes for the most common issues.

**"It says 'command not found'"**  
You're probably not inside the `SuperCoach-VIA` folder. Run `cd SuperCoach-VIA` first, then try again.

**"prediction.py takes forever or crashes"**  
Use the CPU version: `python prediction_cpu.py`. It's slower but works on any laptop without a GPU.

**"I don't have Ubuntu / I'm on Windows or Mac"**  
The project works on Windows too — install [Git for Windows](https://git-scm.com/download/win) and [Python 3.10+](https://python.org/downloads), then follow the same steps. On Windows you can also use WSL (Windows Subsystem for Linux) for a full Ubuntu experience inside Windows.

**"ModuleNotFoundError" or "No module named …"**  
Your Python environment isn't active. Run `source .venv/bin/activate` (or `pip install -r requirements.txt` to install everything fresh), then try again.

**"How do I update every week after a new round?"**  
Run `./refresh_and_rank.sh` after each round finishes. It downloads the latest data, recalculates rankings, updates predictions and rebuilds the charts in this README automatically.

**"Can I see how accurate the model was last round?"**  
Yes — the backtest results live in `data/prediction/backtest/`. Open any CSV in Excel to see predicted vs actual disposals for every player.

**"Can I run this on my phone?"**  
Not yet — this is a local tool you run on a laptop. A browser-based version with no setup required is on the roadmap (see below).

**"Something broke and I don't know how to fix it"**  
Open a terminal in the project folder and type `claude` — then describe what went wrong in plain English. Claude will diagnose and fix it for you.


---

## Related

- [Installation & first-time setup](installation.md)
- [Running predictions & backtests](usage.md)
- [Claude Code setup on Ubuntu](claude-code-setup.md)
