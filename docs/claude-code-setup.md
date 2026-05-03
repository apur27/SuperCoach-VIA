# Setting up Claude Code on Ubuntu

> [← Back to main README](../README.md) · [← AI Agents hub](ai-agents.md)

<!-- This file is part of the SuperCoach-VIA documentation. See README.md for the project overview. -->

**Who is this page for?** A fresh Ubuntu laptop where you want Claude Code installed and running so you can ask plain-English questions about your AFL data, run predictions, or improve the model.

This page walks from zero to having Claude Code running, talking to your repo. Allow ~20 minutes for the first run-through.

This section walks you through getting Claude Code running on a fresh Ubuntu laptop — from zero to having an AI agent that can read your AFL data, write code, and improve the prediction model.

## What you need before you start

- An Ubuntu laptop (20.04 or newer) — a gaming laptop works great, specs don't need to be high for Claude itself
- A Claude subscription — go to [claude.ai](https://claude.ai) and sign up; the **entry level (Pro) plan is enough** for everything in this repo
- Internet connection

---

## Step 1 — Install Node.js

Claude Code runs on Node.js. Install it via the official NodeSource repository (the version in Ubuntu's default package manager is often too old):

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

Verify:
```bash
node --version   # should show v20 or higher
npm --version
```

---

## Step 2 — Install Claude Code

Claude Code is installed as a global npm package:

```bash
npm install -g @anthropic-ai/claude-code
```

Verify:
```bash
claude --version
```

---

## Step 3 — Log in to your Claude account

```bash
claude login
```

This opens a browser window. Log in with the same account you used to sign up at claude.ai. Once authenticated, your terminal will confirm you're logged in.

---

## Step 4 — Install Python and project dependencies

This repo uses Python 3.12. Set up a virtual environment:

```bash
sudo apt install -y python3.12 python3.12-venv python3-pip

# Create a virtual environment
python3.12 -m venv ~/.venv/afl
source ~/.venv/afl/bin/activate

# Install project dependencies
cd /path/to/SuperCoach-VIA
pip install -r requirements.txt
```

To activate the environment every time you open a terminal, add this to your `~/.bashrc`:
```bash
echo 'source ~/.venv/afl/bin/activate' >> ~/.bashrc
source ~/.bashrc
```

---

## Step 5 — Open the project in Claude Code

```bash
cd /path/to/SuperCoach-VIA
claude
```

You'll see the Claude Code prompt. You're now inside an AI-powered terminal that understands your entire codebase. Try:

```
what does this project do?
```

Claude will read the code and explain it to you in plain English.

---

## Step 6 — Verify everything works end to end

```bash
# In the Claude Code prompt, type:
run the prediction for next round
```

Claude will give you the exact command. Run it. If it works, you're set up correctly.

---

## Step 7 — Setting the default model (save your token budget)

By default, Claude Code uses whatever model your account is set to — and on most plans that's the most powerful (and most expensive) model available. For day-to-day work in this repo — asking questions, fixing small bugs, pushing to git, updating the README — you don't need that horsepower. The cheaper Sonnet model handles all of it just as well, and burns through your monthly token budget much more slowly.

The Scientist agent is different. It needs the extra brainpower of Opus for proper statistical analysis, so it's already configured to use Opus regardless of your default. Setting Sonnet as your default means your everyday Claude chats are cheap, and only the Scientist costs you the heavy tokens — which is exactly what you want.

To set Sonnet as your default, run this once:

```bash
# Create or edit ~/.claude/settings.json
mkdir -p ~/.claude
cat > ~/.claude/settings.json << 'EOF'
{
  "model": "sonnet"
}
EOF
```

That's it. From now on, every plain Claude session uses Sonnet. **You don't need to do anything special to invoke the Scientist on Opus** — it's already configured in `.claude/agents/Scientist.md` to override the default for that one agent.

To verify which model you're on, type this inside any Claude Code session:

```
/model
```

It'll print the current model. You can also use `/model` to switch interactively if you ever want to bump a single session up to Opus without changing the default.

| Task | Model used | How |
|------|-----------|-----|
| Everyday Claude (questions, git, README) | Sonnet | Default in `~/.claude/settings.json` |
| Scientist agent | Opus | Configured in `.claude/agents/Scientist.md` |

---

## Troubleshooting common issues

| Problem | Fix |
|---------|-----|
| `claude: command not found` | Run `npm install -g @anthropic-ai/claude-code` again; make sure `/usr/bin` is in your PATH |
| `claude login` doesn't open a browser | Copy the URL it prints and paste it into your browser manually |
| Python packages fail to install | Make sure your venv is activated: `source ~/.venv/afl/bin/activate` |
| `ModuleNotFoundError` when running scripts | You're using the wrong Python — use the full venv path: `/path/to/.venv/afl/bin/python prediction.py` |
| Claude says it can't find files | Make sure you're in the repo directory when you start Claude: `cd SuperCoach-VIA && claude` |

---


---

## Related

- [Using the Scientist agent](scientist-agent.md)
- [Installation & first-time setup](installation.md)
- [Running predictions & backtests](usage.md)
