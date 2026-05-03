# Using AI Agents (Claude Code + Scientist)

> [← Back to main README](../README.md)

<!-- This file is part of the SuperCoach-VIA documentation. See README.md for the live data sections. -->


**Who is this section for?** Anyone curious about Claude Code or the project's specialised Scientist agent — how to install Claude Code on Ubuntu, how to use it day-to-day, and how to use the Scientist for proper analytical work without burning your token budget.

Learn how this entire project was built using Claude. The setup steps walk through getting Claude Code running on Ubuntu, and the second sub-section covers day-to-day use of plain Claude and the Scientist agent.

### Setting up Claude Code on Ubuntu

This section walks you through getting Claude Code running on a fresh Ubuntu laptop — from zero to having an AI agent that can read your AFL data, write code, and improve the prediction model.

#### What you need before you start

- An Ubuntu laptop (20.04 or newer) — a gaming laptop works great, specs don't need to be high for Claude itself
- A Claude subscription — go to [claude.ai](https://claude.ai) and sign up; the **entry level (Pro) plan is enough** for everything in this repo
- Internet connection

---

#### Step 1 — Install Node.js

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

#### Step 2 — Install Claude Code

Claude Code is installed as a global npm package:

```bash
npm install -g @anthropic-ai/claude-code
```

Verify:
```bash
claude --version
```

---

#### Step 3 — Log in to your Claude account

```bash
claude login
```

This opens a browser window. Log in with the same account you used to sign up at claude.ai. Once authenticated, your terminal will confirm you're logged in.

---

#### Step 4 — Install Python and project dependencies

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

#### Step 5 — Open the project in Claude Code

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

#### Step 6 — Verify everything works end to end

```bash
# In the Claude Code prompt, type:
run the prediction for next round
```

Claude will give you the exact command. Run it. If it works, you're set up correctly.

---

#### Step 7 — Setting the default model (save your token budget)

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

#### Troubleshooting common issues

| Problem | Fix |
|---------|-----|
| `claude: command not found` | Run `npm install -g @anthropic-ai/claude-code` again; make sure `/usr/bin` is in your PATH |
| `claude login` doesn't open a browser | Copy the URL it prints and paste it into your browser manually |
| Python packages fail to install | Make sure your venv is activated: `source ~/.venv/afl/bin/activate` |
| `ModuleNotFoundError` when running scripts | You're using the wrong Python — use the full venv path: `/path/to/.venv/afl/bin/python prediction.py` |
| Claude says it can't find files | Make sure you're in the repo directory when you start Claude: `cd SuperCoach-VIA && claude` |

---

### Using Claude and the Scientist agent

Once Claude Code is set up (see above), you interact with it entirely in plain English — no commands to memorise, no code to write. This section covers how to get the most out of it, and specifically how to use the Scientist agent to improve the prediction model.

---

> ## **STOP. READ THIS FIRST. DO NOT WASTE THE SCIENTIST.**
>
> ### **The Scientist is expensive. Use it wisely.**
>
> The Scientist agent runs on **Claude Opus** — the most powerful and **most expensive** Claude model that Anthropic ships. Every time you invoke the Scientist, you are spending **a lot more tokens** than a normal Claude conversation would use.
>
> If you are on an **entry-level Claude subscription** (Pro, or any plan with a monthly token budget), you have a **finite number of tokens per month**. Burning them on trivial questions to the Scientist is the fastest way to **run out before the end of the month** and find yourself unable to use Claude at all.
>
> **The rule is simple:**
>
> > **Use the Scientist ONLY when you need it to read your data and your code, and do something analytical with it.**
> >
> > **For literally everything else, just talk to plain Claude — or use Google.**
>
> Plain Claude (no `@"Scientist (agent)"` prefix) is already extremely capable. It can answer questions, explain code, write small scripts, and chat with you — all on a much cheaper model. Save the Scientist for the heavy lifting it was built for.
>
> **Tip:** set Sonnet as your default model (see [Step 7 in the setup section](#step-7--setting-the-default-model-save-your-token-budget)) so only the Scientist uses Opus. This is the single biggest thing you can do to make your token budget last the month.

#### **Good questions for Scientist vs. Don't waste Scientist on this**

| Good questions for Scientist (worth the cost) | Don't waste Scientist on this (use plain Claude or Google) |
|---|---|
| Analyse the backtest and improve the model | What is the weather today? |
| Why is the model consistently under-predicting midfielders? | Who won the 2024 AFL grand final? |
| Find any data leakage in the prediction pipeline | What does `print()` do in Python? |
| The backtest shows a bias of -1.3 — find the root cause and fix it | How do I open a CSV file in Excel? |
| Optimise `prediction.py` without changing the results | What is machine learning? |
| Which teams is the model most wrong about, and why? | Can you write me a haiku about footy? |
| Round 1 accuracy is always worse than other rounds — investigate and fix it | What time is it? |
| Check `prediction.py` for bugs that could cause wrong predictions | How many players are in an AFL team? |
| Run a statistical analysis on the backtest errors and identify the biggest contributors | Explain what a `for` loop is |
| Look for feature engineering improvements based on the residuals | What's the capital of Australia? |
| Compare the last three backtests and tell me whether MAE is genuinely improving or just noise | Can you summarise the README for me? |
| Investigate why predictions for a specific player are systematically off | Say hello |

#### **A simple rule of thumb**

Ask yourself one question before typing `@"Scientist (agent)"`:

> **"Does answering this require Claude to actually open my CSV files, read my Python code, and do real analytical work on it?"**
>
> - **Yes** → Use the Scientist. This is what it's for.
> - **No** → Use plain Claude. Or Google. Or a calculator. Anything cheaper.

If you just want a chat, a definition, a quick code explanation, or general help — **don't @ the Scientist**. Just type your question normally. Plain Claude will handle it for a fraction of the cost.

Treat the Scientist like calling in a senior consultant. You wouldn't pay a consultant $500/hr to tell you what time it is. Same idea here.

---

#### First time using Claude? Here's exactly what to do

Never spoken to an AI before? Here is the shortest possible path from "zero" to "asking Claude a question about footy data". You only need to do steps 1 to 4 once.

1. Go to [claude.ai](https://claude.ai) and create a free account (the entry-level subscription is enough for most tasks in this repo).
2. Open a terminal on your laptop (press `Ctrl+Alt+T` on Ubuntu, or search "Terminal" in your apps).
3. Navigate to this folder: type `cd ~/git/SuperCoach-VIA` and press Enter. (`cd` stands for "change directory" — it tells the terminal which folder to work in.)
4. Start Claude Code by typing `claude` and pressing Enter. The first time only, it'll ask you to log in via your browser — follow the link it prints.
5. You'll see a `>` prompt. **Type your question in plain English** — no special syntax — and press Enter. For example:
   - *"Who are the top 5 disposal getters in 2026 and what does the data say about their Brownlow chances?"*
   - *"My favourite team is Hawthorn. What do their stats say about their chances of making the grand final?"*
   - *"I play SuperCoach. Which players should I trade in this week based on the prediction model?"*
6. Claude will read the data files in this project and answer you — no coding needed. If it wants to run a script or change a file, it'll tell you what it's about to do first.

That's the whole loop. Open a terminal, type `claude`, ask a question. From there, the rest of this section is about making the most of it (and not burning your token budget on the Scientist).

---

#### How Claude Code works

Start it from the project folder:

```bash
cd /path/to/SuperCoach-VIA
claude
```

You'll see a `>` prompt. Type what you want in plain English. Claude reads every file in the project and can run commands, edit code, and explain what it's doing — all in response to plain-English instructions.

```
what does this project do?
run the prediction for next round
the backtest crashed — fix it
push my changes to main
```

Press `Enter` to send. Claude will respond, and if it needs to run a command or edit a file, it will ask for your permission first (or just do it, depending on your settings).

---

#### The Scientist agent — your data science expert

The Scientist is a specialised sub-agent built into this project. It has deep expertise in data science methodology — feature engineering, model evaluation, bias detection, statistical testing — and is the main tool for improving prediction accuracy.

**How to invoke it:** type `@"Scientist (agent)"` at the start of your message, then describe the task.

```
@"Scientist (agent)" [your task here]
```

That's it. The Scientist will spin up, read the relevant files, do the analysis, and report back. You don't need to tell it where the files are or how the code works — it figures that out itself.

---

#### What to ask the Scientist

##### After every backtest — improve the model

This is the most important use. After running `backtest.py`, hand the results to the Scientist:

```
@"Scientist (agent)" analyse the backtest and improve the prediction model
```

The Scientist will:
1. Read all the backtest CSVs and the log file
2. Calculate where the model is systematically wrong — which teams, which players, which disposal ranges
3. Identify the root cause (e.g. "the model is compressing high-disposal predictions due to the log transform")
4. Make targeted changes to `prediction.py` to fix what it found
5. Tell you exactly what it changed and what improvement to expect

You then re-run the backtest to verify the improvement.

##### Check if a run finished

If you started a backtest or prediction run and your laptop restarted or you closed the terminal:

```
@"Scientist (agent)" check the status of the last backtest run
```

It will look at the output files and logs and tell you exactly what completed, what's missing, and whether you need to re-run anything.

##### Investigate a specific problem

```
@"Scientist (agent)" the model keeps under-predicting Daicos — find out why and fix it
@"Scientist (agent)" round 1 accuracy is always much worse than other rounds — why?
@"Scientist (agent)" look at prediction.py and find any bugs that could cause wrong results
@"Scientist (agent)" which teams is the model most consistently wrong about?
```

##### Optimise slow code

```
@"Scientist (agent)" prediction.py is running slowly — find optimisations without changing the results
```

##### Understand the data

```
@"Scientist (agent)" what does the distribution of disposals look like across positions and teams?
@"Scientist (agent)" are there any data quality issues in the player CSVs I should know about?
```

---

#### The improvement loop — how to get better predictions week by week

```
1. Refresh data        →  type: refresh the data
2. Run backtest        →  python backtest.py --start-year 2026 --start-round 1
3. Invoke Scientist    →  @"Scientist (agent)" analyse the backtest and improve the model
4. Re-run backtest     →  verify MAE dropped
5. Push changes        →  type: push the changes to main
6. Predict next round  →  python prediction.py
```

Repeat steps 3–5 as many times as you like. Each iteration typically improves MAE by 0.2–0.5 disposals. The improvement compounds — the model that predicted 2026 R8 at MAE 4.1 started at MAE 4.9 after several rounds of Scientist-driven improvements.

---

#### Other useful Claude commands (no Scientist needed)

These work by just typing them at the Claude prompt:

| What you type | What Claude does |
|---------------|-----------------|
| `push the changes to main` | Commits all changes and pushes to GitHub |
| `update the readme` | Updates this file based on what changed |
| `what does prediction.py do?` | Explains the code in plain English |
| `the backtest crashed with [paste error] — fix it` | Diagnoses and fixes the error |
| `show me the top 10 most over-predicted players from the last backtest` | Reads the CSV and answers |
| `what is the current prediction accuracy?` | Summarises the latest backtest results |
| `explain why the model under-predicted [player name]` | Looks at their stats and explains |

---

#### Tips for better results

- **Paste the full error message.** If something crashes, copy everything from the terminal and paste it in. Claude fixes it faster with the full stack trace.
- **Be specific about what you care about.** "I care more about getting the top 20 players right than overall accuracy" helps the Scientist make better trade-offs.
- **Ask it to explain before changing.** Type "what would you change and why?" first — you can redirect it before it touches any code.
- **One session, one push.** At the end of any session where the model improved, type "push the changes to main" so you don't lose the work.
- **Tell the Scientist what surprised you.** "The model got Pendlebury badly wrong last round — why?" is better than "improve accuracy." Specific questions get specific answers.
