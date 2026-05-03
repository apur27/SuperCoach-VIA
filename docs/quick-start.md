# Quick Start Guide

> [← Back to main README](../README.md)

<!-- This file is part of the SuperCoach-VIA documentation. See README.md for the live data sections. -->

## Quick start

**Who is this section for?** Anyone who just wants to get the project running and start using it — predict next round, run a backtest, or refresh the data — without first wading through methodology or AFL analysis.

> **Before running any command:** make sure you are inside the `SuperCoach-VIA` folder (`cd SuperCoach-VIA`) and your Python environment is active (`source .venv/bin/activate`). After that, just use `python script.py` — no long paths needed.

New here? These three commands are all you need. Download a copy of the project (a "repo" is just a project folder hosted on GitHub), run `prediction.py` for next week's disposal projections, and `./refresh_and_rank.sh` to refresh data and rebuild the all-time top 100.

#### New to the terminal? Start here

If you've never used a **terminal** before, don't stress — it's just a text window where you type instructions to your computer instead of clicking buttons. Everything in the Quick Start below is typed into that one window.

**To open a terminal on Ubuntu**, press `Ctrl+Alt+T` on your keyboard, or click the apps button (the grid of dots) in the dock and search for *Terminal*. A black or dark-grey window will pop up with a blinking cursor — that's it.

From there, every line of code shown in a grey box below is something you copy, paste into the terminal, and then press **Enter** to run. That's the whole interaction. The computer reads each line, does the work, and prints the result back to the same window.

If anything you see below looks intimidating — long URLs, words like `pip` or `npm`, lots of slashes — that's normal. You only need to copy-paste a handful of commands and the hard parts (downloading code, installing libraries, picking the right version) are handled automatically. The full step-by-step setup, with explanations of what each step does, lives in the [Getting started](#getting-started) and [Setting up Claude Code on Ubuntu](ai-agents.md#setting-up-claude-code-on-ubuntu) sections — start there if you want a slower walk-through.

### Getting started

#### Step 0 — Set up Git and GitHub (skip this if you've done it before)

Brand new to all this? No worries — this is a one-time setup. Once it's done you'll never have to do it again on this laptop. Work through Parts A to F in order and you'll be ready to grab the project.

> **Tip:** if your terminal ever looks frozen after you paste a command, give it a tap of **Enter** — sometimes the cursor just hasn't caught up.

##### Part A — Install Git on Ubuntu

**Git** is a tool that downloads code projects from the internet and keeps track of changes you make to them. It's the "engine" that powers the `git clone` command you'll use in a minute.

Open a terminal (`Ctrl+Alt+T`) and paste these two commands, one at a time, pressing **Enter** after each:

```bash
sudo apt update
sudo apt install git -y
```

`sudo` means "run as administrator" — Ubuntu will ask for your login password the first time. Type it in (you won't see the characters as you type, that's normal) and hit Enter.

Check it worked:

```bash
git --version
```

You should see something like `git version 2.43.0`. If you see "command not found", run the install command again.

##### Part B — Create a free GitHub account

**GitHub** is a website that hosts code projects — think Google Drive, but for code. This SuperCoach-VIA project lives on GitHub, which is why you need a (free) account to grab a copy.

1. Open a browser and go to [github.com](https://github.com).
2. Click **Sign up** in the top right.
3. Pick a **username**, enter your **email**, and choose a **password**.
4. GitHub will email you a verification code — paste it in to finish.

> **Tip:** a free account is plenty. You only need to log in if you want to save your own changes back to GitHub later. Just downloading (cloning) this project is free and anonymous.

##### Part C — Tell Git who you are (one-time setup)

Git stamps your name and email onto any change you make — it's how the project history knows who did what, like signing your work. You only need to do this once per laptop.

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

Use the **same email** as your GitHub account so everything lines up.

##### Part D — Connect to GitHub: choose HTTPS or SSH

To download the project, your laptop needs a way to talk to GitHub. There are two options:

> **Not sure which to choose?** Use **HTTPS** — it's simpler and works everywhere. **SSH** is faster and password-free once configured but takes a few extra steps. You can always switch later.

###### Option 1: HTTPS (simpler)

When you run `git clone https://...` you'll be asked for your GitHub username and password.

**Important:** GitHub no longer accepts your normal account password here. You need a **Personal Access Token (PAT)** instead — basically a long random password that only Git uses.

Step-by-step to create a PAT:

1. Log in to [github.com](https://github.com).
2. Click your **profile picture** (top right) → **Settings**.
3. Scroll down the left sidebar to **Developer settings** (right at the bottom) → **Personal access tokens** → **Tokens (classic)**.
4. Click **Generate new token (classic)**.
5. Give it a name like `SuperCoach laptop`, set **Expiration** to 90 days, and tick the **`repo`** checkbox.
6. Click **Generate token**. ✓ **Copy the token straight away** — GitHub only shows it once, and once you close that page it's gone forever.
7. When Git asks for a password later, paste this token (not your GitHub password).

> **Tip:** save yourself the hassle of pasting the token every time by telling Git to remember it:
> ```bash
> git config --global credential.helper store
> ```
> The next time you clone or pull, type your username + token once and Git will save it for you.

###### Option 2: SSH (recommended for regular use)

**SSH** is a different way of proving you're you. Instead of a password, your laptop holds a secret key and GitHub holds a matching lock. Once it's set up you never type a password again — Git and GitHub just recognise each other automatically.

Step-by-step:

1. Generate a key pair. Press **Enter** to accept the default file location, and either press **Enter** again for no passphrase or type one in for extra security:
   ```bash
   ssh-keygen -t ed25519 -C "your@email.com"
   ```
2. Print your **public** key (the safe-to-share half) to the screen:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
3. Select and copy the entire output — it's the whole line that starts with `ssh-ed25519 ...` and ends with your email.
4. On [github.com](https://github.com): **profile picture** → **Settings** → **SSH and GPG keys** (left sidebar) → **New SSH key**.
5. Paste the key into the **Key** box, give it a friendly title like `My Ubuntu laptop`, then click **Add SSH key**.
6. Test it works:
   ```bash
   ssh -T git@github.com
   ```
   You should see: `Hi username! You've successfully authenticated...`. The first time, it'll ask if you trust GitHub's host — type `yes` and hit Enter.
7. From now on, when you clone, use the **SSH URL** (starts with `git@github.com:`) instead of the HTTPS one — see Part E below.

##### Part E — Clone this project

Right, you're set. **Cloning** just means "download a copy of the project to your laptop". Pick the matching command for whichever option you set up above:

```bash
# HTTPS version (use this if you set up a Personal Access Token)
git clone https://github.com/apur27/SuperCoach-VIA.git

# SSH version (use this if you set up SSH keys)
git clone git@github.com:apur27/SuperCoach-VIA.git
```

Then step into the new folder:

```bash
cd SuperCoach-VIA
```

`cd` stands for **change directory** — it's how you move between folders in the terminal, the same as double-clicking a folder in a file browser.

##### Part F — Keeping your copy up to date

Each week, after the latest round's data is published, you'll want to pull down the new match results, player stats, and rankings. From inside the `SuperCoach-VIA` folder, run:

```bash
git pull
```

In plain English: this downloads any updates that have been made to the project since you last cloned — like refreshing a webpage. Your local files get the new stuff added, but anything you've changed yourself stays put.

> **Tip:** if `git pull` ever complains about local changes you didn't mean to make, the safest first move is `git status` to see what it's worried about. Ask a mate (or Claude) before running anything that says "discard" or "reset".

You're done with setup. The numbered steps below pick up from here.

#### 1. Download the repo

This downloads the whole project to a new folder on your laptop called `SuperCoach-VIA` and then moves your terminal into it.

```bash
git clone https://github.com/apur27/SuperCoach-VIA.git
cd SuperCoach-VIA
```

#### 2. Install dependencies

"Dependencies" are the extra software libraries the scripts in this project use to do their work — things like `pandas` for spreadsheets and `LightGBM` for the prediction model. The single `pip install` command below downloads and sets them all up automatically.

```bash
pip install -r requirements.txt
```

That's it. You're ready to run predictions. To pull the latest match and player data first, see [Refresh data and rankings](#refresh-data-and-rankings) below.

### Predict next week's disposals

Run this command and it will automatically figure out the current year and the next round that hasn't been played yet, then generate predictions for every player:

> **Needs GPU.** See [GPU setup](technical-reference.md#setting-up-gpu-acceleration-optional) if you haven't configured CUDA yet — CPU-only runs are 10–30× slower.

```bash
python prediction.py
```

The result is saved to `data/prediction/next_round_N_prediction_<timestamp>.csv`. Open it in Excel or any spreadsheet app — it has three columns:

| Column | What it means |
|--------|---------------|
| `player` | Player name |
| `team` | Their club |
| `predicted_disposals` | How many disposals the model thinks they'll get |

**Want more detail while it runs?**
```bash
python prediction.py --debug
```

**Want to predict for a specific year?**
```bash
python prediction.py --year 2026
```

**No GPU? Use the CPU version (slower but works anywhere):**
```bash
python prediction_cpu.py
```

### What you'll actually get

Run the prediction script and you'll get a file like this — open it in Excel or Google Sheets:

| player | team | predicted_disposals |
|---|---|---|
| Nick Daicos | Collingwood | 34 |
| Clayton Oliver | Greater Western Sydney | 31 |
| Lachie Neale | Brisbane Lions | 30 |
| Zak Butters | Port Adelaide | 28 |
| Patrick Cripps | Carlton | 27 |
| … | … | … |

**How accurate is it?** Based on the 2026 season backtest (rounds 1–8):

- Average error: roughly **4–5 disposals per player per game**
- About **65–70% of predictions land within ±5 disposals** of the actual result
- About **90%+ land within ±10 disposals**

That's not perfect — football is unpredictable — but it's meaningfully better than guessing, and it gives you a consistent, data-driven starting point for your SuperCoach trades.

The `data/prediction/backtest/` folder shows the model's performance on every past round in full detail — nothing is hidden.

### Run a backtest

A backtest replays past rounds as if you were predicting them at the time, so you can see how accurate the model has been on data it's never seen.

> **Needs GPU.** See [GPU setup](technical-reference.md#setting-up-gpu-acceleration-optional) if you haven't configured CUDA yet — CPU-only runs are 10–30× slower.

```bash
python backtest.py --start-year 2026 --start-round 1
```

> **Heads up:** each round takes about 30 minutes on a GPU, so an 8-round backtest runs for 4–5 hours. Best to kick it off and leave it running.

Output lands in `data/prediction/backtest/`. For the full command reference, file layout, and how to read the log, see [Backtest framework](prediction-model.md#backtest-framework) under the prediction-model section.

### Refresh data and rankings

To pull the latest match and player data and recalculate the all-time top 100 ranking + 2026 team analysis in one command:

> **Needs GPU.** See [GPU setup](technical-reference.md#setting-up-gpu-acceleration-optional) if you haven't configured CUDA yet — CPU-only runs are 10–30× slower.

```bash
./refresh_and_rank.sh
```

#### What does this script actually do?

Think of `refresh_and_rank.sh` as the **"update everything" button** for the whole project. One command, four big jobs, and when it finishes the README and every chart you see in this file have been re-built from scratch with the latest numbers.

1. **It downloads the latest match and player data from the internet** — match results, every player's stats line, fresh from AFL Tables.
2. **It recalculates the all-time top 100 player rankings** — re-scores every player from 1897 onwards using the current ranking formula and writes the updated `all_time_top_100.csv`.
3. **It rebuilds all the 2026 team analysis, finals pathway, Brownlow predictor and stat leaders** — every paragraph in the AFL Insights section is regenerated from the freshest data.
4. **It regenerates all the charts and updates this README** — the auto-marker sections in the file (team analysis, finals pathway, Brownlow predictor, stat leaders, 5-year profiles) are overwritten with the new content. You don't edit those by hand; they belong to this script.

The full run takes roughly **10 to 15 minutes on a GPU**, and longer without one. It's the right thing to run after each round of footy is played. You can also have it run automatically every week — see the troubleshooting section under [Setting up Claude Code on Ubuntu](ai-agents.md#setting-up-claude-code-on-ubuntu) for one way to do that on a schedule.

What it actually executes, end-to-end:
1. `refresh_data.py` — scrape the latest match and player results from AFL Tables
2. `top_players_comprehensive.py` — recompute and write `all_time_top_100.csv`
3. `update_team_analysis.py` — regenerate the 2026 team analysis section + 5-year team-style profiles + the embedded charts in this README

## Troubleshooting and common questions

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
