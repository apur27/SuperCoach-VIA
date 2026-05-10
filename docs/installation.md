# Installation & first-time setup

> [← Back to main README](../README.md)

<!-- This file is part of the SuperCoach-VIA documentation. See README.md for the project overview. -->

**Pick the section that matches you. Most footy fans only need section 1.**

| You are... | Go to | What you'll get |
|---|---|---|
| **A footy fan** who just wants this week's predictions, stat leaders, or the all-time top 100 | [For Fans](#for-fans-zero-setup-required) | Browser-only - no install, no terminal |
| **A SuperCoach player** who wants to download the prediction CSV and slice it in your own spreadsheet | [For Power Users](#for-power-users-download-the-data-no-coding) | A few clicks, no Python, no Git |
| **A developer or data scientist** who wants to run the pipeline, retrain the model, or contribute | [For Contributors](#for-contributors-full-local-setup-on-ubuntu) | Full local setup from zero |

---

## For Fans - zero setup required

Everything you need is already viewable on GitHub in your browser. No install, no terminal, no Python.

- **This week's predicted disposal leaders** → [docs/afl-predictions-2026.md](afl-predictions-2026.md)
- **2026 season hub** (form, ladder, top scorers) → [docs/afl-season-2026.md](afl-season-2026.md)
- **All-time top 100** → [docs/hall-of-fame-top100.md](hall-of-fame-top100.md)
- **The cheat sheets and weekly fan pack** → [docs/start-here-no-code.md](start-here-no-code.md)
- **What the numbers mean** → [docs/glossary.md](glossary.md)

That's it. You're done. Bookmark this repo and check back weekly.

---

## For Power Users - download the data, no coding

You want the raw prediction CSV in Google Sheets or Excel so you can sort, filter, and add your own columns. You don't want to install anything.

1. Go to the latest prediction CSV in this repo at **`data/prediction/`** (file names look like `next_round_<N>_prediction_<timestamp>.csv` - pick the most recent for the upcoming round).
2. Click **Raw** on GitHub, then save the page as a `.csv` file (right-click → Save Page As).
3. Open it in Google Sheets, Excel, or Numbers. The columns are: `player`, `team`, `predicted_disposals`.
4. Set up a watchlist or comparison view using the [Google Sheets template](../templates/google-sheets-template.md).

That's the whole flow. You don't need Python, Git, or anything installed.

---

## For Contributors - full local setup on Ubuntu

**Who is this section for?** Brand-new users on Ubuntu (or anyone who wants the full step-by-step). Walks you from zero - no Git, no GitHub account, no Python - through to having the SuperCoach-VIA project cloned and ready to run.

> **If you have already cloned the repo and just want to run a prediction**, jump to [Running predictions & backtests](usage.md).

## Step 0 - Set up Git and GitHub (skip this if you've done it before)

Brand new to all this? No worries - this is a one-time setup. Once it's done you'll never have to do it again on this laptop. Work through Parts A to F in order and you'll be ready to grab the project.

> **Tip:** if your terminal ever looks frozen after you paste a command, give it a tap of **Enter** - sometimes the cursor just hasn't caught up.

### Part A - Install Git on Ubuntu

**Git** is a tool that downloads code projects from the internet and keeps track of changes you make to them. It's the "engine" that powers the `git clone` command you'll use in a minute.

Open a terminal (`Ctrl+Alt+T`) and paste these two commands, one at a time, pressing **Enter** after each:

```bash
sudo apt update
sudo apt install git -y
```

`sudo` means "run as administrator" - Ubuntu will ask for your login password the first time. Type it in (you won't see the characters as you type, that's normal) and hit Enter.

Check it worked:

```bash
git --version
```

You should see something like `git version 2.43.0`. If you see "command not found", run the install command again.

### Part B - Create a free GitHub account

**GitHub** is a website that hosts code projects - think Google Drive, but for code. This SuperCoach-VIA project lives on GitHub, which is why you need a (free) account to grab a copy.

1. Open a browser and go to [github.com](https://github.com).
2. Click **Sign up** in the top right.
3. Pick a **username**, enter your **email**, and choose a **password**.
4. GitHub will email you a verification code - paste it in to finish.

> **Tip:** a free account is plenty. You only need to log in if you want to save your own changes back to GitHub later. Just downloading (cloning) this project is free and anonymous.

### Part C - Tell Git who you are (one-time setup)

Git stamps your name and email onto any change you make - it's how the project history knows who did what, like signing your work. You only need to do this once per laptop.

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

Use the **same email** as your GitHub account so everything lines up.

### Part D - Connect to GitHub: choose HTTPS or SSH

To download the project, your laptop needs a way to talk to GitHub. There are two options:

> **Not sure which to choose?** Use **HTTPS** - it's simpler and works everywhere. **SSH** is faster and password-free once configured but takes a few extra steps. You can always switch later.

#### Option 1: HTTPS (simpler)

When you run `git clone https://...` you'll be asked for your GitHub username and password.

**Important:** GitHub no longer accepts your normal account password here. You need a **Personal Access Token (PAT)** instead - basically a long random password that only Git uses.

Step-by-step to create a PAT:

1. Log in to [github.com](https://github.com).
2. Click your **profile picture** (top right) → **Settings**.
3. Scroll down the left sidebar to **Developer settings** (right at the bottom) → **Personal access tokens** → **Tokens (classic)**.
4. Click **Generate new token (classic)**.
5. Give it a name like `SuperCoach laptop`, set **Expiration** to 90 days, and tick the **`repo`** checkbox.
6. Click **Generate token**. ✓ **Copy the token straight away** - GitHub only shows it once, and once you close that page it's gone forever.
7. When Git asks for a password later, paste this token (not your GitHub password).

> **Tip:** save yourself the hassle of pasting the token every time by telling Git to remember it:
> ```bash
> git config --global credential.helper store
> ```
> The next time you clone or pull, type your username + token once and Git will save it for you.

#### Option 2: SSH (recommended for regular use)

**SSH** is a different way of proving you're you. Instead of a password, your laptop holds a secret key and GitHub holds a matching lock. Once it's set up you never type a password again - Git and GitHub just recognise each other automatically.

Step-by-step:

1. Generate a key pair. Press **Enter** to accept the default file location, and either press **Enter** again for no passphrase or type one in for extra security:
   ```bash
   ssh-keygen -t ed25519 -C "your@email.com"
   ```
2. Print your **public** key (the safe-to-share half) to the screen:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
3. Select and copy the entire output - it's the whole line that starts with `ssh-ed25519 ...` and ends with your email.
4. On [github.com](https://github.com): **profile picture** → **Settings** → **SSH and GPG keys** (left sidebar) → **New SSH key**.
5. Paste the key into the **Key** box, give it a friendly title like `My Ubuntu laptop`, then click **Add SSH key**.
6. Test it works:
   ```bash
   ssh -T git@github.com
   ```
   You should see: `Hi username! You've successfully authenticated...`. The first time, it'll ask if you trust GitHub's host - type `yes` and hit Enter.
7. From now on, when you clone, use the **SSH URL** (starts with `git@github.com:`) instead of the HTTPS one - see Part E below.

### Part E - Clone this project

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

`cd` stands for **change directory** - it's how you move between folders in the terminal, the same as double-clicking a folder in a file browser.

### Part F - Keeping your copy up to date

Each week, after the latest round's data is published, you'll want to pull down the new match results, player stats, and rankings. From inside the `SuperCoach-VIA` folder, run:

```bash
git pull
```

In plain English: this downloads any updates that have been made to the project since you last cloned - like refreshing a webpage. Your local files get the new stuff added, but anything you've changed yourself stays put.

> **Tip:** if `git pull` ever complains about local changes you didn't mean to make, the safest first move is `git status` to see what it's worried about. Ask a mate (or Claude) before running anything that says "discard" or "reset".

You're done with setup. The numbered steps below pick up from here.

## 1. Download the repo

This downloads the whole project to a new folder on your laptop called `SuperCoach-VIA` and then moves your terminal into it.

```bash
git clone https://github.com/apur27/SuperCoach-VIA.git
cd SuperCoach-VIA
```

## 2. Install dependencies

"Dependencies" are the extra software libraries the scripts in this project use to do their work - things like `pandas` for spreadsheets and `LightGBM` for the prediction model. The single `pip install` command below downloads and sets them all up automatically.

```bash
pip install -r requirements.txt
```

That's it. You're ready to run predictions. To pull the latest match and player data first, see [Refresh data and rankings](usage.md#refresh-data-and-rankings).


---

## Related

- [Running predictions & backtests](usage.md)
- [Troubleshooting](troubleshooting.md)
- [Claude Code setup on Ubuntu](claude-code-setup.md)
- [Technical reference (GPU setup, etc.)](technical-reference.md)
