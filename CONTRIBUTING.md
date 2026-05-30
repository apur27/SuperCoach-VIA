# Contributing to SuperCoach VIA

Thanks for taking the time. This repo welcomes contributions from footy fans, SuperCoach players, data scientists, and developers - in roughly that order of "you don't need to know what a `LightGBM` is to be useful here."

This file collects the lightweight conventions and the maintenance odds-and-ends. For technical setup, see [docs/installation.md](docs/installation.md). For the bigger-picture roadmap, see [docs/roadmap.md](docs/roadmap.md).

---

## How fans can contribute

You do not need to write code or open a pull request. The most useful contributions from fans are:

- **Prediction feedback** - the model got a player obviously wrong and your eye knows why. [Email careerabhi@gmail.com](mailto:careerabhi@gmail.com) with the player name and what the model missed. This is one of the highest-value things a fan can do - human eye sees role changes and tag jobs faster than the model.
- **Bug reports** - a doc looks wrong, a number is impossible, a chart is broken. [Email careerabhi@gmail.com](mailto:careerabhi@gmail.com) with what you found.
- **Feature requests** - a stat you wish was tracked, a doc you wish existed. [Email careerabhi@gmail.com](mailto:careerabhi@gmail.com) with the idea.

No account required - just a plain-English email is fine.

---

## One-time setup after clone

Run this once after cloning the repo to activate the committed pre-commit hook:

```bash
git config core.hooksPath .githooks
```

This tells git to use `.githooks/` (version-controlled) instead of `.git/hooks/` (local-only). Without it, the council-stamp gate will not run on your machine.

**What the pre-commit hook checks:**
The hook runs `scripts/check-council-stamp.sh` against any staged Markdown files. It blocks the commit if a council-authored doc (news articles under `docs/news/`, Hall of Fame stat pages `docs/hall-of-fame-stat-*.md`) is missing a valid `<!-- council-pipeline: ... -->` provenance stamp, or if the recorded DataSentinel or Skeptic verdict is not PASS.

Everything else (README, CSV, Python files, briefs, other docs) is skipped.

**How to add a council stamp to a new article:**
Every file under `docs/news/` and every `docs/hall-of-fame-stat-*.md` must include a block like:

```
<!-- council-pipeline:
  FootyStrategy: PASS
  BriefBuilder: PASS
  DataSentinel: PASS
  Skeptic: PASS
  Scientist: PASS
  Gaffer: PASS
-->
```

All six agents must have run and both gating tiers (DataSentinel and Skeptic) must record PASS before the commit will be accepted. The check is deterministic - it greps the file, it does not invoke any LLM.

---

## How developers can contribute

For code contributions, the friendly defaults are:

1. Fork → branch → PR. Small focused PRs are easier to review than big ones.
2. Read [docs/installation.md](docs/installation.md) (For Contributors section) for environment setup. Run `git config core.hooksPath .githooks` as part of setup (see above).
3. The [Scientist agent system prompt](.claude/agents/Scientist.md) describes the methodology rules: inspect-before-transform, baselines first, no leakage, reproducibility. Code that touches the model or backtest is held to those rules.
4. Don't edit between `<!-- ...-START -->` and `<!-- ...-END -->` markers in the auto-updated docs - those sections are rewritten by the refresh pipeline.
5. The full project history lives on `main`. There is no separate dev branch.

---

## Recommended GitHub repo topics

To help fans and data scientists discover this project, the repo should be tagged with the following topics on GitHub. Anyone with admin access can set them via:

**GitHub → repo page → "About" panel (right sidebar) → cog icon → Topics**

Recommended topics:

```
afl
supercoach
fantasy-sports
australian-football
sports-analytics
machine-learning
data-science
football-data
python
```

These are the hashtag-style labels GitHub uses for search and the "explore topics" feed. Adding them costs nothing and meaningfully helps discoverability for both audiences (footy fans searching `afl` / `supercoach`, and data scientists searching `sports-analytics` / `machine-learning`).

If a topic seems redundant or wrong, open a PR that edits this list rather than just removing it from GitHub - keep the source of truth here so it survives admin-handover.

---

## Conventions

- **No em-dashes (`—`) in committed text.** Use a regular hyphen (`-`). The repo went through a consistency pass and the convention is "hyphens only" for searchability.
- **AFL stats must be verified from repo data**, not from training-data memory. See the data-verification rule in [CLAUDE.md](CLAUDE.md). If a number is genuinely unverifiable from the data, mark it `**[historical record - unverified in data]**`.
- **Plain English over jargon.** This project is read by footy fans, not just data scientists. The [glossary](docs/glossary.md) should grow with us, not the other way around.
- **Honesty over gloss.** A null result reported clearly is better than a positive result built on a leaky pipeline. The model is intentionally honest about its limitations - see [how-to-use-this-for-supercoach.md](docs/how-to-use-this-for-supercoach.md).

---

## Collaboration contact

Found something that warrants a longer conversation than an issue? Reach out via the email in the README's "Why this repo exists" section, or open a Discussion.

---

## License

By contributing, you agree your contributions are licensed under the same MIT License as the rest of the repo - see [LICENSE](LICENSE).
