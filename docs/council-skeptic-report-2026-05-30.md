# Skeptic Adversarial Review — Gap 1 + Gap 2 Resolution

**Reviewer:** The Skeptic (adversarial pass, council meeting 2026-05-30)
**Reviewed at:** 2026-05-30 (UTC)
**Scope:** Gap 1 (pre-commit hook via `core.hooksPath`) and Gap 2 (`.claude/audit/` + `scripts/log-agent-turn.sh`) resolution plan, before Gaffer declares them closed in `docs/ai-architecture.md`.

---

## Verdicts

| Gap | Verdict | One-line reason |
|-----|---------|-----------------|
| **Gap 1** — pre-commit hook relocation | **BLOCK** | The plan is sound but the wiring is not actually live: `core.hooksPath` is still `.git/hooks`, `.githooks/pre-commit` is untracked and non-executable, and `CONTRIBUTING.md` has no setup instruction. The gate is not enforced on this machine right now. |
| **Gap 2** — agent-turn audit log | **PASS_WITH_CONCERNS** | The script works and is committable, but it captures far less than the architecture doc's own Gap 2 recommendation promised. Calling it an "audit trail" overstates what it observes. |

The aggregate recommendation to Gaffer: **do not write "closed" for either gap.** Honest status is **"MVP-live, not production-grade"** for Gap 2 and **"built, not yet wired"** for Gap 1.

---

## Probe 1 — Tripwire observability (Gap 1 enforceability)

**The claim under test:** moving the hook to `.githooks/` + `core.hooksPath` makes the council-stamp provenance gate enforceable, with `CONTRIBUTING.md` wiring it via a setup instruction.

**What I verified on disk (2026-05-30):**

1. **`git config core.hooksPath` returns `/home/abhi/git/SuperCoach-VIA/.git/hooks`** — NOT `.githooks`. The relocation has not been activated. The new hook at `.githooks/pre-commit` is inert on this machine; the live hook is still the local-only one at `.git/hooks/pre-commit`.
2. **`.githooks/pre-commit` is mode `-rw-rw-r--` (not executable).** Even once `core.hooksPath` points at it, Git will not run a non-executable hook. This is a silent failure: no error, the gate simply does not fire.
3. **`.githooks/pre-commit` is untracked** (`git ls-files .githooks/` is empty). It is not yet version-controlled, so the whole premise of Gap 1 — "the wiring travels with the repo" — is not yet true. If committed without `chmod +x` (and without `git update-index --chmod=+x`), it lands in the repo without its executable bit and is dead on every clone.
4. **`CONTRIBUTING.md` contains no `core.hooksPath` / `githooks` instruction** (grep: no match). The doc that is supposed to carry the one-time setup step does not carry it. A fresh contributor has no documented path to enablement.

**The failure mode if an operator forgets `git config core.hooksPath .githooks`:** the gate fails *open*, silently. There is no error, no warning — commits of unstamped council docs succeed exactly as they did before the gap was "closed." This is the worst class of safety control: one that is believed to be on and is actually off. It is strictly worse than the prior `.git/hooks` symlink approach in one respect: the symlink approach at least failed *visibly* (the operator knew they had to wire it). A committed `.githooks/` dir creates the *appearance* of an enforced, repo-travelling gate while the enforcement still depends on a per-clone manual `git config` that nothing checks.

**Is there a better mechanism?** Three escalating options, none of which the current plan implements:

- **Minimum bar:** a bootstrap step (`make setup`, or a line in `refresh_and_rank.sh` / `scripts/`) that runs `git config core.hooksPath .githooks` so wiring is one command, plus committing `.githooks/pre-commit` *with* its executable bit (`git update-index --chmod=+x`). This is what "closed" would minimally require.
- **Better:** move the real enforcement to **CI** (a server-side check on push / PR that runs `scripts/check-council-stamp.sh --dry-run` and fails the build). A local hook is advisory by nature — any operator can `git commit --no-verify`. CI is the only place the gate cannot be bypassed by the committer. The architecture doc already references a `--dry-run` report-only mode "for CI" (§13.4) but no CI workflow exists.
- **Best (already named as future work):** the hook re-runs DataSentinel against source CSVs rather than trusting the recorded `PASS` stamp, so a hand-forged stamp cannot pass. Out of scope for this meeting, but it is the difference between "the gate checks a self-attestation" and "the gate checks the truth."

**Probe 1 verdict: BLOCK on Gap 1.** Not because the design is wrong — it is the right design — but because the artifacts on disk do not match the claim. "Closed" is false today. The honest word is **"built, not yet wired."**

---

## Probe 2 — Caveat-hierarchy fidelity (what the audit log does NOT capture)

**The claim under test:** Gap 2's `scripts/log-agent-turn.sh` + `.claude/audit/*.jsonl` constitutes the "LLM-turn audit trail" that the architecture doc (§13.4 Gap 2, §13 Threat-2/7/9 fixes, §8 Accountability) says is missing.

**The script as built emits exactly this per line:**

```json
{"ts":"...","agent":"...","action":"...","files":[...],"verdict":"PASS|FAIL|BLOCK|DONE|NOTE"}
```

**What the architecture doc's own Gap 2 recommendation promised (lines 391, 558, 674):**

> persist per-session: model version, prompt template hash, tool calls + arguments + exit status, latency, token counts
> `{session_id, agent, model_version, prompt_hash, tool_calls, retrieved_files, output_hash, latency_ms, token_count, timestamp}`

**The gap between what an operator might believe and what they get:**

| Operator might believe the log captures… | Actually captured? |
|--------------------------------------------|--------------------|
| Which model/version produced a turn | **No** — no `model_version` field |
| The prompt that produced the output | **No** — no `prompt_hash` |
| Which tools were called, with what arguments and exit status | **No** — only a free-text `action` string the caller types by hand |
| Which data files were *read* during reasoning | **No** — `files` is a caller-supplied list, not an observed read-set; trivially incomplete or wrong |
| The output, in a tamper-evident form | **No** — no `output_hash` |
| Cost / latency | **No** — no `latency_ms`, no `token_count` |
| That a turn happened *at all* | **Only if the agent voluntarily called the script** |

**The load-bearing weakness:** this is **self-reported, opt-in logging**, not observability. Nothing emits a record automatically — an agent (or operator) must remember to invoke `log-agent-turn.sh` and must type honest values into `--action` and `--files`. The exact forensic questions Gap 2 was meant to answer — "why did the agent write X," "which files did it read," "was DataSentinel actually run" (§13 Threats 2, 7, 9) — are **unanswerable** from a log the agent fills in about itself. A compromised or hallucinating agent writes a clean log line; a skipped DataSentinel turn simply produces no line at all and looks identical to "the script wasn't called." The log cannot distinguish "didn't happen" from "happened but wasn't logged."

This is the real caveat-hierarchy violation: the architecture doc lists Gap 2 under **Australia AI Ethics Principle 8 (Accountability)** and as the fix for three security threats. A free-text, opt-in JSONL emitter does not satisfy any of those claims. It is a useful **operational note-taking convention** — genuinely worth committing — but it is not the audit trail the doc describes. Real observability requires the harness to emit the record (OpenTelemetry tracing / session-transcript capture, as the doc's own recommendation says), not the agent.

**Two smaller technical notes (non-blocking):**
- The script is mode `-rw-` (not executable) on disk. It will need `chmod +x` before it can be invoked as `scripts/log-agent-turn.sh`; works via `bash scripts/log-agent-turn.sh` regardless.
- `set -euo pipefail` is correct; the `json_escape` handles quotes/backslashes/control chars properly. No correctness objection to the JSON emission itself.

**Probe 2 verdict: PASS_WITH_CONCERNS on Gap 2.** Commit it — it has value. But it must NOT be characterised as closing the §8 Accountability gap or the Threat-2/7/9 audit gap. It closes the *note-taking* sliver, not the *observability* gap.

---

## Probe 3 — Lens-tension smoothing (is "closed" premature?)

**The structural tension:** Gaffer is *boss of process, not of truth* (§ line 355). Gaffer commissions and approves cadence but is "structurally forbidden from overriding, softening, or re-running-until-green" a gate verdict. Declaring Gap 1 and Gap 2 "closed" is a process-authority act. But the architect's position — that these gates **must be live before Gaffer's authority is truly enforced** — is a truth-claim about system state. The Skeptic's job is to flag when the process boss smooths over the truth boss.

**This is exactly that case.** Gaffer's authority to enforce the council pipeline is itself *predicated on* the pre-commit gate being live (that is what converts the chain "from the order we usually run things in to the order the harness enforces," line 357). If Gaffer declares the gate "closed" while the gate is in fact unwired (Probe 1: `core.hooksPath` unset, hook untracked, non-executable, undocumented), then Gaffer is asserting an enforcement authority that does not yet exist. That is the precise failure the architect warned about: **declaring the enabling control closed is what would let an unenforced control masquerade as enforced.**

There is no genuine lens *disagreement* being smoothed here — the architect and the Skeptic agree the gates are not yet live. The smoothing risk is in the **status word**. "Closed" is a convergence claim ("done, enforced, move on"). The evidence does not support it. Using "closed" would be a hollow-convergence framing: the word implies the gate blocks bad commits when on this machine it does not.

**The honest status word, per gap:**
- **Gap 1: "built, not yet wired."** The script and `.githooks/pre-commit` exist and are correct in design, but enforcement is not active (`core.hooksPath` unset, hook non-executable/untracked, no `CONTRIBUTING.md` step). It becomes "wired" only when those four facts flip — and "enforced" only when it cannot be bypassed (CI), which is further out.
- **Gap 2: "MVP-live, not production-grade."** The logging primitive exists and runs, but it captures a small, self-reported subset of what Gap 2 specified, and nothing emits records automatically. It is live as a convention; it is not the accountability/observability control the doc claims.

**Probe 3 verdict: declaring either gap "closed" is premature and would itself be a smoothed tension.** Gaffer may legitimately record *progress*; Gaffer may not record *enforcement* that is not on.

---

## Coach-anonymity audit

Not applicable to this doc (no coach names; this is a harness/architecture review). No matches.

## Sentinel smoke test

Not applicable — this review contains no `**[data]**`-tagged player statistics. The factual claims here are file-state assertions, each verified directly against the working tree on 2026-05-30 (paths and `git config`/`git ls-files`/`stat`/`grep` outputs cited inline above).

---

## Recommended wording for `docs/ai-architecture.md`

Gaffer should NOT write "closed" for either gap. Specific replacement wording:

**For Gap 1** (replacing any "closed"/"resolved" status on the §13.4 "No pre-commit gate" row):

> **Status: built, not yet wired.** The committed gate (`.githooks/pre-commit` + `scripts/check-council-stamp.sh`) exists and is correct, but enforcement is not active until each clone runs `git config core.hooksPath .githooks` (now documented in `CONTRIBUTING.md`) AND the hook carries its executable bit in the index. A local hook remains bypassable with `--no-verify`; true enforcement requires a server-side CI check, which is not yet built. Until then this is a *self-service, opt-in* gate, not a *guaranteed* one.

**For Gap 2** (replacing any "closed"/"resolved" status on the §13.4 "No LLM-turn audit trail" row):

> **Status: MVP-live, not production-grade.** `scripts/log-agent-turn.sh` provides an opt-in, self-reported turn-logging convention (`ts, agent, action, files, verdict`). It does NOT capture model version, prompt hash, tool calls/arguments, observed file reads, output hash, latency, or token counts, and nothing emits records automatically. It is useful operational note-taking; it does **not** yet satisfy the Australia AI Ethics Principle 8 (Accountability) gap or the Threat-2/7/9 forensic-audit needs this gap was filed to address. Those require harness-level emission (OpenTelemetry / session-transcript capture), still future work.

**Blocking pre-commit checklist before Gap 1 can be called even "wired":**
1. `git add .githooks/pre-commit && git update-index --chmod=+x .githooks/pre-commit` (commit with executable bit).
2. `chmod +x scripts/log-agent-turn.sh` before committing it.
3. Add the `git config core.hooksPath .githooks` setup step to `CONTRIBUTING.md` (developer section).
4. Run `git config core.hooksPath .githooks` on this machine and verify the hook fires on a test commit of an unstamped `docs/news/*.md`.

I am **uncertain** on exactly one point and flag it rather than assert it: whether the council intends Gap 1 to mean "wired on the operator's machine" or "enforced everywhere via CI." If the former, items 1–4 above clear it to "wired." If the latter, it cannot be "closed" until CI exists. Either way, "closed" today is unsupported. The operator is the final authority on which bar applies.

---

## Out of scope for this review
- Full `[data]` tag verification (none present here).
- The correctness of the *future* DataSentinel-re-run-in-hook design (not yet built).
- Editorial tone of `docs/ai-architecture.md`.
- Whether CI tooling choice (GitHub Actions vs other) is right — that is the operator's call.
