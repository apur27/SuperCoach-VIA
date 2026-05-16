---
name: Brainstorm framing conventions (post-match and design)
description: User runs two distinct brainstorm shapes - post-match themed analyses and design brainstorms - which need different output formats
type: user
---

**Convention A - post-match themed brainstorms:** The user (Richmond-leaning AFL analyst working in the SuperCoach-VIA repo) runs post-match analyses as a structured brainstorm across N tactical themes (typically 6), each requiring: strategic read, what-it-means-for-each-team forward, and one bullet for "next time." Output is consumed downstream as repo documentation in post-match analysis files.

**Convention B - design brainstorms:** The user also runs design brainstorms (e.g. Phase 2 design for "The Crumb", an AFL coaching AI with a 13-agent hierarchy). These arrive as a small set of explicit numbered questions to answer in coaching language. Do NOT force these into the post-match themed-header format - answer the questions as asked, in order. These are explicitly "do not write to files" - the user merges the output with other agents' analysis into a design doc themselves.

**Why:** Both formats slot into downstream pipelines but different ones - post-match into repo documentation, design brainstorms into design docs the user assembles. Forcing one format onto the other breaks the merge.

**How to apply:** Detect which shape the request is. "Brainstorm across N themes for [match]" -> Convention A, preserve per-theme structure with "what this means next time" bullets. "Answer these N questions [about a design/system]" -> Convention B, answer the numbered questions directly, no themed headers, no file writes unless explicitly asked. Apply council deliberation protocol within either shape. Match requested length; do not over-expand.

**Vocabulary note:** User is comfortable with full AFL coaches'-box vocabulary (handball receiver, supply chain, KPF, half-back rebound, premiership quarter, etc). Do not dilute.

**The Crumb context:** "The Crumb" is the user's AFL coaching AI - a 13-agent hierarchy (Tier 1 Senior Coach orchestrator down to Tier 6 Data Steward). FootyStrategy is one of its Tier 4 agents (the tactical council). Phase 2 design work is ongoing as of 2026-05-15, evaluating CrewAI-style production patterns (planner-executor split, role-based data isolation, supervisor-worker graphs).
