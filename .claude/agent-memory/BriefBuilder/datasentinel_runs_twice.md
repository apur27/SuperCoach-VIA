---
name: datasentinel-runs-twice
description: DataSentinel gates the council chain in two passes (before and after FootyStrategy); a Pass-1 PASS is not final clearance
metadata:
  type: project
---

DataSentinel runs TWICE in the canonical council chain — before and after FootyStrategy. Do not treat a pre-FootyStrategy DataSentinel PASS as final clearance.

Canonical chain: BriefBuilder -> DataSentinel (Pass 1: data skeleton) -> FootyStrategy -> DataSentinel (Pass 2: full doc) -> Skeptic -> Gaffer (SHIP).

**Why:** FootyStrategy does not write zero numbers in practice — Pass 2 catches any data-adjacent figures introduced during interpretation fill. Pass 1 only gates the data skeleton BriefBuilder hands over; the ship is gated on Pass 2.

**How to apply:** When BriefBuilder's data skeleton clears Pass 1, that is a checkpoint, not a clearance to publish. The doc still has two gates ahead (Pass 2 + Skeptic) once FootyStrategy adds interpretation. See [[Gaffer canonical council chain]].
