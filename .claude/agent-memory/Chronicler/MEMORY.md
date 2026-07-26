# Chronicler Memory Index

- [Project baselines](project_baselines.md) — per-cycle metric baselines (tests: 239 as of 2026-07-03); compare each run to spot drift
- [Backtest/eval fragility](project_backtest_fragility.md) — non-resumable Optuna + date-stamp decoupled from content round; check backtest max-round vs matches max-round every run
- [Unrunnable-gate defect class](project_unrunnable_gate_class.md) — a gate that cannot run reads as a gate that passed; verify reachability before recording any PASS
- [Recommendation ledger](project_recommendation_ledger.md) — adopted / explicitly-deferred / open recommendations; check before re-raising an item
- [Council-doc staleness pattern](project_council_doc_staleness.md) — 6/8 legacy docs FAIL genuine re-check; staleness is the dominant failure mode; re-verification gate is the standing recommendation
