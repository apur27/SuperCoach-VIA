# Run Reports Index

Newest first. Format: `YYYY-MM-DD | cycle-type | commit | one-line summary`

- 2026-07-13 | weekly-r20 | 2c50e6736 | R19 settled → R20 predictions (413 rows); backtest MAE 3.973 (R1–R19 3.960); 12 HOF pages regenerated. Harness fail-closed & recovered twice (phantom-row ABORT on dropped game → counter-aware delta fix a4c1fdc20; DataSentinel 14 untagged nums caught). 352 tests. HOF hub HELD BACK — three-way kicks/handballs drift, not gated.
- 2026-07-07 | weekly-r19 | 12fa8202d | R18 settled → R19 predictions (412 rows); HOF + 13 pages regenerated, check_hof_numbers PASS; CR-1 round-detection bug fixed. KNOWN GAP: R18 backtest killed mid-Optuna, eval surface frozen at R17; QA PASS_WITH_WARNINGS (244/244).
- 2026-07-03 | harness-hardening-sprint-1 | e84cd41f9 | Council pipeline hardened (enforce=1, shared tag vocab, staged-blob gate, trust badge); 239 tests green; F1 genuine-DataSentinel run on 8 legacy docs returned 6 FAILs (3 staleness, 3 authoring errors incl. jonathan-brown 46≠49) — 15 docs badged, 6 quarantined.
