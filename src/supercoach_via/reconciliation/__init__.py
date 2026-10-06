"""Opt-in AFL Tables reconciliation audit (``scvia reconcile-afltables``).

Deterministic Python only: no model call participates in a verdict. The audit compares
pinned local data with a frozen capture of AFL Tables pages and never edits either.
See ``docs/rewrite/afltables-reconciliation/DESIGN.md``.
"""

SCHEMA_VERSION = 1
