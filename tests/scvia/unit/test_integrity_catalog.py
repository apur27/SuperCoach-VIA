"""The rule catalog: unique rule ids, every rule bound to a registered check, every check reachable."""

from __future__ import annotations

from supercoach_via.integrity import (
    checks_data,
    checks_models,
    checks_release,
    checks_source,
    checks_storage,
)
from supercoach_via.integrity.runner import registry, rule_catalog

FAMILY_MODULES = (checks_storage, checks_data, checks_source, checks_release, checks_models)


def test_rule_ids_are_unique_and_namespaced() -> None:
    ids = [r.rule_id for m in FAMILY_MODULES for r in m.RULES]
    assert len(ids) == len(set(ids))
    assert all("." in i and i == i.lower() for i in ids)


def test_every_rule_belongs_to_a_registered_check() -> None:
    checks = {c.check_id for c in registry()}
    for rule in rule_catalog().values():
        if rule.check_id == "policy.exceptions":
            continue  # bookkeeping rules owned by the collector
        assert rule.check_id in checks, rule.rule_id


def test_every_check_owns_rules_and_a_family() -> None:
    owned = {r.check_id for r in rule_catalog().values()}
    for c in registry():
        assert c.family and c.summary
        assert c.check_id in owned, c.check_id


def test_every_rule_names_an_operator_action() -> None:
    for rule in rule_catalog().values():
        assert rule.action and rule.summary, rule.rule_id


def test_operator_doc_lists_every_rule() -> None:
    from pathlib import Path

    doc = (Path(__file__).resolve().parents[3] / "docs" / "data-integrity.md").read_text(encoding="utf-8")
    missing = sorted(r for r in rule_catalog() if f"`{r}`" not in doc)
    assert not missing, f"docs/data-integrity.md lacks rules: {missing}"
    for rule in ("policy.exception_rejected_current_season", "policy.exception_stale"):
        assert f"`{rule}`" in doc
