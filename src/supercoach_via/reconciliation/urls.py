"""Reconciliation source policy and link normalisation (DESIGN section 6).

Relative ``href`` values are resolved against the page's *final* URL, then the fragment is
stripped, then the result is validated by the allowlist. Nothing here constructs a player URL
from a name: profile URLs come only from links found on captured pages.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlsplit

from supercoach_via.ingest.http import PolicySet, UrlRejectedError, load_policies, validate_url
from supercoach_via.settings import default_config_dir

RECONCILIATION_POLICY_NAME = "afltables_reconciliation"
POLICY_FILE = "reconciliation_source_policy.toml"
SITE = "https://afltables.com"


def policy_path(config_dir: Path | None = None) -> Path:
    return (config_dir or default_config_dir()) / POLICY_FILE


def load_reconciliation_policies(config_dir: Path | None = None) -> PolicySet:
    return load_policies(policy_path(config_dir))


def normalise_link(href: str, page_final_url: str) -> str | None:
    """Absolute, fragment-free URL for ``href`` found on ``page_final_url``.

    ``None`` for same-page fragments and non-web schemes (nothing to fetch). Other hosts are
    returned unchanged so the caller's policy check can record them as rejected links.
    """
    href = href.strip()
    if not href or href.startswith("#"):
        return None
    scheme = urlsplit(href).scheme.lower()
    if scheme in ("mailto", "javascript", "tel", "data"):
        return None
    joined, _fragment = urldefrag(urljoin(page_final_url, href))
    return joined or None


def classify_link(url: str, policies: PolicySet) -> str | None:
    """``None`` when ``url`` is allowed, else the rejection reason."""
    try:
        validate_url(url, policies)
    except UrlRejectedError as exc:
        return str(exc)
    return None
