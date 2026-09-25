"""Import curated Markdown articles listed in the public-content manifest.

Original content is preserved as a frozen/archived publication: numbers are not
recalculated, source tags stay as written, and nothing is restamped as newly verified.
"""

from __future__ import annotations

import hashlib
import posixpath
import re
import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

from supercoach_via.publish.content import render_markdown
from supercoach_via.publish.view_models import Article, ArticleSummary, Source

EXCLUDED_PREFIXES = ("docs/rewrite/", "docs/run-reports/", "docs/sentinel-reports/")
EXCLUDED_NAME = re.compile(r"(council|agent|claude|sentinel|pending-|experiment-log|architecture)", re.I)
_DATE_PREFIX = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-")
_ASOF = re.compile(r"<!--\s*verify-asof:\s*([^>]*?)\s*-->")
_STAMP = re.compile(r"<!--\s*council-pipeline:.*?Gaffer:SHIP@([0-9T:\-+Z.]+)", re.S)
_H1 = re.compile(r"^#\s+(.+?)\s*$", re.M)
_IMG = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    category: str
    scope: Literal["frozen", "live", "archive"]


@dataclass
class BuiltArticles:
    articles: list[Article]
    assets: dict[str, Path] = field(default_factory=dict)


def load_manifest(manifest_path: Path, repo_root: Path) -> list[ManifestEntry]:
    data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("unsupported public-content manifest version")
    entries: list[ManifestEntry] = []
    seen: set[str] = set()
    for raw in data.get("article", []):
        path = str(raw["path"])
        norm = posixpath.normpath(path)
        if norm != path or not path.startswith("docs/") or not path.endswith(".md") or ".." in path.split("/"):
            raise ValueError(f"article path must be a normalized docs/*.md path: {path!r}")
        if path.startswith(EXCLUDED_PREFIXES) or EXCLUDED_NAME.search(Path(path).name):
            raise ValueError(f"operator/agent document cannot be public: {path}")
        if path in seen:
            raise ValueError(f"duplicate manifest entry {path}")
        if raw.get("scope") not in ("frozen", "live", "archive"):
            raise ValueError(f"invalid scope for {path}")
        resolved = (repo_root / path).resolve()
        if repo_root.resolve() not in resolved.parents or not resolved.is_file():
            raise ValueError(f"article missing or outside repository: {path}")
        seen.add(path)
        entries.append(ManifestEntry(path, str(raw["category"]), raw["scope"]))
    return entries


def slug_for(path: str) -> str:
    stem = Path(path).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    if not slug:
        raise ValueError(f"cannot derive slug from {path}")
    return slug[:120]


def _excerpt(text: str) -> str:
    body = _ASOF.sub("", re.sub(r"<!--.*?-->", "", text, flags=re.S))
    for para in re.split(r"\n\s*\n", body):
        p = para.strip()
        if p and not p.startswith(("#", "|", "!", "<", "---", ">", "```", "- ", "* ")):
            plain = re.sub(r"[*_`\[\]]", "", re.sub(r"\]\([^)]*\)", "]", p))
            return re.sub(r"\s+", " ", plain)[:280]
    return ""


_LOCAL_PATH = re.compile(r"(?:/home|/Users|/tmp)/[^\s`\"'<>)\]]*")
_REPO_DIR = "/SuperCoach-VIA/"


def redact_local_paths(text: str) -> tuple[str, int]:
    """Replace absolute local filesystem paths in legacy prose (never publish operator paths).

    A path inside a checkout of this repository keeps its repo-relative remainder; any other
    local path becomes a visible placeholder. Returns the text and the number of paths replaced.
    """

    def sub(m: re.Match[str]) -> str:
        path = m.group(0)
        i = path.find(_REPO_DIR)
        return path[i + len(_REPO_DIR) :] if i >= 0 and path[i + len(_REPO_DIR) :] else "[local path removed]"

    return _LOCAL_PATH.subn(sub, text)


_HEADING_TAG = re.compile(r"<(/?)h([1-5])(?=[\s>])")


def demote_headings(html: str) -> str:
    """Shift sanitized h1..h5 down one level so the page layout's title stays the only H1."""
    return _HEADING_TAG.sub(lambda m: f"<{m.group(1)}h{int(m.group(2)) + 1}", html)


def build_articles(repo_root: Path, manifest_path: Path, *, base: str, asset_prefix: str) -> BuiltArticles:
    entries = load_manifest(manifest_path, repo_root)
    slugs = {e.path: slug_for(e.path) for e in entries}
    if len(set(slugs.values())) != len(slugs):
        raise ValueError("article slugs collide")
    link_map = {path: f"articles/{slug}/" for path, slug in slugs.items()}
    built = BuiltArticles(articles=[])
    for entry in entries:
        raw = (repo_root / entry.path).read_bytes()
        text, redacted = redact_local_paths(raw.decode("utf-8"))
        local_map = dict(link_map)
        for ref in _IMG.findall(text):
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(entry.path), ref.split("#")[0]))
            if (
                resolved.startswith("assets/")
                and (repo_root / resolved).is_file()
                and resolved.endswith((".png", ".svg"))
            ):
                local_map[resolved] = asset_prefix + resolved
                built.assets[resolved] = repo_root / resolved
        html = demote_headings(render_markdown(text, base=base, link_map=local_map, source_path=entry.path))
        h1 = _H1.search(text)
        m = _DATE_PREFIX.match(Path(entry.path).name)
        published = date(int(m[1]), int(m[2]), int(m[3])) if m else None
        asof = _ASOF.search(text)
        stamp = _STAMP.search(text)
        as_of = asof.group(1) if asof else (stamp.group(1)[:10] if stamp else None)
        slug = slugs[entry.path]
        summary = ArticleSummary(
            slug=slug,
            title=(h1.group(1).strip("* ") if h1 else slug.replace("-", " ").title())[:200],
            category=entry.category,
            published=published,
            as_of=as_of,
            scope=entry.scope,
            excerpt=_excerpt(text),
            editorial_state="published_archive",
            original_path=entry.path,
            resource=f"articles/{slug}.json",
        )
        digest = hashlib.sha256(raw).hexdigest()
        built.articles.append(
            Article(
                summary=summary,
                html=html,
                sources=[
                    Source(label="Original repository document", url=None, note=f"{entry.path} sha256 {digest[:16]}")
                ],
                provenance=(
                    (
                        f"Imported from {entry.path} (sha256 {digest}) with {redacted} local filesystem "
                        "path(s) redacted; otherwise unchanged."
                        if redacted
                        else f"Imported unchanged from {entry.path} (sha256 {digest})."
                    )
                    + " Figures are frozen at the article's own as-of scope and were not re-verified or"
                    " recalculated by the rewrite."
                ),
            )
        )
    return built
