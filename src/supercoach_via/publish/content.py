"""Curated article import: Markdown -> allowlist-sanitized HTML, link mapping, metadata.

Legacy Markdown is treated as plain content (never MDX / executable). Raw HTML inside it
is parsed and then sanitized with an allowlist (nh3); scripts, styles, iframes, event
handlers and non-http(s)/mailto URLs are removed. Relative links are mapped through an
explicit link map to public routes under the configured base, or dropped.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Mapping

import nh3
from markdown_it import MarkdownIt

ALLOWED_TAGS = {
    "a",
    "abbr",
    "b",
    "blockquote",
    "br",
    "code",
    "dd",
    "del",
    "details",
    "div",
    "dl",
    "dt",
    "em",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "i",
    "img",
    "li",
    "ol",
    "p",
    "pre",
    "s",
    "span",
    "strong",
    "sub",
    "summary",
    "sup",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
}
ALLOWED_ATTRS: dict[str, set[str]] = {
    "a": {"href", "title"},
    "img": {"src", "alt", "title", "width", "height"},
    "th": {"align", "scope"},
    "td": {"align"},
    "abbr": {"title"},
    "code": {"class"},
}
_ABS = re.compile(r"^(https?:)?//|^mailto:", re.I)


def _md() -> MarkdownIt:
    return MarkdownIt("commonmark", {"html": True, "linkify": False, "typographer": False}).enable(
        ["table", "strikethrough"]
    )


def _map_link(
    value: str,
    *,
    base: str,
    link_map: Mapping[str, str],
    source_path: str | None,
) -> str | None:
    value = value.strip()
    if not value:
        return None
    if value.startswith("#"):
        return value
    if _ABS.match(value):
        return value if value.lower().startswith(("https://", "http://", "mailto:")) else None
    if ":" in value.split("/", 1)[0]:
        return None  # any other scheme (javascript:, data:, vbscript: ...)
    path, _, frag = value.partition("#")
    if source_path is None:
        return None
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source_path), path))
    target = link_map.get(resolved)
    if target is None:
        return None
    return base + target + (f"#{frag}" if frag else "")


def render_markdown(
    text: str,
    *,
    base: str,
    link_map: Mapping[str, str] | None = None,
    source_path: str | None = None,
) -> str:
    html = _md().render(text)
    mapping = link_map or {}

    def attribute_filter(tag: str, attr: str, value: str) -> str | None:
        if attr in ("href", "src"):
            return _map_link(value, base=base, link_map=mapping, source_path=source_path)
        return value

    return nh3.clean(
        html,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRS,
        url_schemes={"http", "https", "mailto"},
        link_rel="noopener noreferrer",
        attribute_filter=attribute_filter,
        strip_comments=True,
    )
