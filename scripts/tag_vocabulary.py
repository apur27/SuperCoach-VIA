"""tag_vocabulary.py — single source of truth for this repo's provenance-tag
vocabulary.

Council docs annotate every source-backed number with a provenance tag. There
are exactly three genuine written forms per tag kind:

    **[data]**                            bare bold tag
    **[data: file ; column ; agg ...]**   bold tag carrying the derivation spec
    [data]:                               inline colon form (e.g. forgotten-heroes)

The same three shapes apply to the ``historical record`` kind. This module is
the ONE matcher both ``inject_trust_badge.py`` (the badge's N) and
``skeptic_sample_tags.py`` (Skeptic's smoke sample) must import, so the two can
never drift apart again.

Scope boundary — positive recognition ONLY. This module recognises the three
genuine tag forms and nothing else. It deliberately does NOT contain the
untagged-number backstop (DataSentinel step 6: flag player-stat-shaped numbers
that carry NO tag). That backstop must stay independent and live in
DataSentinel's own logic: if the TAG-WALK (step 4/5, which uses extract_tags)
and the untagged-number SCAN shared this matcher, any tag form this vocabulary
failed to recognise would be invisible to BOTH — a silent hole where the
vocabulary blinds its own backstop. Keeping the scan out of this module means an
unrecognised tag still surfaces as an untagged number. To help a caller mask
recognised tags before a broad numeric scan, extract_tags() exposes each tag's
``span`` (start, end char offsets into the ORIGINAL text).

Explicitly NOT tags:
  * meta references in prose — ``verified [data] tags``, ``every [data] figure``
    (a ``[data]`` followed by whitespace/word, not wrapped in ``**…**`` and not
    ``]:``);
  * anything on the provenance stamp line (``<!-- council-pipeline: ... -->``),
    which is stripped before matching.

Verification scope (badge honesty). This module enumerates tags subject to
verification — the ``data`` kind is the only one that feeds the trust badge's N
(see count_tags's default ``kinds=("data",)``). ``historical record`` is
recognised so Skeptic can sample it, but it is an unverifiable-against-CSV claim
and never counts toward the badge. ``unverified`` and plain *unbold* ``[data]``
(meta-prose like "every [data] figure") are outside the vocabulary entirely, so
the badge can never be inflated by them. Do NOT widen the ``data`` vocabulary to
plain unbold ``[data]`` — it would count meta-prose and corrupt badge honesty
everywhere. Detecting numbers that SHOULD be tagged but are NOT is deliberately
out of scope here; DataSentinel handles that via its own independent
untagged-number scan (see the scope-boundary paragraph above).

The ``kinds=("data",)`` count is equivalent to the validated reference matcher:

    re.findall(r"\\*\\*\\[data(?:\\]|\\s*:)|(?<!\\*)\\[data\\]:", stamp_stripped)
"""
import re

DEFAULT_KINDS = ("data", "historical record")

# Provenance stamp lines are never counted. `.` does not match newline, so the
# line's content is blanked while the trailing "\n" is preserved — this keeps
# 1-based line numbers aligned for extract_tags().
_STAMP_RE = re.compile(r"^.*<!-- council-pipeline:.*$", re.MULTILINE)


def build_tag_re(kinds=DEFAULT_KINDS):
    """Compile the genuine-tag matcher for the given tag kinds.

    Two capture groups: group 1 = kind for the bold form (`**[kind]**` /
    `**[kind: ...]**`), group 2 = kind for the inline colon form (`[kind]:`).
    The negative lookbehind on the colon branch prevents double-counting
    `**[kind]:` (already caught by the bold branch).
    """
    alt = "|".join(re.escape(k) for k in kinds)
    return re.compile(rf"\*\*\[({alt})(?:\]|\s*:)|(?<!\*)\[({alt})\]:")


# Module-level compiled regex other code can reuse (both default kinds).
TAG_RE = build_tag_re(DEFAULT_KINDS)


def _strip_stamps(text: str) -> str:
    return _STAMP_RE.sub("", text)


def extract_tags(text: str, kinds=DEFAULT_KINDS) -> list[dict]:
    """Genuine tags in document order.

    Each dict: {"line": int (1-based), "tag": "data"|"historical record",
    "text": <stripped source line>, "span": (start, end)}.

    ``span`` is a (start, end) char-offset pair into the ORIGINAL ``text`` — a
    caller (e.g. DataSentinel's untagged-number backstop) can mask these spans
    out before running its own broad numeric scan. Matching runs on the original
    text and skips matches falling on a provenance-stamp line, which is
    equivalent to stripping the stamp first but keeps offsets usable.
    """
    tag_re = build_tag_re(kinds)
    stamp_spans = [(m.start(), m.end()) for m in _STAMP_RE.finditer(text)]
    lines = text.splitlines()

    results: list[dict] = []
    for m in tag_re.finditer(text):
        start = m.start()
        if any(a <= start < b for a, b in stamp_spans):
            continue  # token sits on a provenance-stamp line — never a tag
        kind = m.group(1) or m.group(2)
        line_no = text.count("\n", 0, start) + 1
        source = lines[line_no - 1].strip() if line_no - 1 < len(lines) else ""
        results.append(
            {"line": line_no, "tag": kind, "text": source, "span": (start, m.end())}
        )
    return results


def count_tags(text: str, kinds=("data",)) -> int:
    """Number of genuine tags of the given kinds (default: data only — the
    badge's N). Equivalent to the validated reference matcher."""
    return len(build_tag_re(kinds).findall(_strip_stamps(text)))


def main(argv=None) -> int:
    """CLI: ``python scripts/tag_vocabulary.py <doc.md>``.

    Prints one tab-separated line per recognised tag (default kinds):
    ``line<TAB>tag<TAB>start<TAB>end<TAB>text``. This lets DataSentinel enumerate
    tags deterministically from Bash and mask the (start, end) spans before its
    independent untagged-number scan. Exit 0 with no stdout (a stderr note) when
    the doc has no recognised tags.
    """
    import sys

    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) != 1:
        print("usage: tag_vocabulary.py <doc.md>", file=sys.stderr)
        return 2
    path = argv[0]
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        print(f"cannot read {path}: {exc}", file=sys.stderr)
        return 2

    tags = extract_tags(text)
    if not tags:
        print(f"# no recognised tags in {path}", file=sys.stderr)
        return 0
    for t in tags:
        start, end = t["span"]
        sys.stdout.write(f"{t['line']}\t{t['tag']}\t{start}\t{end}\t{t['text']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
