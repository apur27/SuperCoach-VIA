"""S01-S05: XSS / unsafe URL / spreadsheet-formula payloads, ZIP containment, marker replacement."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import pytest

from supercoach_via.publish import bundle, content, reports


class TestCsvSafety:
    @pytest.mark.parametrize("payload", ['=HYPERLINK("x")', "+1+1", "-cmd", "@SUM(A1)", "\t=1", "\r=1"])
    def test_formula_strings_are_neutralised(self, payload: str) -> None:
        out = bundle.safe_csv_bytes(["name", "value"], [[payload, 3]])
        rows = list(csv.reader(io.StringIO(out.decode("utf-8"))))
        assert rows[1][0].startswith("'")
        assert rows[1][1] == "3"

    def test_numeric_cells_unchanged(self) -> None:
        out = bundle.safe_csv_bytes(["a", "b", "c"], [[-5, 1.25, None]])
        assert out.decode().splitlines()[1] == "-5,1.25,"

    def test_negative_number_strings_are_not_mangled_when_numeric(self) -> None:
        out = bundle.safe_csv_bytes(["a"], [["-12.5"]])
        assert out.decode().splitlines()[1] == "-12.5"


class TestZip:
    def test_deterministic_and_contained(self, tmp_path: Path) -> None:
        files = {"README.md": b"hi", "data/a.csv": b"x,y\n"}
        a = bundle.deterministic_zip(files)
        b = bundle.deterministic_zip(dict(reversed(list(files.items()))))
        assert a == b
        with zipfile.ZipFile(io.BytesIO(a)) as zf:
            assert zf.namelist() == ["README.md", "data/a.csv"]
            assert all(i.date_time == (1980, 1, 1, 0, 0, 0) for i in zf.infolist())

    @pytest.mark.parametrize("bad", ["../evil", "/abs", "a/../../b", "c:\\x", ""])
    def test_rejects_unsafe_member(self, bad: str) -> None:
        with pytest.raises(ValueError):
            bundle.deterministic_zip({bad: b"x"})


class TestSanitize:
    @pytest.mark.parametrize(
        "md",
        [
            "<script>alert(1)</script>",
            '<img src=x onerror="alert(1)">',
            "[x](javascript:alert(1))",
            '<iframe src="https://evil.example"></iframe>',
            "<style>body{}</style>",
            '<a href="data:text/html;base64,xx">x</a>',
        ],
    )
    def test_active_content_removed(self, md: str) -> None:
        html = content.render_markdown(md, base="/")
        low = html.lower()
        # inert escaped text is acceptable; active elements/attributes/URLs are not
        for token in ("<script", "onerror=", 'href="javascript', "<iframe", "<style", 'href="data:'):
            assert token not in low

    def test_tables_and_safe_links_survive(self) -> None:
        html = content.render_markdown("| a | b |\n|---|---|\n| 1 | 2 |\n\n[ok](https://afltables.com/x)", base="/")
        assert "<table>" in html and 'href="https://afltables.com/x"' in html
        assert 'rel="noopener noreferrer"' in html

    def test_relative_doc_links_are_mapped_under_base(self) -> None:
        html = content.render_markdown(
            "[x](afl-history.md)",
            base="/SuperCoach-VIA/",
            link_map={"docs/afl-history.md": "articles/afl-history/"},
            source_path="docs/x.md",
        )
        assert 'href="/SuperCoach-VIA/articles/afl-history/"' in html

    def test_data_tags_preserved_as_text(self) -> None:
        html = content.render_markdown("Goals **[data]** 12", base="/")
        assert "[data]" in html


class TestMarkers:
    def test_replaces_single_marked_section(self) -> None:
        doc = "a\n<!-- GEN:x START -->\nold\n<!-- GEN:x END -->\nb\n"
        out = reports.replace_marked_section(doc, "x", "new")
        assert out == "a\n<!-- GEN:x START -->\nnew\n<!-- GEN:x END -->\nb\n"

    @pytest.mark.parametrize(
        "doc",
        [
            "no markers",
            "<!-- GEN:x START -->\n<!-- GEN:x START -->\n<!-- GEN:x END -->",
            "<!-- GEN:x END -->\n<!-- GEN:x START -->",
        ],
    )
    def test_missing_or_duplicate_markers_fail(self, doc: str) -> None:
        with pytest.raises(reports.MarkerError):
            reports.replace_marked_section(doc, "x", "new")
