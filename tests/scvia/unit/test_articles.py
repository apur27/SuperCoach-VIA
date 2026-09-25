"""Curated article import: explicit manifest, frozen as-of, link/asset mapping, exclusion."""

from __future__ import annotations

from pathlib import Path

import pytest

from supercoach_via.publish import articles


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "docs" / "news").mkdir(parents=True)
    (tmp_path / "docs" / "rewrite").mkdir()
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "chart.png").write_bytes(b"\x89PNG demo")
    (tmp_path / "docs" / "news" / "2026-05-13-demo-story.md").write_text(
        "# Demo Story\n\n<!-- verify-asof: round=9 -->\nIntro text with **[data]** 12 goals.\n\n"
        "See [history](../afl-history.md) and [plan](../rewrite/PLAN.md).\n\n![chart](../../assets/chart.png)\n"
    )
    (tmp_path / "docs" / "afl-history.md").write_text("# AFL History\n\nBody.\n")
    (tmp_path / "docs" / "rewrite" / "PLAN.md").write_text("# secret plan\n")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "public_content.toml").write_text(
        'schema_version = 1\n[[article]]\npath = "docs/news/2026-05-13-demo-story.md"\ncategory = "news"\nscope = "frozen"\n'
        '[[article]]\npath = "docs/afl-history.md"\ncategory = "report"\nscope = "archive"\n'
    )
    return tmp_path


def test_only_manifest_articles_are_built(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    built = articles.build_articles(
        repo, repo / "config" / "public_content.toml", base="/SuperCoach-VIA/", asset_prefix="data/r1/"
    )
    slugs = [a.summary.slug for a in built.articles]
    assert slugs == ["2026-05-13-demo-story", "afl-history"]
    story = built.articles[0]
    assert story.summary.title == "Demo Story"
    assert story.summary.published is not None and story.summary.published.isoformat() == "2026-05-13"
    assert story.summary.as_of == "round=9" and story.summary.scope == "frozen"
    assert story.summary.editorial_state == "published_archive"
    assert 'href="/SuperCoach-VIA/articles/afl-history/"' in story.html
    assert "PLAN.md" not in story.html.split("plan")[0] or "rewrite" not in story.html
    assert 'src="/SuperCoach-VIA/data/r1/assets/chart.png"' in story.html
    assert built.assets == {"assets/chart.png": repo / "assets" / "chart.png"}
    assert "[data]" in story.html
    assert "not re-verified" in story.provenance


def test_manifest_rejects_paths_outside_docs(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "config" / "public_content.toml").write_text(
        'schema_version = 1\n[[article]]\npath = "docs/../.env"\ncategory = "news"\nscope = "frozen"\n'
    )
    with pytest.raises(ValueError):
        articles.build_articles(repo, repo / "config" / "public_content.toml", base="/", asset_prefix="data/r/")


def test_manifest_rejects_rewrite_and_agent_docs(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "config" / "public_content.toml").write_text(
        'schema_version = 1\n[[article]]\npath = "docs/rewrite/PLAN.md"\ncategory = "report"\nscope = "archive"\n'
    )
    with pytest.raises(ValueError):
        articles.build_articles(repo, repo / "config" / "public_content.toml", base="/", asset_prefix="data/r/")


def test_real_manifest_lists_existing_files(repo_root: Path) -> None:
    entries = articles.load_manifest(repo_root / "config" / "public_content.toml", repo_root)
    assert len(entries) > 50
    assert not any("rewrite" in e.path or "council" in e.path for e in entries)


def test_local_filesystem_paths_are_redacted_and_disclosed(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    story = repo / "docs" / "news" / "2026-05-13-demo-story.md"
    story.write_text(
        story.read_text()
        + "\n- **Snapshot path:** `/home/abhi/git/SuperCoach-VIA/data/live_snapshots/x_q1.json`\n\n"
        "```\n/home/abhi/sourceCode/python/coding/.venv/bin/python -c 'print(1)'\n```\n"
        "Also /Users/someone/tmp/file.csv and /tmp/scratch/out.json.\n"
    )
    built = articles.build_articles(
        repo, repo / "config" / "public_content.toml", base="/SuperCoach-VIA/", asset_prefix="data/r1/"
    )
    html = built.articles[0].html
    for marker in ("/home/", "/Users/", "/tmp/"):
        assert marker not in html
    assert "data/live_snapshots/x_q1.json" in html  # repo-relative remainder kept
    assert html.count("[local path removed]") == 3
    assert "4 local filesystem path(s) redacted" in built.articles[0].provenance
    assert "Imported unchanged" not in built.articles[0].provenance
