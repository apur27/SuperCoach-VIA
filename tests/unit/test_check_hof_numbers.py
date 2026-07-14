"""TDD tests for scripts/check_hof_numbers.py."""
import importlib.util
import json
from pathlib import Path

import pytest

_MOD_PATH = Path(__file__).parent.parent.parent / "scripts" / "check_hof_numbers.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_hof_numbers", _MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup_repo(tmp_path, json_data, page_content, key="career_games", subpage="hall-of-fame-stat-games.md"):
    json_file = tmp_path / "_stat_leaders.json"
    json_file.write_text(json.dumps(json_data))
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / subpage).write_text(page_content)
    return json_file


def test_matching_json_and_markdown_returns_zero(tmp_path, capsys):
    chn = _load()
    json_data = {
        "categories": {
            "career_games": {
                "leaders": [{"rank": 1, "rank_label": "1", "name": "Scott Pendlebury", "total": 435.0}]
            }
        }
    }
    json_file = _setup_repo(
        tmp_path, json_data,
        "| 1 | Scott Pendlebury **[data]** | Collingwood | 2006-2026 | 435 |<!-- HOF-TOP:career_games -->\n",
    )
    result = chn.check_hof_numbers(json_path=json_file, repo_root=tmp_path)
    assert result == 0


def test_mismatch_returns_one_with_message(tmp_path, capsys):
    chn = _load()
    json_data = {
        "categories": {
            "career_games": {
                "leaders": [{"rank": 1, "rank_label": "1", "name": "Scott Pendlebury", "total": 435.0}]
            }
        }
    }
    json_file = _setup_repo(
        tmp_path, json_data,
        "| 1 | Scott Pendlebury **[data]** | Collingwood | 2006-2026 | 434 |<!-- HOF-TOP:career_games -->\n",
    )
    result = chn.check_hof_numbers(json_path=json_file, repo_root=tmp_path)
    captured = capsys.readouterr()
    assert result == 1
    assert "434" in captured.out or "435" in captured.out


def test_missing_json_key_skips_gracefully(tmp_path):
    chn = _load()
    json_data = {"categories": {}}
    json_file = tmp_path / "_stat_leaders.json"
    json_file.write_text(json.dumps(json_data))
    result = chn.check_hof_numbers(json_path=json_file, repo_root=tmp_path)
    assert result == 0


def _write_hub(tmp_path, rows):
    """rows: list of (key, inline_text). Writes a minimal hub with HOF-HUB sentinels."""
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    lines = ["| Stat | Page | #1 |", "|--|--|--|"]
    for key, text in rows:
        lines.append(f"| {key} | [x](x.md) | {text} |<!-- HOF-HUB:{key} -->")
    (docs / "hall-of-fame-stat-leaders.md").write_text("\n".join(lines) + "\n")


def test_hub_matching_returns_zero(tmp_path):
    chn = _load()
    json_data = {"categories": {"career_goals": {"leaders": [{"rank": 1, "name": "Tony Lockett", "total": 1360.0}]}}}
    json_file = tmp_path / "_stat_leaders.json"
    json_file.write_text(json.dumps(json_data))
    _write_hub(tmp_path, [("career_goals", "Tony Lockett 1,360 **[data]**")])
    assert chn.check_hof_hub(json_data["categories"], tmp_path) == 0
    # And the full checker passes (no subpages present, hub matches).
    assert chn.check_hof_numbers(json_path=json_file, repo_root=tmp_path) == 0


def test_hub_mismatch_returns_one(tmp_path, capsys):
    chn = _load()
    json_data = {"categories": {"career_goals": {"leaders": [{"rank": 1, "name": "Tony Lockett", "total": 1360.0}]}}}
    json_file = tmp_path / "_stat_leaders.json"
    json_file.write_text(json.dumps(json_data))
    _write_hub(tmp_path, [("career_goals", "Tony Lockett 1,359 **[data]**")])  # wrong
    assert chn.check_hof_hub(json_data["categories"], tmp_path) == 1
    # A hub mismatch must FAIL the overall gate even with no subpages.
    assert chn.check_hof_numbers(json_path=json_file, repo_root=tmp_path) == 1
    out = capsys.readouterr().out
    assert "hub career_goals" in out


def test_hub_absent_is_not_a_failure(tmp_path):
    chn = _load()
    json_data = {"categories": {"career_goals": {"leaders": [{"rank": 1, "name": "X", "total": 1360.0}]}}}
    # No hub file created.
    assert chn.check_hof_hub(json_data["categories"], tmp_path) == 0


def test_missing_subpage_skips_gracefully(tmp_path):
    chn = _load()
    json_data = {
        "categories": {
            "career_games": {
                "leaders": [{"rank": 1, "rank_label": "1", "name": "P", "total": 435.0}]
            }
        }
    }
    json_file = tmp_path / "_stat_leaders.json"
    json_file.write_text(json.dumps(json_data))
    # No docs directory / subpage created
    result = chn.check_hof_numbers(json_path=json_file, repo_root=tmp_path)
    assert result == 0
