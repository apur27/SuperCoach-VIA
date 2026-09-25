"""R11: every public surface of one demo release agrees on values, snapshot and cutoff.

Independent re-reads only: JSON via ``json``, CSV via ``csv``, Markdown via regex, ZIP via
``zipfile`` and chart labels/alt text via the charts manifest. Nothing here calls the
builder's own view-model helpers to re-derive expected values.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any

import pytest

from supercoach_via.publish import builder as B
from tests.scvia.unit.demo_release_env import FORECAST_CUTOFF, full_release


@pytest.fixture(scope="module")
def cand(tmp_path_factory: pytest.TempPathFactory) -> B.ReleaseCandidate:
    c = full_release(tmp_path_factory.getbasetemp())  # the same build test_builder uses
    assert c.ok, c.validation.issues[:5]
    return c


def _j(c: B.ReleaseCandidate, rel: str) -> Any:
    return json.loads((c.public_dir / rel).read_text(encoding="utf-8"))


def _csv(c: B.ReleaseCandidate, rel: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO((c.public_dir / rel).read_text(encoding="utf-8"))))


def _md(c: B.ReleaseCandidate, name: str) -> str:
    return (c.public_dir / "downloads" / "reports" / f"{name}.md").read_text(encoding="utf-8")


def _md_tables(text: str) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    cur: list[list[str]] = []
    for line in text.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                cur.append(cells)
        elif cur:
            tables.append(cur)
            cur = []
    if cur:
        tables.append(cur)
    return tables


def test_snapshot_release_and_cutoff_agree_everywhere(cand: B.ReleaseCandidate) -> None:
    manifest = _j(cand, "release.json")
    sid, rid = manifest["snapshot_id"], manifest["release_id"]
    assert _j(cand, "overview.json")["snapshot_id"] == sid
    assert _j(cand, "downloads.json")["release_id"] == rid
    pset = _j(cand, manifest["forecast"]["artifact"])
    cutoffs = {r["forecast_cutoff"] for r in pset["rows"]} | {pset["forecast_cutoff"]}
    assert len(cutoffs) == 1
    cutoff = cutoffs.pop()
    assert cutoff.startswith(FORECAST_CUTOFF.strftime("%Y-%m-%dT%H:%M"))
    side = _j(cand, "downloads/predictions-legacy.manifest.json")
    assert side["snapshot_id"] == sid and side["release_id"] == rid and side["forecast_cutoff"] == cutoff
    charts = _j(cand, "downloads/charts.json")
    assert charts["snapshot_id"] == sid and charts["release_id"] == rid
    for name in ("season-summary", "stat-leaders", "team-analysis"):
        text = _md(cand, name)
        assert sid in text and rid in text, name
    assert cutoff in _md(cand, "season-summary")
    readme = (cand.public_dir / "downloads/README.md").read_text()
    assert sid in readme and rid in readme and cutoff in readme
    with zipfile.ZipFile(cand.public_dir / "downloads/fan-pack.zip") as zf:
        pack = json.loads(zf.read("manifest.json"))
        assert pack["snapshot_id"] == sid and pack["release_id"] == rid and pack["forecast_cutoff"] == cutoff
        assert zf.read("README.md").decode() == readme


def test_prediction_json_csv_legacy_and_markdown_agree(cand: B.ReleaseCandidate) -> None:
    manifest = _j(cand, "release.json")
    pset = _j(cand, manifest["forecast"]["artifact"])
    by_id = {r["prediction_id"]: r for r in pset["rows"]}
    rich = _csv(cand, "downloads/predictions.csv")
    assert len(rich) == len(by_id)
    for row in rich:
        src = by_id[row["prediction_id"]]
        assert float(row["predicted_disposals"]) == src["predicted_disposals"]  # full precision round-trip
        assert row["player_id"] == src["player_id"] and row["match_id"] == src["match_id"]
        assert row["snapshot_id"] == src["snapshot_id"]
    side = _j(cand, "downloads/predictions-legacy.manifest.json")
    legacy = _csv(cand, "downloads/predictions-legacy.csv")
    for line, meta in zip(legacy, side["rows"], strict=True):
        src = by_id[meta["prediction_id"]]
        assert meta["predicted_disposals"] == src["predicted_disposals"]
        assert int(line["predicted_disposals"]) == round(src["predicted_disposals"])
        assert line["team"] == src["club_name"]
    # season summary lists the top predictions with one decimal
    top = sorted(pset["rows"], key=lambda r: (-r["predicted_disposals"], r["prediction_id"]))[:5]
    text = _md(cand, "season-summary")
    for r in top:
        assert f"| {r['player_name']} | {r['club_name']} |" in text
        assert f"{r['predicted_disposals']:.1f}" in text
    ov = _j(cand, "overview.json")
    assert [r["prediction_id"] for r in ov["prediction_highlights"]] == [r["prediction_id"] for r in top]


def test_leaders_agree_across_overview_markdown_and_chart(cand: B.ReleaseCandidate) -> None:
    ov = _j(cand, "overview.json")
    disposals = [r for r in ov["leaders"] if r["stat"] == "disposals"]
    assert disposals
    tables = _md_tables(_md(cand, "stat-leaders"))
    disp_table = next(t for t in tables if t[0][:2] == ["Rank", "Player"] and "disposals" in " ".join(t[0]).lower())
    assert len(disp_table) - 1 == len(disposals)
    for (rank, name, value, games), leader in zip(disp_table[1:], disposals, strict=True):
        assert name == leader["name"] and float(value) == leader["value"] and int(games) == leader["observed_games"]
        assert int(rank) >= 1
    charts = {c["key"]: c for c in _j(cand, "downloads/charts.json")["charts"]}
    chart = charts["season_disposal_leaders"]
    assert chart["status"] == "available"
    assert [p[0] for p in chart["plotted"]] == [r["name"] for r in disposals]
    assert [p[1] for p in chart["plotted"]] == [r["value"] for r in disposals]
    for r in disposals:
        assert f"{r['name']}: {r['value']:.1f} disposals" in chart["alt_text"]
    # the same alt text is what the Markdown report uses for the image
    assert f"![{chart['alt_text']}](../{chart['path'].removeprefix('downloads/')})" in _md(cand, "stat-leaders")
    png = (cand.public_dir / chart["path"]).read_bytes()
    assert png.startswith(b"\x89PNG")


def test_team_analysis_matches_team_json(cand: B.ReleaseCandidate) -> None:
    season = _j(cand, "release.json")["season"]
    text = _md(cand, "team-analysis")
    ladder = next(t for t in _md_tables(text) if t[0][:3] == ["Pos", "Club", "P"])
    for pos, club, played, won, lost, drawn, pts, pct in ladder[1:]:
        team = next(t for t in _j(cand, "teams/index.json")["teams"] if t["name"] == club)
        ts = _j(cand, f"teams/{team['club_id']}/{season}.json")
        row = next(r for r in ts["ladder"] if r["club_id"] == team["club_id"])
        assert (int(pos), int(played), int(won), int(lost), int(drawn), int(pts)) == (
            row["position"],
            row["played"],
            row["won"],
            row["lost"],
            row["drawn"],
            row["premiership_points"],
        )
        assert (pct == "-" and row["percentage"] is None) or abs(float(pct) - row["percentage"]) < 0.05
        assert ts["position"] == row["position"]


def test_players_csv_matches_player_json(cand: B.ReleaseCandidate) -> None:
    idx = {p["id"]: p for p in _j(cand, "players/index.json")["players"]}
    rows = _csv(cand, "downloads/players.csv")
    assert {r["player_id"] for r in rows} == set(idx)
    for r in rows:
        detail = _j(cand, f"players/{idx[r['player_id']]['key']}.json")
        assert int(r["career_games"]) == detail["career_games"]
        career = {s["stat"]: s for s in detail["career"]}
        for stat in ("disposals", "goals"):
            total = career[stat]["total"] if stat in career else None  # no games -> no career lines
            assert (r[f"{stat}_total"] == "" and total is None) or float(r[f"{stat}_total"]) == total
            assert int(r[f"{stat}_observed_games"]) == (career[stat]["observed_games"] if stat in career else 0)


def test_accuracy_rows_reproduce_the_headline(cand: B.ReleaseCandidate) -> None:
    idx = _j(cand, "accuracy/index.json")
    report = _j(cand, idx["reports"][0]["resource"])
    assert report["rows_resource"] == "downloads/accuracy-rows.csv"
    rows = [r for r in _csv(cand, report["rows_resource"]) if r["origin"] == report["origin"]]
    errs = [float(r["predicted_disposals"]) - float(r["actual"]) for r in rows]
    assert len(errs) == report["headline"]["n"]
    assert math.isclose(sum(abs(e) for e in errs) / len(errs), report["headline"]["mae"], rel_tol=1e-12)
    assert math.isclose(sum(errs) / len(errs), report["headline"]["bias"], rel_tol=1e-9, abs_tol=1e-12)


def test_zip_members_are_the_published_bytes(cand: B.ReleaseCandidate) -> None:
    dl = {i["path"]: i for i in _j(cand, "downloads.json")["items"]}
    with zipfile.ZipFile(cand.public_dir / "downloads/fan-pack.zip") as zf:
        members = [n for n in zf.namelist() if n not in ("manifest.json", "checksums.sha256")]
        assert members
        for name in members:
            published = f"downloads/{name}"
            assert published in dl, name
            assert hashlib.sha256(zf.read(name)).hexdigest() == dl[published]["sha256"]
    # every Markdown image reference resolves inside the release
    for name in ("season-summary", "stat-leaders", "team-analysis"):
        for ref in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", _md(cand, name)):
            assert (cand.public_dir / "downloads" / "reports" / ref).resolve().is_file(), ref
            assert Path(ref).parts[0] == ".."
