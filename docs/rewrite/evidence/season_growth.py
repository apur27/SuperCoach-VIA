"""Attribute a built site's bytes to AFL seasons to size the artifact budget (PLAN §12).

Usage: uv run --locked python docs/rewrite/evidence/season_growth.py <built-site-dir> [out.json]

Attribution rules (bytes as stored, not gzip):
- player-games/<player>/<season>.json, matches/<season>/index.json, matches/detail/m__<season>__*.json,
  teams/<club>/<season>.json, history/yearly_top_100/<season>.json, lists/<season>.json -> that season
- players/<key>.json: each seasons[] entry's serialized bytes -> its season; the rest of the file -> base
- everything else (app shell, assets, articles, downloads, all-time history, indexes) -> base
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

SEASON_FILE = [
    re.compile(r"^player-games/[^/]+/(\d{4})\.json$"),
    re.compile(r"^matches/(\d{4})/index\.json$"),
    re.compile(r"^matches/detail/m__(\d{4})__.+\.json$"),
    re.compile(r"^teams/[^/]+/(\d{4})\.json$"),
    re.compile(r"^history/yearly_top_100/(\d{4})\.json$"),
    re.compile(r"^lists/(\d{4})\.json$"),
]
PLAYER_PAGE = re.compile(r"^players/(?!index\.json$)[^/]+\.json$")


def attribute(site: Path) -> dict[str, object]:
    data_dirs = [d for d in (site / "data").iterdir() if d.is_dir()]
    if len(data_dirs) != 1:
        raise SystemExit(f"expected one release under {site}/data, found {len(data_dirs)}")
    rel_root = data_dirs[0]
    by_season: Counter[int] = Counter()
    by_family: dict[int, Counter[str]] = {}
    base = 0
    total = 0
    for p in site.rglob("*"):
        if not p.is_file():
            continue
        size = p.stat().st_size
        total += size
        if rel_root not in p.parents:
            base += size
            continue
        rel = p.relative_to(rel_root).as_posix()
        for rx in SEASON_FILE:
            m = rx.match(rel)
            if m:
                s = int(m.group(1))
                by_season[s] += size
                by_family.setdefault(s, Counter())[rel.split("/")[0]] += size
                break
        else:
            if PLAYER_PAGE.match(rel):
                doc = json.loads(p.read_bytes())
                seasons = doc.get("seasons", [])
                used = 0
                for line in seasons:
                    b = len(json.dumps(line, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode()) + 1
                    by_season[line["season"]] += b
                    by_family.setdefault(line["season"], Counter())["players"] += b
                    used += b
                base += size - used
            else:
                base += size
    return {
        "site": str(site),
        "release": rel_root.name,
        "total_bytes": total,
        "base_bytes": base,
        "seasons": {
            str(s): {"bytes": by_season[s], "families": dict(sorted(by_family[s].items()))} for s in sorted(by_season)
        },
    }


def main() -> None:
    site = Path(sys.argv[1])
    out = attribute(site)
    mib = 2**20
    seasons = {int(k): v["bytes"] for k, v in out["seasons"].items()}  # type: ignore[union-attr]
    for s in sorted(seasons):
        if s >= 2010 or s % 10 == 0:
            print(f"{s}: {seasons[s] / mib:6.2f} MiB")
    print(f"base {out['base_bytes'] / mib:.1f} MiB, total {out['total_bytes'] / mib:.1f} MiB")  # type: ignore[operator]
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
