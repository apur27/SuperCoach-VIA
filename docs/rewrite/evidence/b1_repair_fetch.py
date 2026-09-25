"""B1 bounded repair fetch (owner-authorized: at most 10 afltables requests).

Fetches, through the rewrite's policy-enforcing HttpClient with retries disabled:
  1. the 2026 season page (fixture for resolving player-page rows to real match dates),
  2. one Hawthorn 2026 match page on which Jack Dalton played (to discover his real
     player-page URL instead of guessing a name-derived one),
  3. the three player pages.
Raw payloads go to the content-addressed archive under var/raw; parsed results and
request accounting go to docs/rewrite/evidence/b1-repair-sources.json. No CSV is
written here; see b1_apply_rows.py.

Run: uv run --locked python docs/rewrite/evidence/b1_repair_fetch.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date
from pathlib import Path

import httpx

from supercoach_via.ingest import afltables as at
from supercoach_via.ingest.http import HttpClient, RawArchive, parse_policies

ROOT = Path(__file__).resolve().parents[3]
BUDGET = 6  # hard stop, below the authorized 10; retries disabled so fetches == requests
OUT = ROOT / "docs/rewrite/evidence/b1-repair-sources.json"


def main() -> None:
    text = (ROOT / "config/source_policies.toml").read_text()
    text = re.sub(r"(?m)^max_attempts = \d+$", "max_attempts = 1", text)
    policies = parse_policies(text)
    proxy = os.environ.get("HTTPS_PROXY")
    ca = os.environ.get("SSL_CERT_FILE") or "/root/.ccr/ca-bundle.crt"
    transport = httpx.HTTPTransport(proxy=proxy, verify=ca) if proxy else None
    record: dict[str, object] = {"budget": BUDGET, "authorized": 10, "fetches": []}
    with HttpClient(
        policies,
        user_agent="supercoach-via-rewrite/0.1 (B1 bounded repair; owner-authorized)",
        archive=RawArchive(ROOT / "var/raw"),
        transport=transport,
    ) as client:

        cache = ROOT / "var/raw/b1-cache"
        cache.mkdir(parents=True, exist_ok=True)

        def get(url: str) -> bytes:
            # A rerun reuses payloads already fetched so it never spends extra requests.
            cached = cache / (hashlib.sha256(url.encode()).hexdigest() + ".html")
            if cached.exists():
                record["fetches"].append({"url": url, "reused_from": str(cached.relative_to(ROOT))})  # type: ignore[union-attr]
                return cached.read_bytes()
            used = sum(client.request_counts.values())
            if used >= BUDGET:
                raise SystemExit(f"request budget exhausted ({used})")
            res = client.fetch(url, conditional=False)
            record["fetches"].append(  # type: ignore[union-attr]
                {"url": url, "status": res.http_status, "outcome": res.outcome.value, "sha256": res.sha256,
                 "bytes": res.bytes, "requests": res.requests, "last_modified": res.last_modified,
                 "fetched_at": res.fetched_at.isoformat(), "error": res.error}
            )
            if not res.ok or res.content is None:
                OUT.write_text(json.dumps(record, indent=2, default=str) + "\n")
                raise SystemExit(f"fetch failed: {url}: {res.error}")
            cached.write_bytes(res.content)
            return res.content

        fixture = at.parse_season_page(get(at.season_url(2026)), season=2026)
        record["fixture"] = {"outcome": fixture.outcome.value, "completed": len(fixture.completed())}

        # Hawthorn v Essendon/whatever in round 5: any 2026 Hawthorn match with Dalton in the lineup.
        r5 = [m for m in fixture.matches if m.stage.code == "5" and "Hawthorn" in m.team_pair and m.detail_url]
        if len(r5) != 1:
            raise SystemExit(f"expected one Hawthorn R5 match, got {len(r5)}")
        m = r5[0]
        detail = at.parse_match_detail(get(m.detail_url), season=2026, game_id=m.source_game_id or "")
        dalton = [p for p in detail.players if p.team == "Hawthorn" and "Dalton" in p.source_name]
        record["dalton_discovery"] = {"match": m.detail_url, "rows": [p.source_name for p in dalton],
                                      "urls": [p.player_url for p in dalton]}
        if len(dalton) != 1 or dalton[0].player_url is None:
            raise SystemExit("could not uniquely discover Jack Dalton's player URL")

        urls = {
            "perez_flynn": at.player_url("F/Flynn_Perez"),
            "brodie_will": at.player_url("W/Will_Brodie"),
            "dalton_jack_2026": dalton[0].player_url,
        }
        players: dict[str, object] = {}
        for key, url in urls.items():
            page = at.parse_player_page(get(url), page_url=url)
            resolved = at.resolve_player_games([g for g in page.games if g.season == 2026], fixture)
            players[key] = {
                "url": url, "name": page.name, "birth_date": page.birth_date, "outcome": page.outcome.value,
                "issues": page.issues, "career_games": len(page.games),
                "seasons": sorted({g.season for g in page.games}),
                "games_2026": [
                    {"team": r.game.team, "opponent": r.game.opponent, "round": r.game.round_token,
                     "result": r.game.result, "jersey": r.game.jersey_token,
                     "counter": r.game.career_game_counter_token, "match_id": r.match_id,
                     "stage_label": r.stage_label, "date": r.match_date, "date_quality": r.date_quality.value,
                     "stats": r.game.stats}
                    for r in resolved
                ],
            }
        record["players"] = players
        record["request_counts"] = client.request_counts
        record["bytes_received"] = client.bytes_received
    OUT.write_text(json.dumps(record, indent=2, default=lambda o: o.isoformat() if isinstance(o, date) else str(o)) + "\n")
    print(json.dumps(record["request_counts"]))


if __name__ == "__main__":
    main()
