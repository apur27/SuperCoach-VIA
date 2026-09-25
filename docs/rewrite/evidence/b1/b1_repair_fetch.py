"""B1 bounded repair fetch (owner-authorised 2026-09-25, at most 10 AFLTables requests).

Run once from the repository root with network access:

    uv run --locked python docs/rewrite/evidence/b1/b1_repair_fetch.py <data_root> <snapshot_id> [--reuse MANIFEST]

``<data_root>``/``<snapshot_id>`` name an unpromoted legacy-import candidate. The script
discovers each player's page from the source match page of a lineup row that could not be
linked, fetches season page + match pages + player pages through the rate-limited
HttpClient with ONE attempt per URL (so requests <= planned URLs <= 10), and writes, next to
this file: ``fetch-manifest.json`` (URLs, status, fetched-at, sha256, bytes, request counts,
per-player evidence), ``raw/<sha256>.html.gz`` (every payload), and ``rows.jsonl`` (the
derived upsert rows, each carrying its source URL + sha256). It never promotes anything:
applying the rows is ``scvia apply-repair`` (offline, re-verifies hashes and re-parses).

``--reuse MANIFEST`` serves every URL that an earlier authorised run already fetched
successfully from that run's archived bytes (no network); their observations keep the
earlier run's fetched-at. Only the remaining URLs reach the network, and the request cap
applies to the SUM of network requests across runs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import gzip
import json
import sys
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path

import httpx

from supercoach_via.domain.ids import ClubRegistry
from supercoach_via.ingest import refresh as rf
from supercoach_via.ingest.http import HttpClient, RawArchive, load_policies
from supercoach_via.settings import RunContext, Settings
from supercoach_via.storage.queries import SnapshotQuery
from supercoach_via.storage.snapshots import load_snapshot

HERE = Path(__file__).resolve().parent
MAX_REQUESTS = 10
USER_AGENT = "SuperCoach-VIA rewrite repair (owner-authorised bounded fetch; github.com/apur27/SuperCoach-VIA)"

#: name token -> (club source name, existing identity or None, proving DOB, excluded namesake DOBs)
TARGETS = {
    "Flynn Perez": ("Hawthorn", "legacy:perez_flynn_25082001", date(2001, 8, 25), ()),
    "Jack Dalton": ("Hawthorn", None, None, (date(1876, 4, 15),)),
    "Will Brodie": ("Port Adelaide", "legacy:brodie_will_23081998", date(1998, 8, 23), ()),
}


def _discovery_matches(data_root: Path, snapshot_id: str) -> dict[str, str]:
    """name token -> a 2026 match id where that token was quarantined (linked via siblings)."""
    manifest = load_snapshot(data_root, snapshot_id)
    out: dict[str, str] = {}
    with SnapshotQuery(data_root, manifest, tables={"quarantine", "lineups"}) as q:
        for token, match_id in q.rows(
            "SELECT json_extract_string(q.raw, '$.token') AS token, min(l.match_id) FROM quarantine q "
            "JOIN lineups l ON l.source_path = q.source_path AND l.source_row = q.source_row "
            "WHERE q.season = 2026 AND q.table_name = 'lineups' AND q.reason = 'lineup_token_unresolved' "
            "GROUP BY 1"
        ):
            out[str(token)] = str(match_id)
    return out


class ReuseTransport(httpx.BaseTransport):
    """Serve earlier-run payloads locally; delegate everything else to the network."""

    def __init__(self, reuse: dict[str, bytes]) -> None:
        self.reuse = reuse
        self.network = httpx.HTTPTransport()
        self.network_requests: Counter[str] = Counter()
        self.served: list[str] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url in self.reuse:
            self.served.append(url)
            return httpx.Response(200, content=self.reuse[url], request=request)
        self.network_requests[request.url.host] += 1
        return self.network.handle_request(request)


def _load_reuse(manifest_path: Path) -> tuple[dict[str, bytes], dict[str, dict[str, object]], int]:
    prior = json.loads(manifest_path.read_text(encoding="utf-8"))
    reuse: dict[str, bytes] = {}
    obs: dict[str, dict[str, object]] = {}
    for o in prior["source_observations"]:
        sha = str(o.get("content_sha256") or "")
        if o.get("http_status") != 200 or not sha:
            continue
        body = gzip.decompress((HERE / "raw" / f"{sha}.html.gz").read_bytes())


        if hashlib.sha256(body).hexdigest() != sha:
            raise SystemExit(f"archived payload for {o['url']} does not match its sha256")
        reuse[str(o["url"])] = body
        obs[str(o["url"])] = o
    return reuse, obs, int(prior["requests_total"])


def main(data_root: Path, snapshot_id: str, reuse_manifest: Path | None = None) -> int:
    manifest = load_snapshot(data_root, snapshot_id)
    with SnapshotQuery(data_root, manifest, tables={"players"}) as q:
        existing = {
            r["player_id"]: r
            for r in q.arrow(
                "SELECT * FROM players WHERE player_id IN (?, ?)",
                ["legacy:perez_flynn_25082001", "legacy:brodie_will_23081998"],
            ).to_pylist()
        }
    discover = _discovery_matches(data_root, snapshot_id)
    targets = [
        rf.PlayerRepairTarget(name, discover[name], club, pid, dob, excl)
        for name, (club, pid, dob, excl) in TARGETS.items()
    ]
    base = rf.base_state_from_snapshot(data_root, snapshot_id)
    policies = load_policies(Path("config/source_policies.toml"))
    at_policy = dataclasses.replace(policies.sources["afltables"], max_attempts=1, requests_per_second=1.0)
    policies = dataclasses.replace(policies, sources={**policies.sources, "afltables": at_policy})
    archive_root = data_root / "raw"
    clubs = ClubRegistry.from_csv(Path("config/team_aliases.csv"))
    started = datetime.now(UTC)
    reuse, prior_obs, prior_requests = _load_reuse(reuse_manifest) if reuse_manifest else ({}, {}, 0)
    transport = ReuseTransport(reuse)
    with HttpClient(policies, user_agent=USER_AGENT, archive=RawArchive(archive_root), transport=transport) as http:
        ctx = RunContext(settings=Settings(data_root=data_root, season=2026))
        ctx.http = http
        res = rf.repair_player_pages(
            base, 2026, targets, ctx, club_resolver=clubs.resolve, existing_players=existing,
            max_requests=MAX_REQUESTS,
        )  # fmt: skip
        observations = list(http.observations)
    network = sum(transport.network_requests.values())
    total = prior_requests + network
    assert total <= MAX_REQUESTS, f"request cap exceeded: {total}"
    # observations for locally re-served payloads keep the earlier run's real fetch record
    res.upserts["source_observations"] = [
        prior_obs.get(str(o["url"]), o) if str(o["url"]) in transport.served else o
        for o in res.upserts["source_observations"]
    ]
    (HERE / "raw").mkdir(exist_ok=True)
    archive = RawArchive(archive_root)
    for obs in res.upserts["source_observations"]:
        sha = obs.get("content_sha256")
        body = archive.get(sha) if sha else None
        if body is not None:
            (HERE / "raw" / f"{sha}.html.gz").write_bytes(gzip.compress(body, mtime=0))
    fact_rows = [{"table": t, "row": r} for t in ("player_games", "players") for r in res.upserts[t]]
    with (HERE / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for rec in fact_rows:
            fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
    report = {
        "authorised": "owner decision on blocker B1, 2026-09-25: <=10 afltables requests",
        "base_snapshot_id": snapshot_id,
        "started_at": started.isoformat(),
        "outcome": res.outcome.value,
        "exit_code": res.exit_code,
        "requests_total": total,
        "network_requests_this_run": dict(transport.network_requests),
        "requests_prior_runs": prior_requests,
        "reused_from_prior_run": sorted(transport.served),
        "client_request_counts": res.request_counts,
        "bytes_received": res.bytes_received,
        "max_attempts_per_url": 1,
        "targets": [dataclasses.asdict(t) for t in targets],
        "evidence": res.evidence,
        "source_observations": res.upserts["source_observations"],
        "http_observations": observations,
        "work_log": res.work_log,
        "issues": res.issues,
        "rows_by_table": {t: len(res.upserts[t]) for t in ("player_games", "players")},
    }
    (HERE / "fetch-manifest.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: report[k] for k in ("outcome", "requests_total", "rows_by_table")}, default=str))
    for ev in res.evidence:
        print(json.dumps(ev, default=str))
    return res.exit_code


if __name__ == "__main__":
    args = sys.argv[1:]
    reuse_path = Path(args[args.index("--reuse") + 1]) if "--reuse" in args else None
    sys.exit(main(Path(args[0]), args[1], reuse_path))
