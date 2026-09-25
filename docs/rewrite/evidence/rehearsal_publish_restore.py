"""Local publication, rollback and backup/restore rehearsal (PLAN Phase 8; nothing leaves the box).

Usage:
  uv run --locked python docs/rewrite/evidence/rehearsal_publish_restore.py \
      --output-root dist/real --data-root var/real --work <scratch-dir> \
      --first <validated release id> --second <validated release id> --unvalidated <failed release id>

Publication goes to a LocalDirectoryDestination under <work>/host (the stand-in for a static
host); `scvia publish` itself is invoked, never a real host. Failure injection replaces the
destination's upload with an OSError (disk-full stand-in) through the library. Writes
<work>/rehearsal-publish-restore.json and prints it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from supercoach_via.publish.release import LocalDirectoryDestination, PublishError, publish_release

REPO = Path(__file__).resolve().parents[3]


def scvia(*args: str) -> dict[str, Any]:
    # fixed argument array (no shell); uv is resolved from PATH like any operator command
    r = subprocess.run(["uv", "run", "--locked", "scvia", *args, "--json"], cwd=REPO,  # noqa: S603, S607
                       capture_output=True, text=True, timeout=1800, check=False)  # fmt: skip
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.startswith("{")]
    payload = json.loads(lines[-1]) if lines else {"stdout": r.stdout[-400:], "stderr": r.stderr[-400:]}
    return {"exit_code": r.returncode, "result": payload}


def live(host: Path) -> str | None:
    link = host / "live"
    return link.resolve().name if link.is_symlink() else None


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


class FailingUpload(LocalDirectoryDestination):
    def upload(self, release_dir: Path) -> None:
        raise OSError(28, "No space left on device (injected)")


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("--output-root", "--data-root", "--work", "--first", "--second", "--unvalidated"):
        ap.add_argument(name, required=True)
    a = ap.parse_args()
    out_root, data_root, work = Path(a.output_root), Path(a.data_root), Path(a.work)
    host = work / "host"
    if work.exists():
        shutil.rmtree(work)
    host.mkdir(parents=True)
    steps: list[dict[str, Any]] = []

    def step(name: str, res: dict[str, Any], expect_ok: bool) -> None:
        ok = (res.get("exit_code") == 0) == expect_ok
        steps.append({"step": name, "expected_success": expect_ok, "as_expected": ok, "live_after": live(host), **res})

    pub = ("--destination", str(host), "--output-root", str(out_root))
    step("publish first", scvia("publish", "--release", a.first, *pub), True)
    step("publish second", scvia("publish", "--release", a.second, *pub), True)
    step("publish an unvalidated (FAIL) release is refused", scvia("publish", "--release", a.unvalidated, *pub), False)

    before = live(host)
    try:
        publish_release(out_root / "releases" / a.first, FailingUpload("host", host), clock=lambda: datetime.now(UTC))
        injected: dict[str, Any] = {"exit_code": 0, "result": "unexpectedly published"}
    except PublishError as exc:
        injected = {"exit_code": 7, "result": str(exc)}
    step("injected upload failure keeps the previous release live", injected, False)
    steps[-1]["as_expected"] = steps[-1]["as_expected"] and live(host) == before

    step("rollback to first", scvia("rollback", "--release", a.first, *pub), True)
    steps[-1]["as_expected"] = steps[-1]["as_expected"] and live(host) == a.first

    receipts = sorted(p.name for p in (out_root / "receipts").glob("*.json"))

    # Backup -> restore into a fresh root -> validate -> reuse the cached model bundle.
    keep = ["current.json", "snapshots", "fragments", "models", "predictions", "evaluations"]
    backup = work / "var-backup.tar"
    with tarfile.open(backup, "w") as tf:
        for k in keep:
            if (data_root / k).exists():
                tf.add(data_root / k, arcname=k)
    restored = work / "restored" / "var"
    restored.mkdir(parents=True)
    with tarfile.open(backup) as tf:
        tf.extractall(restored, filter="data")
    def digest(p: Path) -> str:
        return tree_digest(p) if p.is_dir() else hashlib.sha256(p.read_bytes()).hexdigest()

    digests = {k: (digest(data_root / k), digest(restored / k)) for k in keep if (data_root / k).exists()}
    same = all(x == y for x, y in digests.values())
    step("restored bytes equal the backed-up bytes",
         {"exit_code": 0 if same else 1, "result": {k: v[0][:16] for k, v in digests.items()}}, True)  # fmt: skip
    step("validate restored current snapshot", scvia("validate", "--snapshot", "current", "--data-root", str(restored)), True)
    step("status on restored root", scvia("status", "--data-root", str(restored)), True)
    fc = scvia("forecast", "--train-cutoff", "2025-01-01", "--calibration-end", "2026-01-01",
               "--cutoff", "2026-09-25T10:00:00+00:00", "--data-root", str(restored))  # fmt: skip
    step("forecast on restored root reuses the restored bundle (no retrain)", fc, True)
    t = fc.get("result", {}).get("timings", {})
    steps[-1]["as_expected"] = steps[-1]["as_expected"] and float(t.get("train", 1e9)) < 60

    report = {"at": datetime.now(UTC).isoformat(), "host": str(host), "receipts": receipts,
              "all_as_expected": all(s["as_expected"] for s in steps), "steps": steps}  # fmt: skip
    (work / "rehearsal-publish-restore.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps({"all_as_expected": report["all_as_expected"],
                      "steps": [(s["step"], s["as_expected"], s["live_after"]) for s in steps]}, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
