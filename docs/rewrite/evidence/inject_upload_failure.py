"""Fail a copy before activation and leave the live release untouched.

Usage: python inject_upload_failure.py HOST OUTPUT_ROOT RELEASE_ID
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from supercoach_via.publish.release import LocalDirectoryDestination, PublishError, publish_release
from supercoach_via.settings import utc_now


def main() -> None:
    host, output_root, release = sys.argv[1:4]
    dest_root = Path(host)
    before = LocalDirectoryDestination("local", dest_root).active_release()
    marker = dest_root / "live" / "index.html"
    before_bytes = marker.read_bytes() if marker.is_file() else None

    class Boom(LocalDirectoryDestination):
        def upload(self, release_dir: Path) -> None:
            raise PublishError("injected copy failure")

    try:
        publish_release(Path(output_root) / "releases" / release, Boom("local", dest_root), clock=utc_now)
    except PublishError as exc:
        if "injected copy failure" not in str(exc):
            raise SystemExit(f"unexpected publish error: {exc}") from exc
    else:
        raise SystemExit("injected copy failure was not raised")
    after = LocalDirectoryDestination("local", dest_root).active_release()
    if after != before:
        raise SystemExit(f"live pointer moved from {before} to {after}")
    after_bytes = marker.read_bytes() if marker.is_file() else None
    if after_bytes != before_bytes:
        raise SystemExit("live bytes changed during the injected failure")
    print(json.dumps({"live": after, "injected": release, "bytes_unchanged": True}))


if __name__ == "__main__":
    main()
