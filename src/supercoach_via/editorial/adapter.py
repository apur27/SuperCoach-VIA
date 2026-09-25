"""Optional editorial adapter interface (PLAN 10; AUDIT S01).

The adapter receives a bounded :class:`EvidencePacket` and returns structured draft
Markdown (plus an optional review verdict). It has no shell, Git, network or publish
capability through this interface; the numeric release never depends on it.

:class:`FakeAdapter` is deterministic and used in offline tests.

:class:`SubprocessAdapter` is the documented path for a real text-only model CLI:

- ``argv`` is an argument *array* (never a shell string); ``bypassPermissions`` and
  ``--dangerously-skip-permissions`` are refused outright;
- runs in a fresh dedicated temp workspace (``cwd`` and ``HOME``) containing only
  ``packet.json``; the prompt is sent on stdin;
- the environment is rebuilt from an allowlist (``PATH``, ``LANG``, ``LC_ALL``) plus
  explicitly passed variables, so ambient secrets/tokens never reach the child;
- enforces a wall-clock timeout and a stdout byte cap; a non-zero exit, timeout,
  oversize or malformed JSON raises :class:`AdapterError`.

Expected stdout: ``{"markdown": "...", "claim_refs": [...]}``. The operator chooses a CLI
invocation whose own permission mode actually enforces read-only/no-tools behaviour
(e.g. a restricted text-only mode); nothing here invokes an AI during tests.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from supercoach_via.domain.schemas import ReviewVerdict
from supercoach_via.editorial.evidence import EvidencePacket

_FORBIDDEN_ARGS = ("bypasspermissions", "--dangerously-skip-permissions", "--dangerously")
_ENV_ALLOW = ("PATH", "LANG", "LC_ALL")


class AdapterError(RuntimeError):
    """The editorial adapter failed; the caller records an unpublished draft."""


class DraftResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    markdown: str = Field(max_length=200_000)
    claim_refs: tuple[str, ...] = ()
    adapter: str = "unknown"


class ReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: ReviewVerdict
    reasons: tuple[str, ...] = ()
    reviewer: str = "unknown"


class EditorialAdapter(Protocol):
    name: str

    def draft(self, packet: EvidencePacket) -> DraftResult: ...

    def review(self, packet: EvidencePacket, draft: DraftResult) -> ReviewResult: ...


class FakeAdapter:
    """Deterministic offline adapter: one sentence per claim, via claim references."""

    name = "fake"

    def __init__(self, *, verdict: ReviewVerdict = ReviewVerdict.PASS, extra_text: str = "") -> None:
        self.verdict = verdict
        self.extra_text = extra_text

    def draft(self, packet: EvidencePacket) -> DraftResult:
        lines = [f"- {c.label}: {{{{claim:{c.claim_id}}}}} {c.unit}" for c in packet.claims]
        if self.extra_text:
            lines.append(self.extra_text)
        return DraftResult(markdown="\n".join(lines), claim_refs=tuple(c.claim_id for c in packet.claims),
                           adapter=self.name)  # fmt: skip

    def review(self, packet: EvidencePacket, draft: DraftResult) -> ReviewResult:
        return ReviewResult(verdict=self.verdict, reasons=("fake reviewer",), reviewer=self.name)


class SubprocessAdapter:
    name = "subprocess"

    def __init__(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float = 120.0,
        max_output_bytes: int = 256 * 1024,
        extra_env: Mapping[str, str] | None = None,
        reviewer_argv: Sequence[str] | None = None,
    ) -> None:
        if isinstance(argv, str) or not argv or not all(isinstance(a, str) for a in argv):
            raise ValueError("argv must be a non-empty argument array, not a shell string")
        for a in (*argv, *(reviewer_argv or ())):
            if any(bad in a.lower() for bad in _FORBIDDEN_ARGS):
                raise ValueError(f"refusing permission-bypass argument {a!r}")
        self.argv = list(argv)
        self.reviewer_argv = list(reviewer_argv) if reviewer_argv else None
        self.timeout_s = timeout_s
        self.max_output_bytes = max_output_bytes
        self.extra_env = dict(extra_env or {})

    def _invoke(self, argv: list[str], packet: EvidencePacket, prompt: str) -> dict[str, object]:
        workspace = Path(tempfile.mkdtemp(prefix="scvia-editorial-"))
        try:
            (workspace / "packet.json").write_bytes(packet.to_json())
            env = {k: os.environ[k] for k in _ENV_ALLOW if k in os.environ}
            env.update(self.extra_env)
            env["HOME"] = str(workspace)
            env["TMPDIR"] = str(workspace)
            try:
                proc = subprocess.run(  # noqa: S603 - argument array, validated argv, no shell
                    argv,
                    cwd=workspace,
                    input=prompt.encode(),
                    capture_output=True,
                    env=env,
                    timeout=self.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise AdapterError(f"editorial adapter timed out after {self.timeout_s}s") from exc
            except OSError as exc:
                raise AdapterError(f"editorial adapter could not start: {exc}") from exc
            if len(proc.stdout) > self.max_output_bytes:
                raise AdapterError("editorial adapter output exceeded the byte budget")
            if proc.returncode != 0:
                raise AdapterError(f"editorial adapter exited {proc.returncode}")
            try:
                data = json.loads(proc.stdout.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise AdapterError("editorial adapter output is not JSON") from exc
            if not isinstance(data, dict):
                raise AdapterError("editorial adapter output is not a JSON object")
            return data
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def draft(self, packet: EvidencePacket) -> DraftResult:
        prompt = f"{packet.instructions_boundary}\nTask: {packet.task}\nEvidence is in ./packet.json."
        data = self._invoke(self.argv, packet, prompt)
        md = data.get("markdown")
        refs = data.get("claim_refs", [])
        if not isinstance(md, str) or not isinstance(refs, list):
            raise AdapterError("editorial adapter output lacks markdown/claim_refs")
        return DraftResult(markdown=md, claim_refs=tuple(str(r) for r in refs), adapter=self.name)

    def review(self, packet: EvidencePacket, draft: DraftResult) -> ReviewResult:
        if self.reviewer_argv is None:
            return ReviewResult(verdict=ReviewVerdict.UNKNOWN, reasons=("no reviewer configured",), reviewer=self.name)
        prompt = f"Review this draft against ./packet.json. Reply JSON verdict.\n\n{draft.markdown}"
        data = self._invoke(self.reviewer_argv, packet, prompt)
        try:
            verdict = ReviewVerdict(str(data.get("verdict")))
        except ValueError:
            verdict = ReviewVerdict.UNKNOWN
        reasons = data.get("reasons", [])
        return ReviewResult(verdict=verdict, reasons=tuple(str(r)[:300] for r in reasons)[:20]
                            if isinstance(reasons, list) else (), reviewer=self.name)  # fmt: skip
