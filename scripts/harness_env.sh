# shellcheck shell=bash
# Shared interpreter and agent-CLI resolution for the weekly harness and its hooks (sourced, never executed).
#
# The harness used to hard-code /home/abhi/sourceCode/python/coding/.venv/bin/python and
# /home/abhi/.claude/local/claude. Both vanished in an OS upgrade (2026-10), so the cycle could not start. The
# interpreter is now the repository's own locked environment; build it once with
#     uv sync --locked --group dev --group legacy --extra ml
# Override with SUPERCOACH_PYTHON (interpreter) or CLAUDE (agent CLI). Requires REPO_ROOT to be set.

harness_python() {
  local py="${SUPERCOACH_PYTHON:-$REPO_ROOT/.venv/bin/python}"
  if [ ! -x "$py" ]; then
    echo "FATAL: no Python interpreter at $py." >&2
    echo "  Build the repository environment:  (cd \"$REPO_ROOT\" && uv sync --locked --group dev --group legacy --extra ml)" >&2
    echo "  or point SUPERCOACH_PYTHON at an interpreter that has it." >&2
    return 1
  fi
  printf '%s\n' "$py"
}

harness_claude() {
  local c="${CLAUDE:-$(command -v claude || true)}"
  if [ -z "$c" ] || [ ! -x "$c" ]; then
    echo "FATAL: the claude CLI is not on PATH (set CLAUDE to its path)." >&2
    return 1
  fi
  printf '%s\n' "$c"
}
