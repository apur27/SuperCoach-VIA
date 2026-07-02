#!/usr/bin/env bash
#
# council-content-hash.sh — canonical content hash of a council doc.
#
# Prints sha256 over the file with the VOLATILE provenance lines removed:
#   - the `<!-- council-pipeline: ... -->` stamp (added by Gaffer on SHIP), and
#   - the trust-badge line (contains `council-pipeline-gated`, added at ship).
#
# Removing these makes the hash STABLE across the verify->stamp->ship transition:
# the hash DataSentinel computes when it verifies the full doc equals the hash the
# pre-commit hook recomputes after Gaffer has stamped and badged it.
#
# This is the SINGLE source of truth for the hash. Both the producer
# (scripts/record-sentinel-verdict.sh, called by DataSentinel) and the consumer
# (scripts/check-council-stamp.sh) MUST use this script so their hashes agree.
#
# Usage: scripts/council-content-hash.sh <file.md>   -> prints 64-hex sha256
#
set -euo pipefail

f="${1:?usage: council-content-hash.sh <file>}"
[ -f "$f" ] || { echo "council-content-hash.sh: no such file: $f" >&2; exit 2; }

# 1. drop the volatile provenance lines (stamp + trust badge);
# 2. drop trailing blank lines, so the blank separator that precedes a stamp does
#    not change the hash between the pre-stamp form (what DataSentinel records) and
#    the post-stamp form (what the pre-commit hook recomputes).
grep -vE '<!-- council-pipeline:|council-pipeline-gated' "$f" \
  | awk '{ a[NR] = $0 }
         END { last = NR
               while (last > 0 && a[last] ~ /^[ \t]*$/) last--
               for (i = 1; i <= last; i++) print a[i] }' \
  | sha256sum | awk '{print $1}'
