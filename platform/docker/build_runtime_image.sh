#!/usr/bin/env bash
# Build the platform runtime image used by experiments.
#
# 1. (once) Build the base image from the upstream Dockerfile inside a
#    prepared experiment workspace (never from the repo root — there is no
#    .dockerignore and the context would include platform/):
#      WS=<workspace>; cd "$WS" && docker build --network=host -t hyperagents-buildtest .
#    (The proofgrader_repo prerequisite must exist: see
#     platform/hacp/workspace.py + domains/imo/setup_proofgrader_repo.py)
# 2. (this script) Add the runtime layer and tag as 'hyperagents' — the name
#    upstream build_container() looks for, so it skips rebuilding.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
if ! docker image inspect hyperagents-buildtest >/dev/null 2>&1; then
  echo "ERROR: base image 'hyperagents-buildtest' not found." >&2
  echo "Build it first from a prepared experiment workspace (see header comment)." >&2
  exit 1
fi
docker build --network=host -f "$HERE/Dockerfile.runtime" -t hyperagents "$HERE"
echo "runtime image ready: hyperagents (from hyperagents-buildtest + git runtime layer)"
