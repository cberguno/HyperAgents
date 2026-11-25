#!/bin/bash
set -euo pipefail
D="$RUNNER_TEMP/socket-check"; mkdir -p "$D/channel" "$D/scratch" "$D/secret"; chmod 777 "$D/channel" "$D/scratch"; echo synthetic > "$D/secret/key"
git init -q "$D/scratch"
cat > "$D/Dockerfile" <<'EOF'
FROM python:3.12-alpine
RUN apk add --no-cache git
EOF
docker build -q -t weekly-socket-test "$D" >/dev/null
cleanup(){ docker rm -f weekly-proxy-test >/dev/null 2>&1 || true; }; trap cleanup EXIT
docker run -d --name weekly-proxy-test --read-only --network none --cap-drop ALL --security-opt no-new-privileges -v "$D/channel:/channel" -v "$GITHUB_WORKSPACE/domains/weekly_trading/socket_test/proxy.py:/proxy.py:ro" weekly-socket-test python -I /proxy.py >/dev/null
for i in $(seq 1 50); do [ -S "$D/channel/completion.sock" ] && break; sleep .1; done
[ -S "$D/channel/completion.sock" ]
docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --user 65534:65534 -v "$D/channel:/channel:ro" -v "$D/scratch:/scratch" -v "$GITHUB_WORKSPACE/domains/weekly_trading/socket_test/probe.py:/probe.py:ro" weekly-socket-test python -I /probe.py
# Negative control: intentionally expose secret and require the hostile read to succeed.
docker run --rm --network none --read-only --user 65534:65534 -v "$D/secret:/host-secret:ro" weekly-socket-test python -c 'from pathlib import Path; assert Path("/host-secret/key").read_text().strip()=="synthetic"; print("negative control caught: secret readable")'
