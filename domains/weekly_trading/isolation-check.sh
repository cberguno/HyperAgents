#!/bin/bash
set -euo pipefail
D="$RUNNER_TEMP/weeklytest"; mkdir -p "$D/checkout" "$D/host-secret" "$D/host-score" "$D/remotes"
K="$D/host-secret/key"; S="$D/host-score/data"; echo synthetic-key > "$K"; echo synthetic-score > "$S"
git init -q "$D/checkout"; git -C "$D/checkout" -c user.name=Test -c user.email=test@example.invalid commit -qm init --allow-empty
git init -q --bare "$D/remotes/target.git"; git -C "$D/remotes/target.git" config daemon.receivepack true
git daemon --reuseaddr --export-all --enable=receive-pack --base-path="$D/remotes" --listen=127.0.0.1 --port=19418 "$D/remotes" > "$D/git.log" 2>&1 & G=$!
python3 -m http.server 19419 --bind 127.0.0.1 --directory "$D/host-secret" > "$D/http.log" 2>&1 & H=$!
trap 'kill "$G" "$H" 2>/dev/null || true' EXIT
python3 - <<'PY'
import socket,time
for port in (19418,19419):
 for _ in range(50):
  try:
   with socket.create_connection(('127.0.0.1',port),timeout=1): break
  except OSError: time.sleep(.1)
 else: raise SystemExit('test server failed to start')
PY
cat > "$D/Dockerfile" <<'EOF'
FROM python:3.12-alpine
RUN apk add --no-cache git
EOF
docker build -q -t weekly-security-test "$D" >/dev/null
# This single launch function is the proposed candidate container configuration.
launch() {
  local mode="$1"; shift
  local opts=(--network none)
  if [ "$mode" = negative ]; then opts=(--network host -v "$D:$D:ro" -e OPENROUTER_API_KEY=synthetic); fi
  docker run --rm "${opts[@]}" --read-only --cap-drop ALL --security-opt no-new-privileges --pids-limit 32 --memory 128m --cpus 1 --user 65534:65534 -v "$D/checkout:/repo:ro" -v "$GITHUB_WORKSPACE/domains/weekly_trading/isolation_probe.py:/probe.py:ro" weekly-security-test python -I /probe.py "$K" "$S"
}
launch negative > "$D/negative.json"
launch constrained > "$D/constrained.json"
python3 - "$D/negative.json" "$D/constrained.json" <<'PY'
import json,sys
n,p=(json.load(open(f)) for f in sys.argv[1:])
print('negative control:',json.dumps(n,sort_keys=True)); print('constrained:',json.dumps(p,sort_keys=True))
for k in ('key_file','score_file','key_env','key_network','git_server','push_succeeded'):
 if not n[k]:raise SystemExit('negative control failed to demonstrate '+k)
 if p[k]:raise SystemExit('constrained candidate escaped via '+k)
for k in ('checkout_writable','root_writable','docker_socket'):
 if p[k]:raise SystemExit('constrained candidate escaped via '+k)
print('PASS: negative control exercised the routes and constrained probe did not reach them')
PY
