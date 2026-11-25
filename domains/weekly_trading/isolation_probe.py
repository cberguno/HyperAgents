# Deliberately hostile synthetic probe, not generated code.
import os, socket, json, pathlib
failures=[]
if os.environ.get('OPENROUTER_API_KEY'): failures.append('environment secret')
for p in ['/host-canary/secret','/home/runner/.git-credentials','/score-data/canary','/var/run/docker.sock','/run/secrets/openrouter','/github/home/.git-credentials']:
    if pathlib.Path(p).exists(): failures.append('host or secret path:'+p)
try:
    with socket.create_connection(('1.1.1.1',443),timeout=2): failures.append('network access')
except OSError: pass
try:
    pathlib.Path('/escape_write').write_text('bad'); failures.append('filesystem writable')
except OSError: pass
try:
    pathlib.Path('/input/escape_write').write_text('bad'); failures.append('input writable')
except OSError: pass
try:
    os.system('git push >/dev/null 2>&1') == 0 and failures.append('git push succeeded')
except Exception: pass
if failures: raise SystemExit(json.dumps(failures))
print(json.dumps([0,1]))
