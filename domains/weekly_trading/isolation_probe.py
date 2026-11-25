# Adversarial test only: all secrets and remotes are synthetic.
import json,os, pathlib,socket,subprocess,sys,urllib.request
key,score=sys.argv[1:3]
def readable(p):
 try:
  pathlib.Path(p).read_bytes();return True
 except OSError:return False
def endpoint(port):
 try:
  with socket.create_connection(('127.0.0.1',port),timeout=2): return True
 except OSError:return False
assert subprocess.run(['git','--version'],capture_output=True).returncode==0
# Use a real committed checkout and real receive-pack server, not a missing binary.
p=subprocess.run(['git','-c','safe.directory=/repo','-C','/repo','push','git://127.0.0.1:19418/target.git','HEAD:refs/heads/probe'],capture_output=True,text=True,timeout=10,env={**os.environ,'GIT_TERMINAL_PROMPT':'0'})
try:
 pathlib.Path('/repo/probe-write').write_text('test'); checkout_writable=True
except OSError: checkout_writable=False
try:
 pathlib.Path('/probe-write').write_text('test'); root_writable=True
except OSError: root_writable=False
results={'key_file':readable(key),'score_file':readable(score),'key_env':'OPENROUTER_API_KEY' in os.environ,'key_network':endpoint(19419),'git_server':endpoint(19418),'push_succeeded':p.returncode==0,'checkout_writable':checkout_writable,'root_writable':root_writable,'docker_socket':pathlib.Path('/var/run/docker.sock').exists()}
print(json.dumps(results))
