import http.client,json,os,socket,subprocess,sys
class Connection(http.client.HTTPConnection):
 def connect(self):
  self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);self.sock.settimeout(3);self.sock.connect('/channel/completion.sock')
def ask(method,path,body=None):
 c=Connection('localhost');c.request(method,path,body,{'Content-Type':'application/json'});r=c.getresponse();b=r.read().decode();c.close();return r.status,b
status,body=ask('POST','/v1/chat/completions',json.dumps({'messages':[]}));assert status==200 and 'synthetic completion' in body
for method,path in [('GET','/v1/chat/completions'),('POST','/v1/models'),('POST','/v1/chat/completions?url=elsewhere'),('POST','/v1/chat/completions/../key')]:
 status,_=ask(method,path,'{}');assert status==404,(method,path,status)
assert ask('POST','/v1/chat/completions',json.dumps({'messages':[],'url':'http://evil'}))[0]==400
assert not os.getenv('OPENROUTER_API_KEY')
for p in ['/host-secret/key','/score-data/secret','/var/run/docker.sock','/github/home/.git-credentials','/cache/secret','/logs/secret','/artifacts/secret']:
 assert not os.path.exists(p),p
try:socket.create_connection(('1.1.1.1',443),timeout=2);raise AssertionError('outbound TCP reachable')
except OSError:pass
try:socket.getaddrinfo('github.com',443);raise AssertionError('DNS available')
except socket.gaierror:pass
assert not subprocess.run(['git','-c','safe.directory=/scratch','-C','/scratch','remote'],capture_output=True,text=True,check=True).stdout.strip()
open('/scratch/changed','w').write('scratch works')
print('socket route checks passed; key env, listed files, socket, DNS, TCP absent; scratch writable and no remote')
