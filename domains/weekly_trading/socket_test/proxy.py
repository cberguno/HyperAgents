# Synthetic protocol test only. Not a production OpenRouter proxy.
import json, os, socketserver
from http.server import BaseHTTPRequestHandler
SOCKET='/channel/completion.sock'
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*a): pass
 def answer(self,code,obj):
  b=json.dumps(obj).encode(); self.send_response(code); self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
 def do_GET(self): self.answer(404,{'error':'denied'})
 def do_POST(self):
  if self.path!='/v1/chat/completions':return self.answer(404,{'error':'denied'})
  n=self.headers.get('Content-Length','')
  if not n.isdigit() or int(n)>16384:return self.answer(400,{'error':'invalid'})
  try: req=json.loads(self.rfile.read(int(n)))
  except Exception:return self.answer(400,{'error':'invalid'})
  if not isinstance(req,dict) or set(req)!={'messages'}:return self.answer(400,{'error':'invalid'})
  self.answer(200,{'choices':[{'message':{'content':'synthetic completion'}}]})
class Server(socketserver.UnixStreamServer): pass
try:os.unlink(SOCKET)
except FileNotFoundError:pass
with Server(SOCKET,Handler) as s:
 os.chmod(SOCKET,0o666);s.serve_forever()
