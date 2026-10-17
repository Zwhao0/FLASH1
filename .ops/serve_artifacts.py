import http.server,functools,pathlib,time,os
root=pathlib.Path('/home/campus/zhangweihao/FLASH1/artifacts/multinode')
root.mkdir(exist_ok=True,parents=True)
class Restricted(http.server.SimpleHTTPRequestHandler):
 def do_GET(self):
  if self.client_address[0]!=os.environ.get('FLASH1_SERVE_CLIENT','10.10.4.2'):self.send_error(403);return
  super().do_GET()
 def do_HEAD(self):
  if self.client_address[0]!=os.environ.get('FLASH1_SERVE_CLIENT','10.10.4.2'):self.send_error(403);return
  super().do_HEAD()
s=http.server.ThreadingHTTPServer((os.environ.get('FLASH1_SERVE_ADDR','10.10.4.1'),18080),functools.partial(Restricted,directory=str(root)))
s.timeout=1
end=time.monotonic()+1800
while time.monotonic()<end:s.handle_request()
