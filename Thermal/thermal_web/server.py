"""Loopback HTTP API and offline assets. No upload to a hosted service."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs,urlsplit
import json,mimetypes,platform,secrets
from capture_replay import discover
from heat_function import HeatConfig
from dataclasses import asdict
from .workspace import Workspace


class Application:
    def __init__(self,thermal,weights=None,worker_factory=None):
        self.thermal=Path(thermal).resolve();self.token=secrets.token_urlsafe(32)
        factory=worker_factory or Workspace
        self.workspaces={kind:factory(self.thermal,kind,weights) for kind in ('launcher','simulator')}

    def close(self):
        for workspace in self.workspaces.values():workspace.close()

    def metadata(self):
        captures=[]
        for folder in ('logs','paired_recordings'):
            captures.extend(discover(self.thermal/folder))
        return dict(token=self.token,path=str(self.thermal),platform=platform.system(),captures=captures,
            heat_defaults=asdict(HeatConfig()),camera_backend='Windows Y16 / DirectShow or Media Foundation' if platform.system()=='Windows' else 'libuvc Y16',
            privacy='Runs on this computer. No cloud or screen recording.')


def handler_for(app):
    assets=Path(__file__).resolve().parent/'static'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass

        def reply(self,status,data=b'',kind='application/json'):
            if not isinstance(data,bytes):data=json.dumps(data,allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(data)

        def trusted_host(self):
            try:return urlsplit('//'+self.headers.get('Host','')).hostname in ('localhost','127.0.0.1','::1')
            except ValueError:return False

        def workspace(self,params):
            key=params.get('workspace',['launcher'])[0]
            if key not in app.workspaces:raise ValueError('Unknown workspace.')
            return app.workspaces[key]

        def do_GET(self):
            if not self.trusted_host():self.reply(403,dict(error='Local connections only.'));return
            url=urlsplit(self.path);params=parse_qs(url.query)
            try:
                if url.path=='/api/meta':self.reply(200,app.metadata())
                elif url.path=='/api/state':
                    snapshot=self.workspace(params).snapshot()
                    if str(snapshot['version'])==params.get('since',[None])[0]:self.reply(204,b'')
                    else:self.reply(200,snapshot)
                elif url.path=='/api/export':
                    data=json.dumps(self.workspace(params).export(),allow_nan=False,indent=2).encode()
                    self.send_response(200);self.send_header('Content-Type','application/json')
                    self.send_header('Content-Disposition','attachment; filename="occupancy-session.json"')
                    self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
                elif url.path in ('/','/static/app.js','/static/style.css'):
                    path=assets/('index.html' if url.path=='/' else Path(url.path).name)
                    self.reply(200,path.read_bytes(),(mimetypes.guess_type(path)[0] or 'application/octet-stream')+'; charset=utf-8')
                else:self.reply(404,dict(error='Not found.'))
            except (ValueError,KeyError,FileNotFoundError) as error:self.reply(400,dict(error=str(error)))
            except Exception as error:self.reply(500,dict(error=str(error)))

        def do_POST(self):
            origin=self.headers.get('Origin')
            if not self.trusted_host() or (origin and origin!='http://'+self.headers.get('Host')):
                self.reply(403,dict(error='Local same-origin controls only.'));return
            if self.headers.get('X-Workspace-Token')!=app.token:
                self.reply(403,dict(error='Reload the workspace to reconnect.'));return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=65536:raise ValueError('Invalid command size.')
                payload=json.loads(self.rfile.read(length))
                if urlsplit(self.path).path!='/api/command':self.reply(404,dict(error='Not found.'));return
                key=payload['workspace']
                if key not in app.workspaces:raise ValueError('Unknown workspace.')
                command=payload['command']
                if not isinstance(command,dict) or not isinstance(command.get('action'),str):raise ValueError('Invalid command.')
                app.workspaces[key].submit(command);self.reply(202,dict(accepted=True))
            except (ValueError,KeyError,TypeError) as error:self.reply(400,dict(error=str(error)))

    return Handler


def create_server(app,port=8780):
    server=ThreadingHTTPServer(('127.0.0.1',port),handler_for(app))
    server.daemon_threads=True
    return server
