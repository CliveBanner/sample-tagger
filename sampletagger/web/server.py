import argparse
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from . import api

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript",
    ".css": "text/css",
}

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")

class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        b = body.encode() if isinstance(body, str) else body
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(b)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def do_GET(self):
        route = urllib.parse.urlparse(self.path).path
        if route == "/" or route == "/index":
            self.serve_static("index.html")
        elif route == "/map":
            self.serve_static("map.html")
        elif route == "/review":
            self.serve_static("review.html")
        elif route == "/settings":
            self.serve_static("settings.html")
        elif route.startswith("/static/"):
            self.serve_static(route[len("/static/"):])
        elif route.startswith("/api/"):
            api.handle_get(self, route)
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        route = urllib.parse.urlparse(self.path).path
        if route.startswith("/api/"):
            api.handle_post(self, route)
        else:
            self._send(404, "not found", "text/plain")

    def serve_static(self, filename):
        path = os.path.abspath(os.path.join(STATIC, filename))
        if not path.startswith(STATIC) or not os.path.isfile(path):
            self._send(404, "not found", "text/plain")
            return
        ext = os.path.splitext(path)[1]
        ctype = MIME.get(ext, "application/octet-stream")
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    def log_message(self, *a):
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--db", default=os.path.join(os.getcwd(), "samples.db"))
    args = ap.parse_args()
    
    api.init(args.db)
    
    srv = ThreadingHTTPServer((args.host, args.port), H)
    print(f"dashboard: http://{args.host}:{args.port}  (db: {args.db})", flush=True)
    srv.serve_forever()

if __name__ == "__main__":
    main()
