#!/usr/bin/env python3
"""Local web page, standard library only:  python server.py  ->  http://127.0.0.1:8000
Binds to localhost by default. Use --host 0.0.0.0 only when you deliberately host it.

  GET  /                  the pitch box
  POST /check             run a comparison, save the run, redirect to /run/<id>
  GET  /run/<id>          the report for a saved run (any status, including failures)
  GET  /download/<id>.json   the saved run as a JSON download (never contains a key)
"""
import argparse
import json
import os
import re
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pitchcheck.llm import load_config  # noqa: E402
from pitchcheck.pipeline import CardCache, check_pitch, load_catalog  # noqa: E402
from pitchcheck.render import render_page  # noqa: E402
from pitchcheck.runs import load_run, save_run  # noqa: E402
from pitchcheck.textutil import setup_stdout  # noqa: E402

STATE = {}
LOCK = threading.Lock()  # one editor, one review at a time
MAX_BODY = 20000


class H(BaseHTTPRequestHandler):
    def _send(self, body, code=200, ctype="text/html; charset=utf-8", headers=None):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _page(self, pitch="", result=None, code=200):
        self._send(render_page(STATE["cfg"], STATE["catalog"], pitch, result), code)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._page()
        m = re.fullmatch(r"/run/([a-f0-9]{12})", path)
        if m:
            run = load_run(m.group(1), STATE["runs_dir"])
            return self._page(run["pitch"], run) if run else self._send("run not found", 404)
        m = re.fullmatch(r"/download/([a-f0-9]{12})\.json", path)
        if m:
            run = load_run(m.group(1), STATE["runs_dir"])
            if not run:
                return self._send("run not found", 404)
            return self._send(json.dumps(run, ensure_ascii=False, indent=1), 200, "application/json; charset=utf-8",
                              {"Content-Disposition": f'attachment; filename="premise-check-run-{m.group(1)}.json"'})
        self._send("not found", 404)

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path != "/check":
            return self._send("not found", 404)
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._send("bad request", 400)
        if n > MAX_BODY:
            return self._send("the pitch is too long (a 2-4 sentence pitch is expected)", 413)
        form = urllib.parse.parse_qs(self.rfile.read(n).decode("utf-8", "replace"))
        pitch = (form.get("pitch", [""])[0]).strip()
        with LOCK:
            result = check_pitch(pitch, STATE["catalog"], STATE["cfg"], STATE["cache"])
            save_run(result, STATE["runs_dir"])  # failed runs are kept too
        self.send_response(303)
        self.send_header("Location", f"/run/{result['run_id']}")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *a):
        pass


def make_server(host, port, cfg, catalog, runs_dir, cache_path):
    STATE.update(cfg=cfg, catalog=catalog, runs_dir=runs_dir, cache=CardCache(cache_path))
    return ThreadingHTTPServer((host, port), H)


def main():
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--config", default=None)
    ap.add_argument("--catalog", default=os.path.join(HERE, "data", "catalog.csv"))
    ap.add_argument("--runs-dir", default=os.path.join(HERE, "outputs", "runs"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    srv = make_server(a.host, a.port, cfg, load_catalog(a.catalog), a.runs_dir, os.path.join(HERE, "outputs", "cards_cache.json"))
    print(f"serving on http://{a.host}:{a.port}   provider: {cfg.get('provider')} / {cfg.get('model')}")
    if str(cfg.get("provider", "")).startswith("mock"):
        print("*** MOCK PROVIDER: offline stand-in, not a language model. Set a real provider in config.json. ***")
    srv.serve_forever()


if __name__ == "__main__":
    main()
