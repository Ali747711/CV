"""A loopback-only lookup/tool API for lesson 6; no server starts on import."""
from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from lab import answer_request


@contextmanager
def serve_local():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            # Avoid putting request content into terminal logs.
            return

        def reply(self, status, payload):
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/ask":
                self.reply(404, {"error": "Unknown route."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.reply(400, {"error": "Invalid Content-Length."})
                return
            if not 1 <= length <= 4096:
                self.reply(413, {"error": "Body must contain 1 to 4096 bytes."})
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                response = answer_request(payload)
            except (UnicodeDecodeError, ValueError):
                self.reply(400, {"error": "Invalid request: check JSON, question or tool inputs."})
                return
            self.reply(200, response)

    # Port 0 lets the OS choose an available port. The context closes only our server.
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        server.daemon_threads = True
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            worker.join(timeout=3)
            if worker.is_alive():
                raise RuntimeError("Lesson server did not stop cleanly.")
