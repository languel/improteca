"""Serve the folder for development, telling the browser not to cache, so an
edited module is always the one that runs.

    python3 tools/serve.py [port]   (default 5310)
"""

import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class NoCache(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5310
    root = Path(__file__).resolve().parents[1]
    server = ThreadingHTTPServer(("", port), partial(NoCache, directory=str(root)))
    print(f"Serving {root} at http://localhost:{port}/ (no caching)", flush=True)
    server.serve_forever()
