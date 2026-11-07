# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Africa Centres for Disease Control and Prevention (Africa CDC)
"""Build the Blind-Spot Radar dashboard and serve it on this computer.

Run:   python run.py            then open http://localhost:8000 in your browser
       python run.py --port 8080    use another port if 8000 is busy
       python run.py --no-build     serve the page already in out/ without rebuilding

Stop with Ctrl+C. The dashboard is a static page. The server only shares the files in out/ and
listens on this computer only (127.0.0.1), so nobody else on the network can reach it.
On WSL2 the Windows browser can open localhost directly.
"""
import argparse
import functools
import http.server
import socketserver
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.path = "/blindspot.html"
        return super().do_GET()


class Server(socketserver.TCPServer):
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser(description="Build the dashboard and serve it on localhost.")
    ap.add_argument("--port", type=int, default=8000, help="port to listen on (default 8000)")
    ap.add_argument("--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1, this computer only)")
    ap.add_argument("--no-build", action="store_true", help="do not rebuild; serve the existing page in out/")
    args = ap.parse_args()

    out = ROOT / "out"
    if args.no_build and (out / "blindspot.html").exists():
        print("Serving the existing page in out/ (not rebuilt).")
    else:
        import build_dashboard_data
        build_dashboard_data.main()

    handler = functools.partial(Handler, directory=str(out))
    try:
        server = Server((args.host, args.port), handler)
    except OSError as e:
        sys.exit(f"Could not start on {args.host}:{args.port} ({e}). Try another port, for example: python run.py --port {args.port + 1}")
    print(f"\nBlind-Spot Radar is running. Open  http://localhost:{args.port}  in your browser.\nPress Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
