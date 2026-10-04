"""Local web server for the trailcam / phone phenology viewer.

    python3 server.py [--port 8000] [--host 127.0.0.1] [--rebuild]

Serves the static app, the image index (built on first start), and resized
JPEG previews of the raw images. Previews are cached under ./cache; raw_data
is only ever read.
"""
import argparse
import hashlib
import mimetypes
import os
import subprocess
import sys
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.realpath(os.path.join(HERE, "..", "raw_data"))
STATIC = os.path.join(HERE, "static")
INDEX = os.path.join(HERE, "data", "index.json")
CACHE = os.path.join(HERE, "cache")

MAX_W = 4096
_locks = {}
_locks_guard = threading.Lock()


def build_index():
    subprocess.run([sys.executable, os.path.join(HERE, "build_index.py")], check=True)


def safe_raw_path(rel):
    """Resolve a raw_data-relative path, refusing anything outside raw_data."""
    if not rel:
        return None
    path = os.path.realpath(os.path.join(RAW, rel))
    if not path.startswith(RAW + os.sep) or not os.path.isfile(path):
        return None
    return path


def preview(path, width):
    """Return the cache path of a JPEG preview `width` px wide (made on demand)."""
    st = os.stat(path)
    key = hashlib.sha1(f"{path}|{st.st_mtime_ns}|{st.st_size}|{width}".encode()).hexdigest()
    out = os.path.join(CACHE, str(width), key[:2], key + ".jpg")
    if os.path.exists(out):
        return out
    with _locks_guard:
        lock = _locks.setdefault(out, threading.Lock())
    with lock:
        if os.path.exists(out):
            return out
        with Image.open(path) as im:
            # JPEG draft mode decodes at 1/2, 1/4 or 1/8 scale: much faster for
            # the 9216x5184 trailcam frames.
            if im.format == "JPEG":
                im.draft("RGB", (width, width))
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            if im.width > width:
                h = round(im.height * width / im.width)
                im = im.resize((width, h), Image.LANCZOS)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            tmp = out + f".{threading.get_ident()}.tmp"
            im.save(tmp, "JPEG", quality=85, optimize=True, progressive=True)
            os.replace(tmp, out)
    with _locks_guard:
        _locks.pop(out, None)
    return out


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=STATIC, **kw)

    def log_message(self, fmt, *args):
        if os.environ.get("VIEWER_VERBOSE"):
            super().log_message(fmt, *args)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/api/index":
            if "rebuild" in q or not os.path.exists(INDEX):
                build_index()
            return self.send_file(INDEX, "application/json", cache=False)
        if url.path == "/img":
            path = safe_raw_path(q.get("p", [""])[0])
            if not path:
                return self.send_error(HTTPStatus.NOT_FOUND)
            try:
                width = max(64, min(MAX_W, int(q.get("w", ["1600"])[0])))
                return self.send_file(preview(path, width), "image/jpeg")
            except Exception as e:
                return self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))
        if url.path == "/raw":
            path = safe_raw_path(q.get("p", [""])[0])
            if not path:
                return self.send_error(HTTPStatus.NOT_FOUND)
            ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
            return self.send_file(path, ctype)
        return super().do_GET()

    def end_headers(self):
        if urlparse(self.path).path in ("/", "/index.html", "/app.js", "/style.css"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def send_file(self, path, ctype, cache=True):
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError:
            return self.send_error(HTTPStatus.NOT_FOUND)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=86400" if cache else "no-cache")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--rebuild", action="store_true", help="rescan raw_data before starting")
    args = ap.parse_args()
    if args.rebuild or not os.path.exists(INDEX):
        build_index()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"Phenology viewer: http://{args.host}:{args.port}/  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
