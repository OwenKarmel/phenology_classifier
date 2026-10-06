"""Local web server for the trailcam / phone phenology viewer.

    python3 server.py [--port 8000] [--host 127.0.0.1] [--rebuild]
                      [--annotations ../annotated_images] [--model-logs ../yolo/logs]

Serves the static app, the image index (built on first start), and resized
JPEG previews of the raw images. Previews are cached under ./cache; raw_data
is only ever read. Boxes drawn in the Annotate tab are written as a YOLO
dataset to the annotations folder (see annotations.py). The Model tab shows
the logs that yolo/pipeline.py writes to yolo/logs/ (read only).
"""
import argparse
import hashlib
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageOps

from annotations import Store

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.realpath(os.path.join(HERE, "..", "raw_data"))
STATIC = os.path.join(HERE, "static")
INDEX = os.path.join(HERE, "data", "index.json")
CACHE = os.path.join(HERE, "cache")
ANNOTATIONS = os.path.join(HERE, "..", "annotated_images")
STORE = None  # annotations.Store, set in main()
MODEL_LOGS = os.path.join(HERE, "..", "yolo", "logs")
LOG_NAME = re.compile(r"^[\w.-]+\.log$")
LOG_TAIL = 512 * 1024   # a log opens at its last 512 KB
LOG_CHUNK = 2 * 1024 * 1024

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


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError, TypeError):
        return True
    return True


def model_status(log_name):
    """Status written by yolo/runlog.py next to the log, with "running" checked
    against the process: a run killed outright never writes its exit."""
    path = os.path.join(MODEL_LOGS, log_name[:-4] + ".json")
    try:
        with open(path) as fh:
            st = json.load(fh)
    except (OSError, ValueError):
        st = {"state": "unknown"}
    if st.get("state") == "running" and not pid_alive(st.get("pid")):
        st["state"] = "lost"
    try:
        info = os.stat(os.path.join(MODEL_LOGS, log_name))
        st.update(size=info.st_size, modified=info.st_mtime)
    except OSError:
        pass
    st["name"] = log_name
    return st


def model_runs():
    try:
        names = sorted((n for n in os.listdir(MODEL_LOGS) if LOG_NAME.match(n)), reverse=True)
    except FileNotFoundError:
        names = []
    return {"dir": os.path.abspath(MODEL_LOGS), "now": time.time(), "runs": [model_status(n) for n in names]}


def utf8_prefix(data):
    """`data` without a UTF-8 character cut off at its end."""
    for i in range(1, min(4, len(data)) + 1):
        b = data[-i]
        if b & 0xC0 == 0x80:  # continuation byte: look further back
            continue
        if b >= 0xC0:  # lead byte of the last i bytes
            need = 2 if b < 0xE0 else 3 if b < 0xF0 else 4
            return data if i >= need else data[:-i]
        return data  # ASCII
    return data


def model_log(name, offset):
    """Text of one log from byte `offset` (negative: its last LOG_TAIL bytes)."""
    if not LOG_NAME.match(name or ""):
        raise ValueError("bad log name")
    path = os.path.join(MODEL_LOGS, name)
    with open(path, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        start = max(0, size - LOG_TAIL) if offset < 0 else min(offset, size)
        fh.seek(start)
        data = fh.read(LOG_CHUNK)
    skipped = 0
    if offset < 0 and start > 0:  # start at a whole line
        nl = data.find(b"\n")
        skipped = start + nl + 1 if nl >= 0 else start
        data = data[nl + 1:] if nl >= 0 else data
        start = skipped
    data = utf8_prefix(data)
    return {"name": name, "offset": start, "next": start + len(data), "size": size,
            "skipped": skipped, "text": data.decode("utf-8", "replace"), "status": model_status(name),
            "now": time.time()}


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
        if url.path == "/api/annotations":
            return self.send_json(STORE.snapshot())
        if url.path == "/api/model/runs":
            return self.send_json(model_runs())
        if url.path == "/api/model/log":
            try:
                return self.send_json(model_log(q.get("name", [""])[0], int(q.get("offset", ["-1"])[0])))
            except ValueError as e:
                return self.send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
            except FileNotFoundError:
                return self.send_json({"error": "no such log"}, HTTPStatus.NOT_FOUND)
        if url.path == "/raw":
            path = safe_raw_path(q.get("p", [""])[0])
            if not path:
                return self.send_error(HTTPStatus.NOT_FOUND)
            ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
            return self.send_file(path, ctype)
        return super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        # JSON only: a cross-site form post can't send this content type.
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            return self.send_json({"error": "expected application/json"}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 4_000_000:
                return self.send_json({"error": "request too large"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            body = json.loads(self.rfile.read(n) or b"{}")
            if url.path == "/api/annotations":
                source = os.path.normpath(str(body.get("source") or "")).replace(os.sep, "/")
                path = safe_raw_path(source)
                if not path:
                    return self.send_json({"error": f"no such image: {source}"}, HTTPStatus.NOT_FOUND)
                rec = STORE.save(source, path, body.get("status"), body.get("boxes"))
                return self.send_json({"record": rec})
            if url.path == "/api/classes":
                return self.send_json({"classes": STORE.set_classes(body.get("classes"))})
        except (ValueError, TypeError) as e:  # includes bad JSON
            return self.send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
        except OSError as e:
            return self.send_json({"error": f"could not write annotations: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
        return self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def send_json(self, obj, status=HTTPStatus.OK):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def end_headers(self):
        if urlparse(self.path).path in ("/", "/index.html", "/app.js", "/annotate.js", "/model.js", "/style.css"):
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
    global STORE, MODEL_LOGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--rebuild", action="store_true", help="rescan raw_data before starting")
    ap.add_argument("--annotations", default=ANNOTATIONS, metavar="DIR",
                    help="folder for the YOLO annotations (default: annotated_images in the project root)")
    ap.add_argument("--model-logs", default=MODEL_LOGS, metavar="DIR",
                    help="training logs shown in the Model tab (default: yolo/logs in the project root)")
    args = ap.parse_args()
    MODEL_LOGS = args.model_logs
    STORE = Store(args.annotations, os.path.join(HERE, "..", "raw_data"))
    STORE.write_dataset(STORE.classes())  # creates the folder with an empty dataset
    if args.rebuild or not os.path.exists(INDEX):
        build_index()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"Phenology viewer: http://{args.host}:{args.port}/  (Ctrl+C to stop)")
    print(f"Annotations are saved to {STORE.root}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
