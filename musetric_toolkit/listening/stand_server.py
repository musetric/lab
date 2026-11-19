import json
import logging
import re
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from musetric_toolkit.listening.audition import ONLY, WITHOUT
from musetric_toolkit.listening.round_files import CLIPS, MESSAGES, STAND
from musetric_toolkit.listening.stand_round import StandRound

LOG = logging.getLogger(__name__)
HOST = "127.0.0.1"
MAX_BODY = 2_000_000
PAGE = "index.html"
CLIP_FILE = re.compile(r"/clips/([\w-]+)/([\w-]+)\.(wav|png)")
OK = 200
BAD_REQUEST = 400
NOT_FOUND = 404
HTML_KIND = "text/html; charset=utf-8"
JSON_KIND = "application/json; charset=utf-8"
TEXT_KIND = "text/plain; charset=utf-8"


class StandHandler(BaseHTTPRequestHandler):
    state: StandRound

    def log_message(self, *args) -> None:
        return

    def send(
        self, code: int, body: bytes, kind: str, headers: dict | None = None
    ) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def fail(self, code: int, text: str) -> None:
        self.send(code, text.encode("utf-8"), TEXT_KIND)

    def json_body(self) -> dict | None:
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY:
            return None
        try:
            value = json.loads(self.rfile.read(length))
        except ValueError:
            return None
        return value if isinstance(value, dict) else None

    def send_round(self) -> None:
        state = self.state
        with state.lock:
            payload = {"manifest": state.manifest, "labels": state.labels}
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send(OK, body, JSON_KIND)

    def send_clip_file(self, path: str) -> None:
        state = self.state
        found = CLIP_FILE.fullmatch(path)
        if not found or found[1] not in state.clips or found[2] not in state.stems:
            self.fail(NOT_FOUND, "not found")
            return
        file = state.folder / CLIPS / found[1] / f"{found[2]}.{found[3]}"
        if not file.is_file():
            self.fail(NOT_FOUND, "not found")
            return
        kind = "audio/wav" if found[3] == "wav" else "image/png"
        self.send(OK, file.read_bytes(), kind)

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/":
            self.send(OK, (STAND / PAGE).read_bytes(), HTML_KIND)
        elif path == f"/{MESSAGES.name}":
            self.send(OK, MESSAGES.read_bytes(), JSON_KIND)
        elif path == "/api/round":
            self.send_round()
        else:
            self.send_clip_file(path)

    def post_save(self, body: dict) -> None:
        if self.state.save(body.get("clip"), body.get("data")):
            self.send(OK, b"{}", JSON_KIND)
        else:
            self.fail(BAD_REQUEST, "bad label")

    def post_audition(self, body: dict) -> None:
        state = self.state
        clip, stem, mode = body.get("clip"), body.get("stem"), body.get("mode")
        probe = dict(
            cls=next(iter(state.classes)), stem=stem, polygon=body.get("polygon")
        )
        if (
            clip not in state.clips
            or mode not in (ONLY, WITHOUT)
            or not state.valid_region(probe)
        ):
            self.fail(BAD_REQUEST, "bad audition")
            return
        wave, boost, start = state.audition(clip, stem, mode, body["polygon"])
        headers = {"X-Boost-Db": f"{boost:.1f}", "X-Start-S": f"{start:.3f}"}
        self.send(OK, wave, "audio/wav", headers)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        body = self.json_body()
        if body is None:
            self.fail(BAD_REQUEST, "bad body")
        elif path == "/api/save":
            self.post_save(body)
        elif path == "/api/audition":
            self.post_audition(body)
        else:
            self.fail(NOT_FOUND, "not found")


def serve_command(args) -> None:
    state = StandRound(args.round)
    handler = type("RoundHandler", (StandHandler,), {"state": state})
    server = ThreadingHTTPServer((HOST, args.port), handler)
    address = f"http://{HOST}:{args.port}"
    LOG.info("%s  %d clips, labels in %s", address, len(state.clips), state.labels_path)
    if args.open:
        webbrowser.open(address)
    server.serve_forever()
