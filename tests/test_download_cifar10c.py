from __future__ import annotations

import importlib.util
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "download_cifar10c.py"
SPEC = importlib.util.spec_from_file_location("download_cifar10c", SCRIPT)
assert SPEC and SPEC.loader
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


def test_download_resumes_after_repeated_truncation(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 2048
    bytes_per_connection = 64 * 1024

    class TruncatingHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            range_header = self.headers.get("Range")
            start = (
                int(range_header.removeprefix("bytes=").removesuffix("-"))
                if range_header
                else 0
            )
            remaining = len(payload) - start
            self.send_response(206 if range_header else 200)
            self.send_header("Content-Length", str(remaining))
            if range_header:
                self.send_header(
                    "Content-Range",
                    f"bytes {start}-{len(payload) - 1}/{len(payload)}",
                )
            self.end_headers()
            self.wfile.write(payload[start : start + bytes_per_connection])
            self.close_connection = True

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), TruncatingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    destination = tmp_path / "archive.tar"
    try:
        downloader.download_with_resume(
            f"http://127.0.0.1:{server.server_port}/archive.tar",
            destination,
            retry_delay=0,
            report_interval=3600,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert destination.read_bytes() == payload


def test_resume_refuses_a_server_that_ignores_range(tmp_path: Path) -> None:
    class NoRangeHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", "4")
            self.end_headers()
            self.wfile.write(b"full")

        def log_message(self, format: str, *args: object) -> None:
            return

    destination = tmp_path / "archive.tar"
    destination.write_bytes(b"partial")
    server = ThreadingHTTPServer(("127.0.0.1", 0), NoRangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(downloader.DownloadProtocolError):
            downloader.download_with_resume(
                f"http://127.0.0.1:{server.server_port}/archive.tar",
                destination,
                retry_delay=0,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert destination.read_bytes() == b"partial"
