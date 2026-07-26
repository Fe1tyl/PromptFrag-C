"""Resume, verify, and safely extract the official CIFAR-10-C archive."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import os
import re
import socket
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

OFFICIAL_URL = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
OFFICIAL_MD5 = "56bf5dcef84df0e2308c6dcbcbbd8499"
CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)")
UNSATISFIED_RANGE = re.compile(r"bytes \*/(\d+)")
CHUNK_SIZE = 8 * 1024 * 1024


class DownloadProtocolError(RuntimeError):
    """The server response is unsafe to append to the partial archive."""


class TransferInterrupted(RuntimeError):
    """The server closed a response before all advertised bytes arrived."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--url", default=OFFICIAL_URL)
    parser.add_argument("--remove-archive", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--max-no-progress-retries",
        type=int,
        default=12,
        help="Stop after this many consecutive reconnects that add no bytes.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Base delay in seconds between reconnect attempts.",
    )
    return parser.parse_args()


def md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()  # nosec B324 - required to verify the published checksum
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _response_layout(
    response: object,
    existing: int,
    known_total: int | None,
) -> tuple[str, int | None]:
    status = int(getattr(response, "status", response.getcode()))
    content_range = response.headers.get("Content-Range")
    content_length_text = response.headers.get("Content-Length")
    content_length = int(content_length_text) if content_length_text else None

    if status == 206:
        match = CONTENT_RANGE.fullmatch(content_range or "")
        if not match:
            raise DownloadProtocolError(
                "The server returned a partial response without a valid Content-Range."
            )
        start, end = int(match.group(1)), int(match.group(2))
        total = None if match.group(3) == "*" else int(match.group(3))
        if start != existing:
            raise DownloadProtocolError(
                f"The server resumed at byte {start}, but the local archive has "
                f"{existing} bytes."
            )
        if end < start:
            raise DownloadProtocolError("The server returned an invalid byte range.")
        if content_length is not None and content_length != end - start + 1:
            raise DownloadProtocolError(
                "Content-Length and Content-Range disagree; refusing to append."
            )
        mode = "ab" if existing else "wb"
    elif status == 200:
        if existing:
            raise DownloadProtocolError(
                "The server ignored the resume request. Keep the partial archive and "
                "try again later, or use --force-download to restart from byte zero."
            )
        total = content_length
        mode = "wb"
    else:
        raise DownloadProtocolError(f"Unexpected HTTP status {status}.")

    if total is not None and known_total is not None and total != known_total:
        raise DownloadProtocolError(
            f"The remote file size changed from {known_total} to {total} bytes."
        )
    return mode, total


def _print_progress(
    downloaded: int,
    total: int | None,
    session_start_size: int,
    session_started: float,
) -> None:
    elapsed = max(time.monotonic() - session_started, 1e-6)
    speed = (downloaded - session_start_size) / elapsed
    if total:
        percent = 100.0 * downloaded / total
        print(
            f"\rDownloaded {downloaded / 2**30:.2f}/{total / 2**30:.2f} "
            f"GiB ({percent:.1f}%) at {speed / 2**20:.1f} MiB/s",
            end="",
            flush=True,
        )
    else:
        print(
            f"\rDownloaded {downloaded / 2**30:.2f} GiB "
            f"at {speed / 2**20:.1f} MiB/s",
            end="",
            flush=True,
        )


def download_with_resume(
    url: str,
    destination: Path,
    *,
    max_no_progress_retries: int = 12,
    retry_delay: float = 2.0,
    report_interval: float = 5.0,
) -> None:
    if max_no_progress_retries < 0:
        raise ValueError("max_no_progress_retries must be non-negative")
    if retry_delay < 0:
        raise ValueError("retry_delay must be non-negative")

    session_start_size = destination.stat().st_size if destination.exists() else 0
    session_started = time.monotonic()
    known_total: int | None = None
    no_progress_failures = 0
    headers = {"User-Agent": "PromptFrag-C/0.1 academic benchmark"}

    while True:
        existing = destination.stat().st_size if destination.exists() else 0
        attempt_start_size = existing
        request_headers = dict(headers)
        if existing:
            request_headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(url, headers=request_headers)

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                mode, response_total = _response_layout(
                    response, existing, known_total
                )
                if response_total is not None:
                    known_total = response_total
                downloaded = existing
                last_report = time.monotonic()
                with destination.open(mode) as handle:
                    while True:
                        try:
                            chunk = response.read(CHUNK_SIZE)
                        except http.client.IncompleteRead as exc:
                            chunk = exc.partial
                            if chunk:
                                handle.write(chunk)
                                downloaded += len(chunk)
                            raise TransferInterrupted(
                                "The HTTP response ended before Content-Length bytes "
                                "were received."
                            ) from exc
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= report_interval:
                            _print_progress(
                                downloaded,
                                known_total,
                                session_start_size,
                                session_started,
                            )
                            last_report = now

                current_size = destination.stat().st_size
                _print_progress(
                    current_size,
                    known_total,
                    session_start_size,
                    session_started,
                )
                print()
                if known_total is None:
                    return
                if current_size == known_total:
                    return
                if current_size > known_total:
                    raise DownloadProtocolError(
                        f"The local archive is {current_size} bytes, larger than the "
                        f"advertised {known_total} bytes."
                    )
                raise TransferInterrupted(
                    f"Connection ended at {current_size} of {known_total} bytes."
                )
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and destination.exists():
                match = UNSATISFIED_RANGE.fullmatch(
                    exc.headers.get("Content-Range", "")
                )
                current_size = destination.stat().st_size
                if match and current_size == int(match.group(1)):
                    print(f"Remote file is already complete ({current_size} bytes).")
                    return
            error: Exception = exc
        except DownloadProtocolError:
            raise
        except (
            TransferInterrupted,
            urllib.error.URLError,
            socket.timeout,
            TimeoutError,
            ConnectionError,
            OSError,
        ) as exc:
            error = exc

        current_size = destination.stat().st_size if destination.exists() else 0
        progressed = current_size > attempt_start_size
        if progressed:
            no_progress_failures = 0
            delay = retry_delay
        else:
            no_progress_failures += 1
            if no_progress_failures > max_no_progress_retries:
                raise RuntimeError(
                    "Download made no progress after "
                    f"{max_no_progress_retries} reconnect attempts. The partial "
                    "archive is safe; rerun the same command later to resume."
                ) from error
            delay = min(retry_delay * (2 ** (no_progress_failures - 1)), 30.0)

        print(
            f"Transfer interrupted ({error}). Retrying from byte {current_size} "
            f"in {delay:.1f}s..."
        )
        if delay:
            time.sleep(delay)


def validate_members(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"unsafe archive path: {member.name}") from exc
        if member.issym() or member.islnk():
            raise RuntimeError(f"archive links are not allowed: {member.name}")


def extract_archive(archive_path: Path, data_root: Path) -> None:
    with tarfile.open(archive_path, mode="r") as archive:
        validate_members(archive, data_root)
        if sys.version_info >= (3, 12):
            archive.extractall(data_root, filter="data")
        else:
            archive.extractall(data_root)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    dataset_root = data_root / "CIFAR-10-C"
    archive_path = data_root / "CIFAR-10-C.tar"

    if (dataset_root / "labels.npy").exists() and not args.force_download:
        print(f"Dataset already exists: {dataset_root}")
        return
    if args.force_download:
        archive_path.unlink(missing_ok=True)

    if not archive_path.exists() or md5sum(archive_path) != OFFICIAL_MD5:
        download_with_resume(
            args.url,
            archive_path,
            max_no_progress_retries=args.max_no_progress_retries,
            retry_delay=args.retry_delay,
        )

    actual_md5 = md5sum(archive_path)
    if actual_md5 != OFFICIAL_MD5:
        raise SystemExit(
            f"Checksum mismatch: expected {OFFICIAL_MD5}, received {actual_md5}. "
            "If the reported byte count reached the remote total, rerun with "
            "--force-download; otherwise rerun normally to resume."
        )
    print("Checksum verified. Extracting...")
    extract_archive(archive_path, data_root)
    if not (dataset_root / "labels.npy").exists():
        raise SystemExit("Extraction completed, but CIFAR-10-C/labels.npy is missing")
    if args.remove_archive:
        os.remove(archive_path)
    print(f"CIFAR-10-C ready: {dataset_root}")


if __name__ == "__main__":
    main()
