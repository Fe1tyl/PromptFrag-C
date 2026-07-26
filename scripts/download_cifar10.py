"""Resume, verify, and safely extract the official clean CIFAR-10 archive."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from download_cifar10c import download_with_resume, extract_archive, md5sum

OFFICIAL_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
OFFICIAL_MD5 = "c58f30108f718f92721af3b95e74349a"
ARCHIVE_NAME = "cifar-10-python.tar.gz"


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


def dataset_is_valid(data_root: Path) -> bool:
    """Use torchvision's published per-file checksums to validate extraction."""
    from torchvision.datasets import CIFAR10

    try:
        CIFAR10(root=str(data_root), train=False, download=False)
    except RuntimeError:
        return False
    return True


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    archive_path = data_root / ARCHIVE_NAME

    if dataset_is_valid(data_root) and not args.force_download:
        print(f"Clean CIFAR-10 already exists: {data_root / 'cifar-10-batches-py'}")
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
    print("Checksum verified. Extracting clean CIFAR-10...")
    extract_archive(archive_path, data_root)
    if not dataset_is_valid(data_root):
        raise SystemExit(
            "Extraction completed, but torchvision's CIFAR-10 integrity check failed"
        )
    if args.remove_archive:
        os.remove(archive_path)
    print(f"Clean CIFAR-10 ready: {data_root / 'cifar-10-batches-py'}")


if __name__ == "__main__":
    main()
