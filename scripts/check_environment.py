
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    import torch

    cuda_available = torch.cuda.is_available()
    payload = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": version("torchvision"),
        "open_clip_torch": version("open_clip_torch"),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "scipy": version("scipy"),
        "cuda_available": cuda_available,
        "torch_cuda": torch.version.cuda,
    }
    if cuda_available:
        payload["gpu"] = torch.cuda.get_device_name(0)
        payload["compute_capability"] = list(torch.cuda.get_device_capability(0))
        payload["vram_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
        first = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
        second = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
        result = first @ second
        torch.cuda.synchronize()
        payload["cuda_smoke_test"] = bool(torch.isfinite(result).all().item())
    else:
        payload["cuda_smoke_test"] = False

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.require_cuda and not cuda_available:
        raise SystemExit("CUDA is required but unavailable")
    if cuda_available and not payload["cuda_smoke_test"]:
        raise SystemExit("CUDA is visible, but the compute smoke test failed")


if __name__ == "__main__":
    main()

