"""Evaluate the frozen PromptFrag-C prompt bank on CIFAR-10.1 v6.

CIFAR-10.1 is a naturally resampled test set with the same ten classes as
CIFAR-10.  Model weights and prompt text are unchanged.  The script saves
per-prompt metrics and per-example predictions for paired bootstrap analyses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

from promptfragc.clip_model import (
    encode_prompt_bank,
    infer_prompt_probabilities,
    load_clip_bundle,
)
from promptfragc.constants import CIFAR10_CLASSES, DEFAULT_PROMPT_ID, PROMPTS
from promptfragc.metrics import classification_metrics


URLS = {
    "cifar10.1_v6_data.npy": (
        "https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/"
        "master/datasets/cifar10.1_v6_data.npy"
    ),
    "cifar10.1_v6_labels.npy": (
        "https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/"
        "master/datasets/cifar10.1_v6_labels.npy"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/CIFAR-10.1")
    parser.add_argument("--output-dir", default="outputs/revision/cifar10_1")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ece-bins", type=int, default=15)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_with_retries(url: str, destination: Path, retries: int = 8) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "PromptFrag-C-revision/1.0"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                with partial.open("wb") as handle:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
            os.replace(partial, destination)
            return
        except (OSError, urllib.error.URLError) as error:
            if partial.exists():
                partial.unlink()
            if attempt == retries:
                raise RuntimeError(f"failed to download {url}: {error}") from error
            time.sleep(min(2**attempt, 30))


class CIFAR101Dataset:
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform) -> None:
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        from PIL import Image

        image = Image.fromarray(self.images[index])
        return self.transform(image), int(self.labels[index])


def evaluate_model(
    model_name: str,
    pretrained: str,
    images: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    device: str,
    ece_bins: int,
) -> list[dict[str, object]]:
    import torch
    from torch.utils.data import DataLoader

    bundle = load_clip_bundle(model_name, pretrained, device)
    text_features = encode_prompt_bank(bundle, CIFAR10_CLASSES, PROMPTS)
    dataset = CIFAR101Dataset(images, labels, bundle.preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bundle.device.type == "cuda",
    )
    probability_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    for batch_images, batch_labels in loader:
        probabilities = infer_prompt_probabilities(
            bundle, text_features, batch_images
        )
        probability_batches.append(probabilities.detach().cpu().numpy())
        label_batches.append(batch_labels.numpy())
    probabilities = np.concatenate(probability_batches, axis=1)
    targets = np.concatenate(label_batches)

    rows: list[dict[str, object]] = []
    predicted_classes = probabilities.argmax(axis=2).astype(np.int16)
    confidences = probabilities.max(axis=2).astype(np.float32)
    np.savez_compressed(
        output_dir / f"{model_name}_{pretrained}_predictions.npz",
        targets=targets.astype(np.int16),
        predicted_classes=predicted_classes,
        confidences=confidences,
    )
    for prompt_index, (prompt_id, template) in enumerate(PROMPTS):
        metrics = classification_metrics(
            probabilities[prompt_index], targets, n_bins=ece_bins
        )
        rows.append(
            {
                "model": model_name,
                "pretrained": pretrained,
                "dataset": "CIFAR-10.1-v6",
                "prompt_id": prompt_id,
                "prompt_template": template,
                "method": "single",
                "n_samples": int(targets.size),
                **metrics,
            }
        )
    ensemble = probabilities.mean(axis=0)
    ensemble /= ensemble.sum(axis=1, keepdims=True)
    rows.append(
        {
            "model": model_name,
            "pretrained": pretrained,
            "dataset": "CIFAR-10.1-v6",
            "prompt_id": "ensemble",
            "prompt_template": "mean probability across 12 prompts",
            "method": "ensemble",
            "n_samples": int(targets.size),
            **classification_metrics(ensemble, targets, n_bins=ece_bins),
        }
    )
    del bundle, text_features, probabilities
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in URLS.items():
        download_with_retries(url, data_dir / filename)

    images = np.load(data_dir / "cifar10.1_v6_data.npy")
    labels = np.load(data_dir / "cifar10.1_v6_labels.npy")
    if images.shape != (2000, 32, 32, 3) or labels.shape != (2000,):
        raise ValueError(
            f"unexpected CIFAR-10.1 shapes: images={images.shape}, labels={labels.shape}"
        )
    if images.dtype != np.uint8 or not set(np.unique(labels)).issubset(range(10)):
        raise ValueError("CIFAR-10.1 arrays have unexpected values or dtypes")

    rows: list[dict[str, object]] = []
    for model_name in ("ViT-B-32", "RN50"):
        rows.extend(
            evaluate_model(
                model_name=model_name,
                pretrained="openai",
                images=images,
                labels=labels,
                output_dir=output_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
                ece_bins=args.ece_bins,
            )
        )
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
    manifest = {
        "dataset": "CIFAR-10.1 v6",
        "source_repository": "https://github.com/modestyachts/CIFAR-10.1",
        "files": {
            filename: {
                "url": url,
                "sha256": sha256(data_dir / filename),
                "bytes": (data_dir / filename).stat().st_size,
            }
            for filename, url in URLS.items()
        },
        "n_samples": int(labels.size),
        "class_counts": np.bincount(labels, minlength=10).astype(int).tolist(),
        "models": ["ViT-B-32/openai", "RN50/openai"],
        "prompts": len(PROMPTS),
        "default_prompt": DEFAULT_PROMPT_ID,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
