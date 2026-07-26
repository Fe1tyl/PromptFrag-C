"""Resumable zero-shot CLIP evaluation runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Hugging Face's Xet/CAS reconstruction is fragile on interrupted or filtered
# connections. Use the regular resumable HTTPS downloader instead. These values
# must be set before huggingface_hub is first imported.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
# Required by deterministic cuBLAS matrix multiplications on CUDA 10.2+.
# This must be set before torch initializes CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

from .clip_model import encode_prompt_bank, infer_prompt_probabilities, load_clip_bundle
from .constants import (
    CIFAR10_CLASSES,
    METRIC_COLUMNS,
    PROMPTS,
    STANDARD_CORRUPTIONS,
)
from .data import make_dataset
from .metrics import classification_metrics

RESULT_COLUMNS = (
    "model",
    "pretrained",
    "corruption",
    "severity",
    "prompt_id",
    "prompt_template",
    "method",
    "n_samples",
    *METRIC_COLUMNS,
    "elapsed_seconds",
    "seed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to experiment JSON")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing metrics and manifest instead of resuming",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "data_root",
        "output_dir",
        "models",
        "corruptions",
        "severities",
        "batch_size",
        "num_workers",
        "seed",
        "ece_bins",
        "device",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"config is missing fields: {sorted(missing)}")
    unknown_corruptions = set(config["corruptions"]) - set(STANDARD_CORRUPTIONS)
    if unknown_corruptions:
        raise ValueError(f"unknown/nonstandard corruptions: {sorted(unknown_corruptions)}")
    if not config["models"]:
        raise ValueError("config must contain at least one model")
    if not set(config["severities"]).issubset({1, 2, 3, 4, 5}):
        raise ValueError("severities must be drawn from 1..5")
    if int(config["batch_size"]) <= 0 or int(config["num_workers"]) < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    return config


def set_reproducibility(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def source_hash(project_root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((project_root / "src").rglob("*.py"))
    paths.extend(sorted((project_root / "configs").glob("*.json")))
    for path in paths:
        digest.update(str(path.relative_to(project_root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def gpu_snapshot() -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        result.update(
            {
                "name": torch.cuda.get_device_name(0),
                "total_memory_bytes": int(props.total_memory),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
            }
        )
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=driver_version,power.limit",
            "--format=csv,noheader,nounits",
        ]
        snapshot = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        if snapshot:
            driver, power_limit = [part.strip() for part in snapshot.split(",", maxsplit=1)]
            result["driver_version"] = driver
            result["power_limit_watts"] = float(power_limit)
    except (OSError, subprocess.SubprocessError, ValueError):
        result["nvidia_smi_snapshot"] = "unavailable"
    return result


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("promptfragc")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def completed_conditions(metrics_path: Path) -> set[tuple[str, str, str, int]]:
    if not metrics_path.exists():
        return set()
    counts: dict[tuple[str, str, str, int], int] = {}
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["model"],
                row["pretrained"],
                row["corruption"],
                int(row["severity"]),
            )
            counts[key] = counts.get(key, 0) + 1
    expected = len(PROMPTS) + 1
    return {key for key, count in counts.items() if count >= expected}


def append_rows(metrics_path: Path, rows: list[dict[str, Any]]) -> None:
    existing_rows: list[dict[str, Any]] = []
    if metrics_path.exists() and metrics_path.stat().st_size > 0:
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            existing_rows = list(csv.DictReader(handle))
    temporary = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, metrics_path)


def evaluate_condition(
    bundle,
    text_features,
    config: dict[str, Any],
    corruption: str,
    severity: int,
    artifact_prefix: str,
) -> tuple[list[dict[str, float | str | int]], int]:
    import torch
    from torch.utils.data import DataLoader

    dataset = make_dataset(
        data_root=config["data_root"],
        corruption=corruption,
        severity=severity,
        transform=bundle.preprocess,
        sample_limit=config.get("sample_limit"),
        seed=int(config["seed"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=bundle.device.type == "cuda",
        persistent_workers=int(config["num_workers"]) > 0,
    )

    probability_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    for images, labels in loader:
        probabilities = infer_prompt_probabilities(bundle, text_features, images)
        probability_batches.append(probabilities.detach().cpu().numpy())
        label_batches.append(labels.numpy())
    probabilities = np.concatenate(probability_batches, axis=1)
    targets = np.concatenate(label_batches, axis=0)

    result_rows: list[dict[str, float | str | int]] = []
    for prompt_index, (prompt_id, template) in enumerate(PROMPTS):
        row: dict[str, float | str | int] = {
            "prompt_id": prompt_id,
            "prompt_template": template,
            "method": "single",
        }
        row.update(
            classification_metrics(
                probabilities[prompt_index],
                targets,
                n_bins=int(config["ece_bins"]),
            )
        )
        result_rows.append(row)

    ensemble_probabilities = probabilities.mean(axis=0)
    ensemble_probabilities /= ensemble_probabilities.sum(axis=1, keepdims=True)
    ensemble_row: dict[str, float | str | int] = {
        "prompt_id": "ensemble",
        "prompt_template": "mean probability across all frozen prompts",
        "method": "ensemble",
    }
    ensemble_row.update(
        classification_metrics(
            ensemble_probabilities,
            targets,
            n_bins=int(config["ece_bins"]),
        )
    )
    result_rows.append(ensemble_row)

    if bool(config.get("save_predictions", False)):
        # Prediction archives are condition-local and compact enough for pilot
        # debugging. Full probabilities are intentionally not persisted.
        prediction_dir = Path(config["output_dir"]) / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = prediction_dir / (
            f"{artifact_prefix}_{corruption}_s{severity}.npz"
        )
        confidence = probabilities.max(axis=2).astype(np.float16)
        predictions = probabilities.argmax(axis=2).astype(np.uint8)
        true_probability = np.take_along_axis(
            probabilities,
            targets[None, :, None],
            axis=2,
        ).squeeze(2).astype(np.float16)
        np.savez_compressed(
            prediction_path,
            targets=targets.astype(np.uint8),
            confidence=confidence,
            predictions=predictions,
            true_probability=true_probability,
            prompt_ids=np.asarray([item[0] for item in PROMPTS]),
        )

    del probabilities, probability_batches
    if bundle.device.type == "cuda":
        torch.cuda.empty_cache()
    return result_rows, int(targets.size)


def run(config_path: str | Path, overwrite: bool = False) -> None:
    import open_clip
    import torch

    project_root = Path.cwd()
    config = load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "raw_metrics.csv"
    manifest_path = output_dir / "run_manifest.json"
    current_source_hash = source_hash(project_root)
    if overwrite:
        metrics_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    elif manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("source_sha256") != current_source_hash:
            raise RuntimeError(
                "source code changed since this output directory was created; "
                "use a new output_dir or rerun with --overwrite"
            )
        if previous_manifest.get("config") != config:
            raise RuntimeError(
                "configuration changed since this output directory was created; "
                "use a new output_dir or rerun with --overwrite"
            )

    logger = setup_logger(output_dir)
    set_reproducibility(int(config["seed"]))
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the configuration but is unavailable")

    now = datetime.now(timezone.utc).astimezone().isoformat()
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": now,
        "config_path": str(Path(config_path).resolve()),
        "config": config,
        "source_sha256": current_source_hash,
        "environment_sensitive": True,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": package_version("torchvision"),
            "open_clip_torch": package_version("open_clip_torch"),
            "open_clip_module": getattr(open_clip, "__version__", "unknown"),
            "numpy": np.__version__,
            "gpu": gpu_snapshot(),
        },
    }
    write_json_atomic(manifest_path, manifest)

    completed = completed_conditions(metrics_path)
    conditions = []
    if bool(config.get("include_clean", True)):
        conditions.append(("clean", 0))
    conditions.extend(
        (corruption, int(severity))
        for corruption in config["corruptions"]
        for severity in config["severities"]
    )

    try:
        for model_spec in config["models"]:
            model_name = model_spec["name"]
            pretrained = model_spec["pretrained"]
            artifact_prefix = "".join(
                character if character.isalnum() or character in "._-" else "_"
                for character in f"{model_name}_{pretrained}"
            )
            logger.info("Loading model %s / %s", model_name, pretrained)
            bundle = load_clip_bundle(model_name, pretrained, config["device"])
            text_features = encode_prompt_bank(bundle, CIFAR10_CLASSES, PROMPTS)

            for condition_index, (corruption, severity) in enumerate(conditions, start=1):
                key = (model_name, pretrained, corruption, severity)
                if key in completed:
                    logger.info(
                        "Skipping completed condition %s/%s %s s%d",
                        model_name,
                        pretrained,
                        corruption,
                        severity,
                    )
                    continue
                logger.info(
                    "Condition %d/%d: %s/%s %s s%d",
                    condition_index,
                    len(conditions),
                    model_name,
                    pretrained,
                    corruption,
                    severity,
                )
                started = time.perf_counter()
                metric_rows, n_samples = evaluate_condition(
                    bundle,
                    text_features,
                    config,
                    corruption,
                    severity,
                    artifact_prefix,
                )
                elapsed = time.perf_counter() - started
                rows = []
                for metric_row in metric_rows:
                    rows.append(
                        {
                            "model": model_name,
                            "pretrained": pretrained,
                            "corruption": corruption,
                            "severity": severity,
                            **metric_row,
                            "n_samples": n_samples,
                            "elapsed_seconds": round(elapsed, 4),
                            "seed": int(config["seed"]),
                        }
                    )
                append_rows(metrics_path, rows)
                logger.info(
                    "Completed %s s%d: %d samples in %.1f s",
                    corruption,
                    severity,
                    n_samples,
                    elapsed,
                )

            del text_features, bundle
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        manifest["status"] = "completed"
        manifest["completed_at"] = datetime.now(timezone.utc).astimezone().isoformat()
        write_json_atomic(manifest_path, manifest)
        logger.info("Experiment completed: %s", metrics_path)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failed_at"] = datetime.now(timezone.utc).astimezone().isoformat()
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        write_json_atomic(manifest_path, manifest)
        logger.exception("Experiment failed")
        raise


def main() -> None:
    args = parse_args()
    run(args.config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
