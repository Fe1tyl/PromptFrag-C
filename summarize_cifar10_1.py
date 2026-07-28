"""Summarize CIFAR-10.1 validation without reusing it for prompt selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from promptfragc.constants import DEFAULT_PROMPT_ID, PROMPTS


PROMPT_IDS = [prompt_id for prompt_id, _ in PROMPTS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="outputs/revision/cifar10_1/metrics.csv")
    parser.add_argument(
        "--pareto-summary",
        default="outputs/revision/pareto/paper_pareto_summary.csv",
    )
    parser.add_argument("--output-dir", default="outputs/revision/cifar10_1")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def paired_accuracy_bootstrap(
    selected_correct: np.ndarray,
    default_correct: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    differences = selected_correct.astype(float) - default_correct.astype(float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(samples, differences.size))
    estimates = differences[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(differences.mean()), float(low), float(high)


def analyze(
    metrics_path: Path,
    pareto_path: Path,
    output_dir: Path,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    metrics = pd.read_csv(metrics_path)
    selections = pd.read_csv(pareto_path).set_index("model")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []

    for model_index, (model, model_metrics) in enumerate(
        metrics.groupby("model", sort=True)
    ):
        stable_prompt = str(selections.loc[model, "baseline_selected_prompt"])
        singles = model_metrics[model_metrics["method"] == "single"].copy()
        oracle_prompt = str(
            singles.sort_values(
                ["accuracy", "ece", "prompt_id"],
                ascending=[False, True, True],
                kind="stable",
            ).iloc[0]["prompt_id"]
        )
        roles = {
            "default": DEFAULT_PROMPT_ID,
            "cifar10c_selected": stable_prompt,
            "external_oracle_descriptive": oracle_prompt,
            "probability_ensemble": "ensemble",
        }
        for role, prompt_id in roles.items():
            selected = model_metrics[model_metrics["prompt_id"] == prompt_id]
            if selected.empty:
                continue
            row = selected.iloc[0]
            summary_rows.append(
                {
                    "model": model,
                    "role": role,
                    "prompt_id": prompt_id,
                    "n_samples": int(row["n_samples"]),
                    "accuracy": float(row["accuracy"]),
                    "ece": float(row["ece"]),
                    "aurc": float(row["aurc"]),
                    "nll": float(row["nll"]),
                    "brier": float(row["brier"]),
                }
            )

        prediction_path = output_dir / f"{model}_openai_predictions.npz"
        arrays = np.load(prediction_path)
        targets = arrays["targets"]
        predicted = arrays["predicted_classes"]
        stable_index = PROMPT_IDS.index(stable_prompt)
        default_index = PROMPT_IDS.index(DEFAULT_PROMPT_ID)
        stable_correct = predicted[stable_index] == targets
        default_correct = predicted[default_index] == targets
        delta, ci_low, ci_high = paired_accuracy_bootstrap(
            stable_correct,
            default_correct,
            samples=bootstrap_samples,
            seed=seed + model_index,
        )
        stable_only = int(np.sum(stable_correct & ~default_correct))
        default_only = int(np.sum(default_correct & ~stable_correct))
        discordant = stable_only + default_only
        if discordant:
            mcnemar_p = float(
                stats.binomtest(
                    stable_only, discordant, p=0.5, alternative="two-sided"
                ).pvalue
            )
        else:
            mcnemar_p = 1.0
        comparison_rows.append(
            {
                "model": model,
                "selected_prompt_id": stable_prompt,
                "default_prompt_id": DEFAULT_PROMPT_ID,
                "accuracy_delta_selected_minus_default": delta,
                "paired_bootstrap_ci95_low": ci_low,
                "paired_bootstrap_ci95_high": ci_high,
                "selected_correct_default_wrong": stable_only,
                "default_correct_selected_wrong": default_only,
                "exact_mcnemar_p": mcnemar_p,
                "prompt_accuracy_min": float(singles["accuracy"].min()),
                "prompt_accuracy_max": float(singles["accuracy"].max()),
                "prompt_accuracy_range": float(
                    singles["accuracy"].max() - singles["accuracy"].min()
                ),
                "external_oracle_prompt_id": oracle_prompt,
            }
        )

    summary = pd.DataFrame(summary_rows)
    comparisons = pd.DataFrame(comparison_rows)
    summary.to_csv(output_dir / "paper_external_validation.csv", index=False)
    comparisons.to_csv(output_dir / "paired_accuracy_comparisons.csv", index=False)
    manifest = {
        "selection_rule": (
            "Prompt selected only from CIFAR-10-C using the preregistered score; "
            "CIFAR-10.1 was not used for selection."
        ),
        "bootstrap_samples": bootstrap_samples,
        "paired_test": "exact McNemar/binomial test on discordant predictions",
        "outputs": [
            "paper_external_validation.csv",
            "paired_accuracy_comparisons.csv",
        ],
    }
    (output_dir / "summary_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = analyze(
        Path(args.metrics),
        Path(args.pareto_summary),
        Path(args.output_dir),
        args.bootstrap_samples,
        args.seed,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
