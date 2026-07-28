"""Revision analyses: Pareto fronts and stability-score weight sensitivity.

This script uses only the frozen CIFAR-10-C metric table.  It does not rerun
CLIP inference, and it keeps the paper's preregistered score (alpha=0.5,
beta=0.2) as the baseline operating point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


STD_WEIGHTS = (0.0, 0.25, 0.5, 1.0)
ECE_WEIGHTS = (0.0, 0.1, 0.2, 0.5, 1.0)
BASELINE_STD_WEIGHT = 0.5
BASELINE_ECE_WEIGHT = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/full/raw_metrics.csv")
    parser.add_argument("--output-dir", default="outputs/revision/pareto")
    return parser.parse_args()


def aggregate_prompts(frame: pd.DataFrame) -> pd.DataFrame:
    result = (
        frame.groupby("prompt_id", sort=True)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            sd_accuracy=("accuracy", "std"),
            mean_ece=("ece", "mean"),
        )
        .reset_index()
    )
    result["sd_accuracy"] = result["sd_accuracy"].fillna(0.0)
    return result


def pareto_mask(summary: pd.DataFrame) -> np.ndarray:
    """Return nondominated rows for max accuracy, min SD, and min ECE."""
    values = summary[["mean_accuracy", "sd_accuracy", "mean_ece"]].to_numpy(float)
    keep = np.ones(values.shape[0], dtype=bool)
    for index, candidate in enumerate(values):
        accuracy, sd_accuracy, ece = candidate
        weakly_better = (
            (values[:, 0] >= accuracy)
            & (values[:, 1] <= sd_accuracy)
            & (values[:, 2] <= ece)
        )
        strictly_better = (
            (values[:, 0] > accuracy)
            | (values[:, 1] < sd_accuracy)
            | (values[:, 2] < ece)
        )
        if np.any(weakly_better & strictly_better):
            keep[index] = False
    return keep


def score(summary: pd.DataFrame, alpha: float, beta: float) -> pd.Series:
    return (
        summary["mean_accuracy"]
        - alpha * summary["sd_accuracy"]
        - beta * summary["mean_ece"]
    )


def choose(summary: pd.DataFrame, alpha: float, beta: float) -> str:
    scored = summary.assign(stability_score=score(summary, alpha, beta))
    ranked = scored.sort_values(
        ["stability_score", "mean_accuracy", "mean_ece", "prompt_id"],
        ascending=[False, False, True, True],
        kind="stable",
    )
    return str(ranked.iloc[0]["prompt_id"])


def analyze(input_path: Path, output_dir: Path) -> dict[str, object]:
    frame = pd.read_csv(input_path)
    required = {
        "model", "pretrained", "corruption", "severity", "method",
        "prompt_id", "accuracy", "ece",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"input is missing columns: {sorted(missing)}")
    singles = frame[(frame["method"] == "single") & (frame["corruption"] != "clean")]
    if singles.empty:
        raise ValueError("no corrupted single-prompt rows found")

    output_dir.mkdir(parents=True, exist_ok=True)
    front_rows: list[pd.DataFrame] = []
    grid_rows: list[dict[str, object]] = []
    loco_rows: list[dict[str, object]] = []
    paper_rows: list[dict[str, object]] = []

    for (model, pretrained), model_frame in singles.groupby(
        ["model", "pretrained"], sort=True
    ):
        summary = aggregate_prompts(model_frame)
        summary["pareto_nondominated"] = pareto_mask(summary)
        summary["baseline_score"] = score(
            summary, BASELINE_STD_WEIGHT, BASELINE_ECE_WEIGHT
        )
        summary.insert(0, "pretrained", pretrained)
        summary.insert(0, "model", model)
        front_rows.append(summary)

        baseline_prompt = choose(
            summary, BASELINE_STD_WEIGHT, BASELINE_ECE_WEIGHT
        )
        pareto_prompts = summary.loc[
            summary["pareto_nondominated"], "prompt_id"
        ].astype(str).tolist()

        for alpha in STD_WEIGHTS:
            for beta in ECE_WEIGHTS:
                selected = choose(summary, alpha, beta)
                selected_row = summary[summary["prompt_id"] == selected].iloc[0]
                grid_rows.append(
                    {
                        "model": model,
                        "pretrained": pretrained,
                        "sd_weight_alpha": alpha,
                        "ece_weight_beta": beta,
                        "selected_prompt_id": selected,
                        "selected_is_pareto": bool(
                            selected_row["pareto_nondominated"]
                        ),
                        "mean_accuracy": float(selected_row["mean_accuracy"]),
                        "sd_accuracy": float(selected_row["sd_accuracy"]),
                        "mean_ece": float(selected_row["mean_ece"]),
                        "stability_score": float(
                            selected_row["mean_accuracy"]
                            - alpha * selected_row["sd_accuracy"]
                            - beta * selected_row["mean_ece"]
                        ),
                    }
                )

        model_grid = pd.DataFrame(grid_rows)
        model_grid = model_grid[
            (model_grid["model"] == model)
            & (model_grid["pretrained"] == pretrained)
        ]
        baseline_share = float(
            (model_grid["selected_prompt_id"] == baseline_prompt).mean()
        )

        loco_front_count = 0
        loco_baseline_same_count = 0
        corruptions = sorted(model_frame["corruption"].astype(str).unique())
        for held_out in corruptions:
            training = model_frame[model_frame["corruption"] != held_out]
            training_summary = aggregate_prompts(training)
            training_summary["pareto_nondominated"] = pareto_mask(training_summary)
            selected = choose(
                training_summary, BASELINE_STD_WEIGHT, BASELINE_ECE_WEIGHT
            )
            on_front = bool(
                training_summary.loc[
                    training_summary["prompt_id"] == selected,
                    "pareto_nondominated",
                ].iloc[0]
            )
            loco_front_count += int(on_front)
            loco_baseline_same_count += int(selected == baseline_prompt)
            loco_rows.append(
                {
                    "model": model,
                    "pretrained": pretrained,
                    "held_out_corruption": held_out,
                    "selected_prompt_id": selected,
                    "selected_is_pareto": on_front,
                    "matches_all_corruption_selection": selected == baseline_prompt,
                }
            )

        paper_rows.append(
            {
                "model": model,
                "pareto_prompt_ids": ", ".join(pareto_prompts),
                "baseline_selected_prompt": baseline_prompt,
                "baseline_is_pareto": baseline_prompt in pareto_prompts,
                "unique_grid_selections": int(
                    model_grid["selected_prompt_id"].nunique()
                ),
                "baseline_selection_share_20_grid_points": baseline_share,
                "loco_selected_is_pareto_folds": loco_front_count,
                "loco_folds": len(corruptions),
                "loco_matches_all_corruption_selection_folds": (
                    loco_baseline_same_count
                ),
            }
        )

    fronts = pd.concat(front_rows, ignore_index=True)
    grids = pd.DataFrame(grid_rows)
    loco = pd.DataFrame(loco_rows)
    paper = pd.DataFrame(paper_rows)
    fronts.to_csv(output_dir / "prompt_objectives_and_pareto.csv", index=False)
    grids.to_csv(output_dir / "weight_sensitivity_grid.csv", index=False)
    loco.to_csv(output_dir / "loco_pareto_audit.csv", index=False)
    paper.to_csv(output_dir / "paper_pareto_summary.csv", index=False)

    manifest = {
        "input": str(input_path.resolve()),
        "objective_vector": {
            "maximize": ["mean_accuracy"],
            "minimize": ["sd_accuracy", "mean_ece"],
        },
        "baseline_score": "mean_accuracy - 0.5*sd_accuracy - 0.2*mean_ece",
        "weight_grid": {
            "sd_weight_alpha": list(STD_WEIGHTS),
            "ece_weight_beta": list(ECE_WEIGHTS),
            "points": len(STD_WEIGHTS) * len(ECE_WEIGHTS),
        },
        "outputs": [
            "prompt_objectives_and_pareto.csv",
            "weight_sensitivity_grid.csv",
            "loco_pareto_audit.csv",
            "paper_pareto_summary.csv",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = analyze(Path(args.input), Path(args.output_dir))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
