"""Statistical summaries and paper-ready figures for PromptFrag-C."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .constants import DEFAULT_PROMPT_ID, METRIC_COLUMNS, PROMPTS

PROMPT_IDS = [item[0] for item in PROMPTS]
PRIMARY_METRICS = ("accuracy", "ece", "aurc")
STABILITY_STD_WEIGHT = 0.5
STABILITY_ECE_WEIGHT = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="raw_metrics.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in frame.columns
    ]
    return frame.reset_index()


def validate_input(frame: pd.DataFrame) -> None:
    required = {
        "model",
        "pretrained",
        "corruption",
        "severity",
        "prompt_id",
        "prompt_template",
        "method",
        "n_samples",
        *METRIC_COLUMNS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"raw metrics missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("raw metrics file is empty")
    if frame[list(METRIC_COLUMNS)].isna().any().any():
        raise ValueError("raw metrics contain missing metric values")
    finite = np.isfinite(frame[list(METRIC_COLUMNS)].to_numpy(dtype=float))
    if not finite.all():
        raise ValueError("raw metrics contain non-finite values")


def build_prompt_summary(singles: pd.DataFrame) -> pd.DataFrame:
    corrupted = singles[singles["corruption"] != "clean"]
    summary = corrupted.groupby(
        ["model", "pretrained", "prompt_id", "prompt_template"],
        sort=True,
    )[list(METRIC_COLUMNS)].agg(["mean", "std", "min", "max"])
    return flatten_columns(summary)


def build_condition_summary(singles: pd.DataFrame) -> pd.DataFrame:
    corrupted = singles[singles["corruption"] != "clean"]
    summary = corrupted.groupby(
        ["model", "pretrained", "corruption", "severity"],
        sort=True,
    )[list(METRIC_COLUMNS)].agg(["mean", "std", "min", "max"])
    return flatten_columns(summary)


def interaction_decomposition(singles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    corrupted = singles[singles["corruption"] != "clean"].copy()
    corrupted["condition"] = (
        corrupted["corruption"].astype(str)
        + "_s"
        + corrupted["severity"].astype(int).astype(str)
    )
    for (model, pretrained), group in corrupted.groupby(["model", "pretrained"]):
        for metric in PRIMARY_METRICS:
            pivot = group.pivot_table(
                index="prompt_id",
                columns="condition",
                values=metric,
                aggfunc="mean",
            ).reindex(PROMPT_IDS)
            pivot = pivot.dropna(axis=1)
            values = pivot.to_numpy(dtype=float)
            if values.shape[0] != len(PROMPT_IDS) or values.shape[1] < 2:
                continue
            grand = values.mean()
            prompt_means = values.mean(axis=1, keepdims=True)
            condition_means = values.mean(axis=0, keepdims=True)
            interaction = values - prompt_means - condition_means + grand
            ss_prompt = values.shape[1] * float(np.square(prompt_means - grand).sum())
            ss_condition = values.shape[0] * float(
                np.square(condition_means - grand).sum()
            )
            ss_interaction = float(np.square(interaction).sum())
            ss_total = float(np.square(values - grand).sum())
            denominator = ss_total if ss_total > 0 else 1.0
            rows.append(
                {
                    "model": model,
                    "pretrained": pretrained,
                    "metric": metric,
                    "n_prompts": values.shape[0],
                    "n_conditions": values.shape[1],
                    "ss_total": ss_total,
                    "ss_prompt": ss_prompt,
                    "ss_condition": ss_condition,
                    "ss_interaction": ss_interaction,
                    "prompt_share": ss_prompt / denominator,
                    "condition_share": ss_condition / denominator,
                    "interaction_share": ss_interaction / denominator,
                }
            )
    return pd.DataFrame(rows)


def friedman_tests(singles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    corrupted = singles[singles["corruption"] != "clean"].copy()
    corrupted["condition"] = (
        corrupted["corruption"].astype(str)
        + "_s"
        + corrupted["severity"].astype(int).astype(str)
    )
    for (model, pretrained), group in corrupted.groupby(["model", "pretrained"]):
        for metric in PRIMARY_METRICS:
            pivot = group.pivot_table(
                index="condition",
                columns="prompt_id",
                values=metric,
                aggfunc="mean",
            )
            available_prompts = [item for item in PROMPT_IDS if item in pivot.columns]
            pivot = pivot[available_prompts].dropna()
            if pivot.shape[0] < 2 or pivot.shape[1] < 3:
                continue
            samples = [pivot[column].to_numpy(dtype=float) for column in pivot.columns]
            values = pivot.to_numpy(dtype=float)
            if np.allclose(np.ptp(values, axis=1), 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                statistic, p_value = stats.friedmanchisquare(*samples)
            kendalls_w = float(statistic) / (
                float(pivot.shape[0]) * float(pivot.shape[1] - 1)
            )
            rows.append(
                {
                    "test": "Friedman matched-condition prompt effect",
                    "model": model,
                    "pretrained": pretrained,
                    "metric": metric,
                    "n_conditions": int(pivot.shape[0]),
                    "n_prompts": int(pivot.shape[1]),
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "effect_name": "Kendall_W",
                    "effect_size": kendalls_w,
                }
            )
    return pd.DataFrame(rows)


def metric_means(frame: pd.DataFrame) -> dict[str, float]:
    return {metric: float(frame[metric].mean()) for metric in METRIC_COLUMNS}


def choose_prompt(training: pd.DataFrame) -> str:
    aggregate = training.groupby("prompt_id").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        ece_mean=("ece", "mean"),
    )
    aggregate["accuracy_std"] = aggregate["accuracy_std"].fillna(0.0)
    aggregate["stability_score"] = (
        aggregate["accuracy_mean"]
        - STABILITY_STD_WEIGHT * aggregate["accuracy_std"]
        - STABILITY_ECE_WEIGHT * aggregate["ece_mean"]
    )
    return str(aggregate["stability_score"].idxmax())


def build_loco_selection(frame: pd.DataFrame) -> pd.DataFrame:
    singles = frame[(frame["method"] == "single") & (frame["corruption"] != "clean")]
    ensembles = frame[
        (frame["method"] == "ensemble") & (frame["corruption"] != "clean")
    ]
    clean = frame[(frame["method"] == "single") & (frame["corruption"] == "clean")]
    rows: list[dict[str, float | str | int]] = []

    for (model, pretrained), model_group in singles.groupby(["model", "pretrained"]):
        corruptions = sorted(model_group["corruption"].unique())
        clean_group = clean[
            (clean["model"] == model) & (clean["pretrained"] == pretrained)
        ]
        if clean_group.empty:
            clean_best = DEFAULT_PROMPT_ID
        else:
            clean_score = clean_group.set_index("prompt_id")["accuracy"] - (
                STABILITY_ECE_WEIGHT * clean_group.set_index("prompt_id")["ece"]
            )
            clean_best = str(clean_score.idxmax())

        for held_out in corruptions:
            training = model_group[model_group["corruption"] != held_out]
            test = model_group[model_group["corruption"] == held_out]
            stable_prompt = choose_prompt(training)
            oracle_scores = test.groupby("prompt_id").agg(
                accuracy=("accuracy", "mean"),
                ece=("ece", "mean"),
            )
            oracle_prompt = str(
                (oracle_scores["accuracy"] - STABILITY_ECE_WEIGHT * oracle_scores["ece"])
                .idxmax()
            )
            policies = (
                ("default", DEFAULT_PROMPT_ID),
                ("stability_loco", stable_prompt),
                ("clean_selected", clean_best),
                ("oracle_upper_bound", oracle_prompt),
            )
            for policy, prompt_id in policies:
                evaluated = test[test["prompt_id"] == prompt_id]
                row: dict[str, float | str | int] = {
                    "model": model,
                    "pretrained": pretrained,
                    "held_out_corruption": held_out,
                    "policy": policy,
                    "selected_prompt_id": prompt_id,
                    "n_severities": int(evaluated["severity"].nunique()),
                }
                row.update(metric_means(evaluated))
                rows.append(row)

            ensemble_test = ensembles[
                (ensembles["model"] == model)
                & (ensembles["pretrained"] == pretrained)
                & (ensembles["corruption"] == held_out)
            ]
            if not ensemble_test.empty:
                ensemble_row: dict[str, float | str | int] = {
                    "model": model,
                    "pretrained": pretrained,
                    "held_out_corruption": held_out,
                    "policy": "probability_ensemble",
                    "selected_prompt_id": "ensemble",
                    "n_severities": int(ensemble_test["severity"].nunique()),
                }
                ensemble_row.update(metric_means(ensemble_test))
                rows.append(ensemble_row)
    return pd.DataFrame(rows)


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def _friedman_statistic_and_w(values: np.ndarray) -> tuple[float, float, float]:
    """Return Friedman chi-square, p-value, and Kendall's W for a block matrix."""
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 3:
        return math.nan, math.nan, math.nan
    if np.allclose(np.ptp(values, axis=1), 0.0):
        return 0.0, 1.0, 0.0
    statistic, p_value = stats.friedmanchisquare(
        *(values[:, index] for index in range(values.shape[1]))
    )
    kendalls_w = float(statistic) / (
        float(values.shape[0]) * float(values.shape[1] - 1)
    )
    return float(statistic), float(p_value), kendalls_w


def corruption_aggregated_friedman_tests(
    singles: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Conservative prompt tests using corruption type, not severity, as the block.

    Each prompt/metric is first averaged over all available severities within a
    corruption type. Kendall's W intervals then resample whole corruption
    blocks, preserving the matched-prompt structure.
    """
    rows: list[dict[str, float | str | int]] = []
    corrupted = singles[singles["corruption"] != "clean"].copy()
    grouped = (
        corrupted.groupby(
            ["model", "pretrained", "corruption", "prompt_id"],
            sort=True,
        )[list(PRIMARY_METRICS)]
        .mean()
        .reset_index()
    )
    for group_index, ((model, pretrained), group) in enumerate(
        grouped.groupby(["model", "pretrained"], sort=True)
    ):
        n_severities = int(
            corrupted[
                (corrupted["model"] == model)
                & (corrupted["pretrained"] == pretrained)
            ]["severity"].nunique()
        )
        for metric_index, metric in enumerate(PRIMARY_METRICS):
            pivot = group.pivot_table(
                index="corruption",
                columns="prompt_id",
                values=metric,
                aggfunc="mean",
            )
            available_prompts = [item for item in PROMPT_IDS if item in pivot.columns]
            pivot = pivot[available_prompts].dropna()
            values = pivot.to_numpy(dtype=float)
            statistic, p_value, kendalls_w = _friedman_statistic_and_w(values)
            if not np.isfinite(kendalls_w):
                continue

            rng = np.random.default_rng(
                seed + group_index * len(PRIMARY_METRICS) + metric_index
            )
            bootstrap_w = np.empty(bootstrap_samples, dtype=float)
            for sample_index in range(bootstrap_samples):
                indices = rng.integers(0, values.shape[0], size=values.shape[0])
                _, _, sampled_w = _friedman_statistic_and_w(values[indices])
                bootstrap_w[sample_index] = sampled_w
            ci_low, ci_high = np.quantile(bootstrap_w, [0.025, 0.975])
            rows.append(
                {
                    "test": "Friedman corruption-aggregated prompt effect",
                    "model": model,
                    "pretrained": pretrained,
                    "metric": metric,
                    "n_corruptions": int(pivot.shape[0]),
                    "n_severities_aggregated": n_severities,
                    "n_prompts": int(pivot.shape[1]),
                    "statistic": statistic,
                    "p_value": p_value,
                    "effect_name": "Kendall_W",
                    "effect_size": kendalls_w,
                    "effect_ci95_low": float(ci_low),
                    "effect_ci95_high": float(ci_high),
                    "bootstrap_samples": int(bootstrap_samples),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_value_holm_global"] = holm_adjust(
            result["p_value"].astype(float).tolist()
        )
    return result


def paired_bootstrap_difference(
    stable: np.ndarray,
    baseline: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    differences = np.asarray(stable, dtype=float) - np.asarray(baseline, dtype=float)
    if differences.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(samples, differences.size))
    bootstrap = differences[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return float(differences.mean()), float(lower), float(upper)


def compare_selection_policies(
    loco: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for (model, pretrained), group in loco.groupby(["model", "pretrained"]):
        model_rows: list[dict[str, float | str | int]] = []
        for metric_index, metric in enumerate(PRIMARY_METRICS):
            pivot = group.pivot_table(
                index="held_out_corruption",
                columns="policy",
                values=metric,
                aggfunc="mean",
            )
            if not {"stability_loco", "default"}.issubset(pivot.columns):
                continue
            paired = pivot[["stability_loco", "default"]].dropna()
            stable = paired["stability_loco"].to_numpy(dtype=float)
            baseline = paired["default"].to_numpy(dtype=float)
            if np.allclose(stable, baseline):
                statistic, p_value = 0.0, 1.0
            else:
                statistic, p_value = stats.wilcoxon(
                    stable,
                    baseline,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
            delta, ci_low, ci_high = paired_bootstrap_difference(
                stable,
                baseline,
                samples=bootstrap_samples,
                seed=seed + metric_index,
            )
            model_rows.append(
                {
                    "test": "Wilcoxon stability_loco vs default",
                    "model": model,
                    "pretrained": pretrained,
                    "metric": metric,
                    "n_corruptions": int(paired.shape[0]),
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "mean_delta_stability_minus_default": delta,
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                }
            )
        adjusted = holm_adjust([float(row["p_value"]) for row in model_rows])
        for row, adjusted_p in zip(model_rows, adjusted):
            row["p_value_holm"] = adjusted_p
            rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_value_holm_global"] = holm_adjust(
            result["p_value"].astype(float).tolist()
        )
    return result


def build_loco_summary(loco: pd.DataFrame) -> pd.DataFrame:
    if loco.empty:
        return pd.DataFrame()
    summary = loco.groupby(
        ["model", "pretrained", "policy"],
        sort=True,
    )[list(METRIC_COLUMNS)].agg(["mean", "std"])
    return flatten_columns(summary)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def generate_figures(
    frame: pd.DataFrame,
    prompt_summary: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    corrupted = frame[frame["corruption"] != "clean"].copy()

    for (model, pretrained), model_frame in corrupted.groupby(["model", "pretrained"]):
        label = f"{model}_{pretrained}"
        singles = model_frame[model_frame["method"] == "single"]
        ensembles = model_frame[model_frame["method"] == "ensemble"]

        heatmap = singles.pivot_table(
            index="prompt_id",
            columns="corruption",
            values="accuracy",
            aggfunc="mean",
        ).reindex(PROMPT_IDS)
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(heatmap, cmap="viridis", ax=ax, cbar_kws={"label": "Accuracy"})
        ax.set_title(f"Prompt × corruption accuracy — {model}/{pretrained}")
        ax.set_xlabel("Corruption")
        ax.set_ylabel("Prompt")
        fig.tight_layout()
        path = figure_dir / f"{safe_filename(label)}_accuracy_heatmap.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))

        summary_group = prompt_summary[
            (prompt_summary["model"] == model)
            & (prompt_summary["pretrained"] == pretrained)
        ]
        best_prompt = str(
            summary_group.sort_values("accuracy_mean", ascending=False).iloc[0][
                "prompt_id"
            ]
        )
        selected = singles[
            singles["prompt_id"].isin({DEFAULT_PROMPT_ID, best_prompt})
        ].copy()
        selected["series"] = selected["prompt_id"]
        ensemble_lines = ensembles.copy()
        ensemble_lines["series"] = "ensemble"
        line_data = pd.concat([selected, ensemble_lines], ignore_index=True)
        line_data = line_data[line_data["severity"] > 0]
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        sns.lineplot(
            data=line_data,
            x="severity",
            y="accuracy",
            hue="series",
            marker="o",
            errorbar=("ci", 95),
            ax=ax,
        )
        ax.set_title(f"Accuracy versus corruption severity — {model}/{pretrained}")
        ax.set_ylim(0.0, 1.0)
        fig.tight_layout()
        path = figure_dir / f"{safe_filename(label)}_severity_accuracy.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))

        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        sns.scatterplot(
            data=summary_group,
            x="ece_mean",
            y="accuracy_mean",
            hue="prompt_id",
            s=80,
            ax=ax,
            legend=False,
        )
        for _, row in summary_group.iterrows():
            ax.annotate(
                row["prompt_id"],
                (row["ece_mean"], row["accuracy_mean"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_title(f"Prompt accuracy–calibration trade-off — {model}/{pretrained}")
        ax.set_xlabel("Mean ECE (lower is better)")
        ax.set_ylabel("Mean accuracy (higher is better)")
        fig.tight_layout()
        path = figure_dir / f"{safe_filename(label)}_accuracy_ece_scatter.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))

        ranges = (
            singles.groupby(["corruption", "severity"])["accuracy"]
            .agg(lambda values: float(values.max() - values.min()))
            .groupby("corruption")
            .mean()
            .sort_values(ascending=False)
            .reset_index(name="accuracy_range")
        )
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        sns.barplot(data=ranges, y="corruption", x="accuracy_range", ax=ax)
        ax.set_title(f"Prompt-induced accuracy range — {model}/{pretrained}")
        ax.set_xlabel("Mean within-severity max–min accuracy across prompts")
        ax.set_ylabel("")
        fig.tight_layout()
        path = figure_dir / f"{safe_filename(label)}_prompt_accuracy_range.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))
    return written


def analyze(
    input_path: str | Path,
    output_dir: str | Path,
    bootstrap_samples: int = 10_000,
    seed: int = 20260723,
) -> dict[str, object]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_path)
    validate_input(frame)

    singles = frame[frame["method"] == "single"].copy()
    prompt_summary = build_prompt_summary(singles)
    condition_summary = build_condition_summary(singles)
    interaction = interaction_decomposition(singles)
    friedman = friedman_tests(singles)
    conservative_friedman = corruption_aggregated_friedman_tests(
        singles,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    loco = build_loco_selection(frame)
    loco_summary = build_loco_summary(loco)
    policy_tests = compare_selection_policies(
        loco,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )

    outputs = {
        "prompt_summary.csv": prompt_summary,
        "condition_summary.csv": condition_summary,
        "interaction_decomposition.csv": interaction,
        "statistical_tests.csv": friedman,
        "corruption_aggregated_tests.csv": conservative_friedman,
        "loco_selection.csv": loco,
        "loco_selection_summary.csv": loco_summary,
        "policy_comparisons.csv": policy_tests,
    }
    for filename, table in outputs.items():
        table.to_csv(output_dir / filename, index=False)

    figures = generate_figures(frame, prompt_summary, output_dir)
    summary = {
        "input": str(input_path.resolve()),
        "rows": int(frame.shape[0]),
        "models": sorted(frame["model"].unique().tolist()),
        "corruptions": sorted(
            value for value in frame["corruption"].unique().tolist() if value != "clean"
        ),
        "prompt_count": int(singles["prompt_id"].nunique()),
        "stability_score": {
            "formula": "mean_accuracy - 0.5*std_accuracy - 0.2*mean_ece",
            "std_weight": STABILITY_STD_WEIGHT,
            "ece_weight": STABILITY_ECE_WEIGHT,
        },
        "tables": [str(output_dir / name) for name in outputs],
        "figures": figures,
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = analyze(
        args.input,
        args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
