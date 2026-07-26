"""Compile analysis outputs into paper-ready main and supplementary tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PRIMARY_METRICS = ("accuracy", "ece", "aurc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="outputs/full/raw_metrics.csv")
    parser.add_argument("--analysis-dir", default="outputs/full/analysis")
    parser.add_argument("--output-dir", default="paper/tables")
    return parser.parse_args()


def format_p(value: float) -> str:
    return f"{value:.2e}" if value < 0.001 else f"{value:.4f}"


def format_metric_value(metric: str, value: float) -> str:
    if metric in {"accuracy", "ece"}:
        return f"{100.0 * value:.2f}%"
    return f"{value:.4f}"


def format_delta(metric: str, value: float) -> str:
    if metric in {"accuracy", "ece"}:
        return f"{100.0 * value:+.2f} pp"
    return f"{value:+.4f}"


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [
        [str(value).replace("|", "\\|") for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_experimental_design(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, pretrained), group in raw.groupby(
        ["model", "pretrained"], sort=True
    ):
        singles = group[group["method"] == "single"]
        ensembles = group[group["method"] == "ensemble"]
        corrupted = group[group["corruption"] != "clean"]
        rows.append(
            {
                "model": model,
                "pretrained": pretrained,
                "clean_conditions": int(
                    group[group["corruption"] == "clean"][
                        ["corruption", "severity"]
                    ].drop_duplicates().shape[0]
                ),
                "corruption_severity_conditions": int(
                    corrupted[["corruption", "severity"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "corruption_types": int(corrupted["corruption"].nunique()),
                "severity_levels": int(corrupted["severity"].nunique()),
                "paraphrastic_prompts": int(singles["prompt_id"].nunique()),
                "single_prompt_rows": int(singles.shape[0]),
                "ensemble_rows": int(ensembles.shape[0]),
                "images_per_condition": int(group["n_samples"].mode().iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def build_conservative_prompt_table(
    conservative: pd.DataFrame,
    condition_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in conservative.itertuples(index=False):
        metric = str(item.metric)
        group = condition_summary[
            (condition_summary["model"] == item.model)
            & (condition_summary["pretrained"] == item.pretrained)
        ].copy()
        group["prompt_range"] = (
            group[f"{metric}_max"] - group[f"{metric}_min"]
        )
        maximum = group.sort_values("prompt_range", ascending=False).iloc[0]
        scale = 100.0 if metric in {"accuracy", "ece"} else 1.0
        unit = "percentage points" if scale == 100.0 else "absolute AURC"
        rows.append(
            {
                "model": item.model,
                "pretrained": item.pretrained,
                "metric": metric,
                "n_corruptions": int(item.n_corruptions),
                "n_prompts": int(item.n_prompts),
                "mean_within_condition_prompt_range": float(
                    scale * group["prompt_range"].mean()
                ),
                "maximum_within_condition_prompt_range": float(
                    scale * maximum["prompt_range"]
                ),
                "maximum_range_condition": (
                    f"{maximum['corruption']} severity {int(maximum['severity'])}"
                ),
                "range_unit": unit,
                "friedman_chi_square": float(item.statistic),
                "p_value": float(item.p_value),
                "p_value_holm_global": float(item.p_value_holm_global),
                "kendall_w": float(item.effect_size),
                "kendall_w_ci95_low": float(item.effect_ci95_low),
                "kendall_w_ci95_high": float(item.effect_ci95_high),
                "bootstrap_samples": int(item.bootstrap_samples),
            }
        )
    return pd.DataFrame(rows)


def policy_result(metric: str, delta: float, adjusted_p: float) -> str:
    if adjusted_p >= 0.05:
        return "no significant change"
    if metric == "accuracy":
        return "improved" if delta > 0 else "worsened"
    return "improved" if delta < 0 else "worsened"


def build_loco_table(
    comparisons: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in comparisons.itertuples(index=False):
        group = summary[
            (summary["model"] == item.model)
            & (summary["pretrained"] == item.pretrained)
        ].set_index("policy")
        metric = str(item.metric)
        rows.append(
            {
                "model": item.model,
                "pretrained": item.pretrained,
                "metric": metric,
                "n_corruptions": int(item.n_corruptions),
                "default_mean": float(group.loc["default", f"{metric}_mean"]),
                "stability_loco_mean": float(
                    group.loc["stability_loco", f"{metric}_mean"]
                ),
                "mean_delta_stability_minus_default": float(
                    item.mean_delta_stability_minus_default
                ),
                "bootstrap_ci95_low": float(item.bootstrap_ci95_low),
                "bootstrap_ci95_high": float(item.bootstrap_ci95_high),
                "wilcoxon_p_value": float(item.p_value),
                "p_value_holm_within_model": float(item.p_value_holm),
                "p_value_holm_global": float(item.p_value_holm_global),
                "result_at_global_alpha_0_05": policy_result(
                    metric,
                    float(item.mean_delta_stability_minus_default),
                    float(item.p_value_holm_global),
                ),
            }
        )
    return pd.DataFrame(rows)


def build_selection_frequency(loco: pd.DataFrame) -> pd.DataFrame:
    selected = loco[
        loco["policy"].isin(["default", "clean_selected", "stability_loco"])
    ]
    return (
        selected.groupby(
            ["model", "pretrained", "policy", "selected_prompt_id"], sort=True
        )
        .size()
        .reset_index(name="n_held_out_corruptions")
    )


def build_prompt_extrema(prompt_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, pretrained), group in prompt_summary.groupby(
        ["model", "pretrained"], sort=True
    ):
        for criterion, metric, ascending in (
            ("highest_mean_accuracy", "accuracy_mean", False),
            ("lowest_mean_accuracy", "accuracy_mean", True),
            ("lowest_mean_ece", "ece_mean", True),
            ("lowest_mean_aurc", "aurc_mean", True),
        ):
            item = group.sort_values(metric, ascending=ascending).iloc[0]
            rows.append(
                {
                    "model": model,
                    "pretrained": pretrained,
                    "criterion": criterion,
                    "prompt_id": item["prompt_id"],
                    "prompt_template": item["prompt_template"],
                    "value": float(item[metric]),
                }
            )
    return pd.DataFrame(rows)


def build_markdown(
    design: pd.DataFrame,
    conservative: pd.DataFrame,
    loco: pd.DataFrame,
    interaction: pd.DataFrame,
    policies: pd.DataFrame,
    selections: pd.DataFrame,
    extrema: pd.DataFrame,
) -> str:
    design_display = design[
        [
            "model",
            "corruption_types",
            "severity_levels",
            "corruption_severity_conditions",
            "paraphrastic_prompts",
            "images_per_condition",
        ]
    ].rename(
        columns={
            "model": "Model",
            "corruption_types": "Corruptions",
            "severity_levels": "Severities",
            "corruption_severity_conditions": "Shifted conditions",
            "paraphrastic_prompts": "Prompts",
            "images_per_condition": "Images/condition",
        }
    )

    conservative_display = pd.DataFrame(
        {
            "Model": conservative["model"],
            "Metric": conservative["metric"].str.upper(),
            "Mean prompt range": conservative.apply(
                lambda row: (
                    f"{row['mean_within_condition_prompt_range']:.2f} pp"
                    if row["range_unit"] == "percentage points"
                    else f"{row['mean_within_condition_prompt_range']:.4f}"
                ),
                axis=1,
            ),
            "Maximum (condition)": conservative.apply(
                lambda row: (
                    (
                        f"{row['maximum_within_condition_prompt_range']:.2f} pp "
                        f"({row['maximum_range_condition']})"
                    )
                    if row["range_unit"] == "percentage points"
                    else (
                        f"{row['maximum_within_condition_prompt_range']:.4f} "
                        f"({row['maximum_range_condition']})"
                    )
                ),
                axis=1,
            ),
            "Kendall W [95% CI]": conservative.apply(
                lambda row: (
                    f"{row['kendall_w']:.3f} "
                    f"[{row['kendall_w_ci95_low']:.3f}, "
                    f"{row['kendall_w_ci95_high']:.3f}]"
                ),
                axis=1,
            ),
            "Global Holm p": conservative["p_value_holm_global"].map(format_p),
        }
    )

    loco_display = pd.DataFrame(
        {
            "Model": loco["model"],
            "Metric": loco["metric"].str.upper(),
            "Default": loco.apply(
                lambda row: format_metric_value(row["metric"], row["default_mean"]),
                axis=1,
            ),
            "LOCO stability": loco.apply(
                lambda row: format_metric_value(
                    row["metric"], row["stability_loco_mean"]
                ),
                axis=1,
            ),
            "Delta [95% CI]": loco.apply(
                lambda row: (
                    f"{format_delta(row['metric'], row['mean_delta_stability_minus_default'])} "
                    f"[{format_delta(row['metric'], row['bootstrap_ci95_low'])}, "
                    f"{format_delta(row['metric'], row['bootstrap_ci95_high'])}]"
                ),
                axis=1,
            ),
            "Global Holm p": loco["p_value_holm_global"].map(format_p),
            "Result": loco["result_at_global_alpha_0_05"],
        }
    )

    interaction_display = interaction.copy()
    for column in ("prompt_share", "condition_share", "interaction_share"):
        interaction_display[column] = interaction_display[column].map(
            lambda value: f"{100.0 * value:.2f}%"
        )
    interaction_display = interaction_display[
        ["model", "metric", "prompt_share", "condition_share", "interaction_share"]
    ].rename(
        columns={
            "model": "Model",
            "metric": "Metric",
            "prompt_share": "Prompt share",
            "condition_share": "Condition share",
            "interaction_share": "Interaction share",
        }
    )

    policies_display = policies[
        ["model", "policy", "accuracy_mean", "ece_mean", "aurc_mean"]
    ].copy()
    policies_display["accuracy_mean"] = policies_display["accuracy_mean"].map(
        lambda value: f"{100.0 * value:.2f}%"
    )
    policies_display["ece_mean"] = policies_display["ece_mean"].map(
        lambda value: f"{100.0 * value:.2f}%"
    )
    policies_display["aurc_mean"] = policies_display["aurc_mean"].map(
        lambda value: f"{value:.4f}"
    )
    policies_display.columns = ["Model", "Policy", "Accuracy", "ECE", "AURC"]

    selections_display = selections.rename(
        columns={
            "model": "Model",
            "policy": "Policy",
            "selected_prompt_id": "Prompt",
            "n_held_out_corruptions": "Corruptions selected",
        }
    )[["Model", "Policy", "Prompt", "Corruptions selected"]]

    extrema_display = extrema.copy()
    extrema_display["value"] = extrema_display.apply(
        lambda row: (
            f"{100.0 * row['value']:.2f}%"
            if row["criterion"] != "lowest_mean_aurc"
            else f"{row['value']:.4f}"
        ),
        axis=1,
    )
    extrema_display = extrema_display.rename(
        columns={
            "model": "Model",
            "criterion": "Criterion",
            "prompt_id": "Prompt",
            "prompt_template": "Template",
            "value": "Value",
        }
    )[["Model", "Criterion", "Prompt", "Template", "Value"]]

    return f"""# Paper-Ready Tables

These tables are compiled from `outputs/full/raw_metrics.csv` and the
deterministic analysis artifacts in `outputs/full/analysis`. Values are not
manually transcribed.

## Recommended Main Table I — Experimental Design and Coverage

{markdown_table(design_display)}

## Recommended Main Table II — Conservative Prompt Effects

The five severity levels are averaged within each corruption before testing.
The 15 corruption types are the matched blocks. Kendall's W confidence
intervals use 10,000 corruption-block bootstrap samples. Holm correction is
global across the six model-by-metric tests.

{markdown_table(conservative_display)}

## Recommended Main Table III — LOCO Stability Selection versus Default Prompt

Deltas are stability minus default. Lower ECE and AURC are better. Confidence
intervals are paired corruption-block bootstrap intervals; Wilcoxon p-values
are Holm-adjusted globally across the six model-by-metric comparisons.

{markdown_table(loco_display)}

## Supplementary Table S1 — Variance Decomposition

These balanced descriptive shares use all 75 corruption-severity conditions.
They quantify variation in observed metrics and are not causal variance
components.

{markdown_table(interaction_display)}

## Supplementary Table S2 — Policy-Level Means

{markdown_table(policies_display)}

## Supplementary Table S3 — Selected-Prompt Frequencies

{markdown_table(selections_display)}

## Supplementary Table S4 — Prompt Extrema

Means are over the 75 corrupted conditions.

{markdown_table(extrema_display)}

## Reporting Notes

- Treat Main Table II as the inferential prompt-effect result; the original
  75-block analysis remains a higher-powered sensitivity analysis.
- Interpret LOCO selection as a benchmark protocol that uses labeled data from
  other corruption types, not as an unsupervised deployment method.
- The RN50 AURC mean-difference bootstrap interval excludes zero, but the paired
  Wilcoxon test is not significant after global Holm correction. Report this as
  inconclusive rather than significant.
- For ViT-B/32, LOCO stability selection chose the same prompt as the
  clean-selected policy for every held-out corruption; it does not establish an
  adaptive-selection advantage over a fixed clean-selected prompt.
"""


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw)
    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_path)
    condition_summary = pd.read_csv(analysis_dir / "condition_summary.csv")
    conservative_raw = pd.read_csv(
        analysis_dir / "corruption_aggregated_tests.csv"
    )
    comparisons = pd.read_csv(analysis_dir / "policy_comparisons.csv")
    loco_summary = pd.read_csv(analysis_dir / "loco_selection_summary.csv")
    loco_selection = pd.read_csv(analysis_dir / "loco_selection.csv")
    interaction = pd.read_csv(analysis_dir / "interaction_decomposition.csv")
    prompt_summary = pd.read_csv(analysis_dir / "prompt_summary.csv")

    design = build_experimental_design(raw)
    conservative = build_conservative_prompt_table(
        conservative_raw, condition_summary
    )
    loco = build_loco_table(comparisons, loco_summary)
    selections = build_selection_frequency(loco_selection)
    extrema = build_prompt_extrema(prompt_summary)

    outputs = {
        "table_1_experimental_design.csv": design,
        "table_2_conservative_prompt_effects.csv": conservative,
        "table_3_loco_vs_default.csv": loco,
        "table_s1_variance_decomposition.csv": interaction,
        "table_s2_policy_summary.csv": loco_summary,
        "table_s3_selection_frequency.csv": selections,
        "table_s4_prompt_extrema.csv": extrema,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    markdown = build_markdown(
        design,
        conservative,
        loco,
        interaction,
        loco_summary,
        selections,
        extrema,
    )
    (output_dir / "paper_tables.md").write_text(markdown, encoding="utf-8")
    manifest = {
        "source_raw_metrics": str(raw_path.resolve()),
        "source_analysis_directory": str(analysis_dir.resolve()),
        "generated_tables": [str((output_dir / name).resolve()) for name in outputs],
        "markdown_compilation": str(
            (output_dir / "paper_tables.md").resolve()
        ),
    }
    (output_dir / "table_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
