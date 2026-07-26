# Paper-Ready Tables

These tables are compiled from `outputs/full/raw_metrics.csv` and the
deterministic analysis artifacts in `outputs/full/analysis`. Values are not
manually transcribed.

## Recommended Main Table I — Experimental Design and Coverage

| Model | Corruptions | Severities | Shifted conditions | Prompts | Images/condition |
| --- | --- | --- | --- | --- | --- |
| RN50 | 15 | 5 | 75 | 12 | 10000 |
| ViT-B-32 | 15 | 5 | 75 | 12 | 10000 |

## Recommended Main Table II — Conservative Prompt Effects

The five severity levels are averaged within each corruption before testing.
The 15 corruption types are the matched blocks. Kendall's W confidence
intervals use 10,000 corruption-block bootstrap samples. Holm correction is
global across the six model-by-metric tests.

| Model | Metric | Mean prompt range | Maximum (condition) | Kendall W [95% CI] | Global Holm p |
| --- | --- | --- | --- | --- | --- |
| RN50 | ACCURACY | 11.66 pp | 18.79 pp (zoom_blur severity 4) | 0.904 [0.854, 0.949] | 1.11e-25 |
| RN50 | ECE | 18.55 pp | 29.77 pp (gaussian_noise severity 5) | 0.362 [0.261, 0.615] | 1.01e-08 |
| RN50 | AURC | 0.1152 | 0.1923 (glass_blur severity 1) | 0.819 [0.693, 0.917] | 6.38e-23 |
| ViT-B-32 | ACCURACY | 4.75 pp | 10.66 pp (glass_blur severity 4) | 0.669 [0.598, 0.788] | 4.46e-18 |
| ViT-B-32 | ECE | 9.11 pp | 12.33 pp (shot_noise severity 2) | 0.907 [0.863, 0.953] | 1.08e-25 |
| ViT-B-32 | AURC | 0.0396 | 0.1139 (impulse_noise severity 5) | 0.664 [0.600, 0.784] | 4.46e-18 |

## Recommended Main Table III — LOCO Stability Selection versus Default Prompt

Deltas are stability minus default. Lower ECE and AURC are better. Confidence
intervals are paired corruption-block bootstrap intervals; Wilcoxon p-values
are Holm-adjusted globally across the six model-by-metric comparisons.

| Model | Metric | Default | LOCO stability | Delta [95% CI] | Global Holm p | Result |
| --- | --- | --- | --- | --- | --- | --- |
| RN50 | ACCURACY | 47.29% | 48.32% | +1.03 pp [+0.68 pp, +1.39 pp] | 9.16e-04 | improved |
| RN50 | ECE | 6.16% | 8.10% | +1.93 pp [+0.91 pp, +2.84 pp] | 0.0105 | worsened |
| RN50 | AURC | 0.3205 | 0.3159 | -0.0046 [-0.0083, -0.0007] | 0.0730 | no significant change |
| ViT-B-32 | ACCURACY | 70.73% | 72.02% | +1.28 pp [+0.90 pp, +1.66 pp] | 7.32e-04 | improved |
| ViT-B-32 | ECE | 4.20% | 5.20% | +1.00 pp [+0.44 pp, +1.44 pp] | 0.0205 | worsened |
| ViT-B-32 | AURC | 0.1157 | 0.1099 | -0.0057 [-0.0100, -0.0020] | 0.0128 | improved |

## Supplementary Table S1 — Variance Decomposition

These balanced descriptive shares use all 75 corruption-severity conditions.
They quantify variation in observed metrics and are not causal variance
components.

| Model | Metric | Prompt share | Condition share | Interaction share |
| --- | --- | --- | --- | --- |
| RN50 | accuracy | 4.37% | 94.65% | 0.98% |
| RN50 | ece | 29.24% | 14.07% | 56.69% |
| RN50 | aurc | 3.15% | 95.60% | 1.25% |
| ViT-B-32 | accuracy | 0.55% | 98.90% | 0.55% |
| ViT-B-32 | ece | 51.17% | 36.29% | 12.53% |
| ViT-B-32 | aurc | 0.79% | 98.34% | 0.87% |

## Supplementary Table S2 — Policy-Level Means

| Model | Policy | Accuracy | ECE | AURC |
| --- | --- | --- | --- | --- |
| RN50 | clean_selected | 47.83% | 6.69% | 0.3241 |
| RN50 | default | 47.29% | 6.16% | 0.3205 |
| RN50 | oracle_upper_bound | 50.26% | 12.36% | 0.3060 |
| RN50 | probability_ensemble | 47.04% | 8.35% | 0.3287 |
| RN50 | stability_loco | 48.32% | 8.10% | 0.3159 |
| ViT-B-32 | clean_selected | 72.02% | 5.20% | 0.1099 |
| ViT-B-32 | default | 70.73% | 4.20% | 0.1157 |
| ViT-B-32 | oracle_upper_bound | 72.52% | 6.12% | 0.1078 |
| ViT-B-32 | probability_ensemble | 72.65% | 9.92% | 0.1088 |
| ViT-B-32 | stability_loco | 72.02% | 5.20% | 0.1099 |

## Supplementary Table S3 — Selected-Prompt Frequencies

| Model | Policy | Prompt | Corruptions selected |
| --- | --- | --- | --- |
| RN50 | clean_selected | p04 | 15 |
| RN50 | default | p00 | 15 |
| RN50 | stability_loco | p06 | 2 |
| RN50 | stability_loco | p10 | 13 |
| ViT-B-32 | clean_selected | p03 | 15 |
| ViT-B-32 | default | p00 | 15 |
| ViT-B-32 | stability_loco | p03 | 15 |

## Supplementary Table S4 — Prompt Extrema

Means are over the 75 corrupted conditions.

| Model | Criterion | Prompt | Template | Value |
| --- | --- | --- | --- | --- |
| RN50 | highest_mean_accuracy | p06 | the object depicted here is {label}. | 50.07% |
| RN50 | lowest_mean_accuracy | p11 | {label}, shown in a photograph. | 39.00% |
| RN50 | lowest_mean_ece | p00 | a photo of a {label}. | 6.16% |
| RN50 | lowest_mean_aurc | p10 | a natural image containing {label}. | 0.3139 |
| ViT-B-32 | highest_mean_accuracy | p03 | a picture showing {label}. | 72.02% |
| ViT-B-32 | lowest_mean_accuracy | p08 | an example image of the class {label}. | 68.52% |
| ViT-B-32 | lowest_mean_ece | p00 | a photo of a {label}. | 4.20% |
| ViT-B-32 | lowest_mean_aurc | p04 | this image shows {label}. | 0.1098 |

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
