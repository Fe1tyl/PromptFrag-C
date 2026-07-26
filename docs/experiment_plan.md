# Code Experiment Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-23T22:20:00+08:00
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

## Experiment Overview

- **Title**: PromptFrag-C: Prompt-Induced Reliability Drift in Zero-Shot CLIP
  under Common Image Corruptions
- **Objective**: Quantify prompt-by-corruption reliability interactions and test
  whether a preregistered stability-selection rule transfers to an unseen
  corruption type.
- **Primary RQ**: How strongly do paraphrastic class prompts interact with
  corruption type and severity to affect accuracy, calibration, and selective
  risk?
- **Secondary RQ**: Does leave-one-corruption-out stability selection outperform
  the conventional default prompt on held-out corruption types?
- **Hypothesis H1**: Prompt identity has a non-zero matched-condition effect on
  accuracy, ECE, and AURC.
- **Hypothesis H2**: The prompt-by-condition interaction explains a meaningful
  share of metric variation.
- **Hypothesis H3**: Stability selection improves mean held-out accuracy and/or
  AURC relative to the default prompt without materially worsening ECE.
- **Type**: inference benchmark and statistical analysis

## Variables

- Independent variables: model architecture, prompt identity, corruption type,
  corruption severity.
- Dependent variables: accuracy, ECE, NLL, Brier score, AURC, and risk at 80%
  coverage.
- Controls: same class labels, checkpoint family, preprocessing pipeline,
  image subset, seed, batch precision, and metric implementation.
- Potential confounds: prompt grammar, class-name ambiguity, checkpoint training
  data, CIFAR image upscaling, GPU/library nondeterminism, and using labeled
  corruption data for prompt selection.

## Setup

- **Language/Framework**: Python 3.12, PyTorch, OpenCLIP
- **Pilot command**:
  `.\.venv\Scripts\python.exe -m promptfragc.runner --config .\configs\pilot.json`
- **Full command**:
  `.\.venv\Scripts\python.exe -m promptfragc.runner --config .\configs\full.json`
- **Working Directory**:
  `C:\Users\24324\Documents\Codex\2026-07-22\ieee-aann-2026`
- **Dependencies**: see `requirements.txt`
- **Environment**: Windows, NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB VRAM,
  observed GPU power cap 115 W

## Inputs

| Input | Path | Description |
|---|---|---|
| CIFAR-10-C | `data/CIFAR-10-C/` | 15 corruptions, five severities |
| CIFAR-10 test | `data/cifar-10-batches-py/` | Clean reference condition |
| Prompt registry | `src/promptfragc/constants.py` | Frozen paraphrastic prompts |
| Configuration | `configs/*.json` | Frozen execution settings |

## Expected Outputs

| Output | Path | Format | Success Criterion |
|---|---|---|---|
| Raw metrics | `outputs/<run>/raw_metrics.csv` | CSV | Complete unique conditions |
| Manifest | `outputs/<run>/run_manifest.json` | JSON | Versions and source hash present |
| Analysis tables | `outputs/<run>/analysis/*.csv` | CSV | Non-empty and finite metrics |
| Figures | `outputs/<run>/analysis/figures/*.png` | PNG | Non-empty, 300 DPI |
| Log | `outputs/<run>/run.log` | text | No unhandled exception |

## Monitoring Configuration

- **Pilot timeout**: 60 minutes
- **Full timeout**: 12 hours
- **Monitor files**: `run.log`, `raw_metrics.csv`
- **Experiment type override**: analysis/inference benchmark
- **Progress signal**: completed conditions in `run.log`
- **Hard failures**: non-zero exit, CUDA unavailable, invalid dataset checksum,
  missing output, or non-finite metric.

The runner never silently retries an out-of-memory error. Lowering the batch
size requires an explicit configuration change.

## Analysis Plan

- Primary descriptive result: prompt metric range and standard deviation within
  every model/corruption/severity condition.
- Prompt effect test: Friedman test across matched corruption-severity
  conditions; effect size is Kendall's W.
- Conservative sensitivity analysis: average the five severities within each
  corruption and repeat the matched-prompt Friedman test using only the 15
  corruption types as blocks. Report Kendall's W with a corruption-block
  bootstrap 95% confidence interval and apply Holm correction across the six
  model-by-metric tests.
- Interaction result: balanced sums-of-squares decomposition into prompt main,
  condition main, and prompt-by-condition interaction shares.
- Selection evaluation: leave one corruption type out, select the prompt on the
  other 14 types using
  `mean_accuracy - 0.5*std_accuracy - 0.2*mean_ECE`, then evaluate on all five
  severities of the held-out type.
- Baselines: default prompt, clean-selected prompt, probability ensemble, and
  held-out oracle upper bound.
- Multiple comparisons: Holm adjustment where pairwise policy tests are used.
- Practical importance: report raw differences and confidence intervals, not
  only p-values.

## Scope and Limitations

- This is an empirical reliability study, not a claim of universal CLIP
  robustness.
- CIFAR-10-C is synthetic corruption shift; conclusions may not generalize to
  natural distribution shift.
- The stability selector uses labeled corruption data from other corruption
  types. It is a benchmark-selection protocol, not a fully unsupervised
  deployment method.
- Prompts are frozen before results, but their linguistic equivalence is not
  established by a human study.
- Timing is hardware-sensitive and is not a primary scientific outcome.
