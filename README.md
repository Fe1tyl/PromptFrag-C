# PromptFrag-C

PromptFrag-C studies whether paraphrastic class prompts cause reliability drift
in zero-shot CLIP under common image corruptions. The project is designed for a
Windows laptop with an RTX 5060 8 GB GPU and uses inference only.

## Frozen research question

> How strongly do paraphrastic prompts interact with corruption type and
> severity to affect zero-shot CLIP accuracy, calibration, and selective risk,
> and can leave-one-corruption-out stability selection reduce that drift?

Primary outcomes:

- accuracy (higher is better);
- expected calibration error, ECE (lower is better);
- negative log-likelihood, NLL (lower is better);
- area under the risk-coverage curve, AURC (lower is better).

The study uses the 15 standard CIFAR-10-C corruptions and severities 1--5.
`ViT-B-32/openai` is the primary model; `RN50/openai` is the architecture
replication.

## 1. Install

Open PowerShell in this directory and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

The setup script creates `.venv`, installs the pinned CUDA 12.8 pair
PyTorch 2.9.1 / torchvision 0.24.1 suitable for an RTX 50-series GPU, installs
the remaining dependencies, and writes an environment snapshot under
`outputs/environment/`.

For more reliable access from mainland China, general Python packages use the
Tsinghua TUNA HTTPS PyPI mirror and the large CUDA wheels use the Alibaba Cloud
PyTorch mirror. The CUDA wheel step bypasses pip's cache to prevent reuse of a
truncated download. The two CUDA wheel URLs are explicitly pinned for 64-bit
Python 3.12 on Windows. Sources can be overridden with `-PyPIIndexUrl`,
`-TorchWheelUrl`, and `-TorchVisionWheelUrl`.

## 2. Download CIFAR-10 and CIFAR-10-C

The CIFAR-10-C archive is about 2.9 GB and expands to roughly 3 GB. Clean
CIFAR-10 is about 170 MB:

```powershell
.\.venv\Scripts\python.exe .\scripts\download_cifar10.py --data-root .\data --remove-archive
.\.venv\Scripts\python.exe .\scripts\download_cifar10c.py --data-root .\data --remove-archive
```

The scripts support download resumption, automatically reconnect when a
response ends early, and verify the official MD5 checksums before extraction.
If a network interruption still stops a script, run the same command again;
the partial archive is preserved and resumed. Inference reads both datasets
offline and never starts a dataset download.

## 3. Run the pilot

The simplest option checks CUDA, downloads/verifies the data when needed, runs
the tests, executes the pilot, and generates the analysis:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_pilot.ps1
```

The pilot evaluates one model, three corruptions, three severities, and 1,000
images per condition. The equivalent manual commands are:

```powershell
.\.venv\Scripts\python.exe -m promptfragc.runner --config .\configs\pilot.json
.\.venv\Scripts\python.exe -m promptfragc.analyze --input .\outputs\pilot\raw_metrics.csv --output-dir .\outputs\pilot\analysis
```

The launcher disables Hugging Face Xet/CAS and uses its regular resumable HTTPS
download path for model weights. This avoids file-reconstruction failures on
interrupted or filtered connections without changing the selected checkpoint.

Success means that `raw_metrics.csv`, analysis tables, and PNG figures are
non-empty and the environment check reports CUDA as available.

## 4. Run the full experiment

```powershell
.\.venv\Scripts\python.exe -m promptfragc.runner --config .\configs\full.json
.\.venv\Scripts\python.exe -m promptfragc.analyze --input .\outputs\full\raw_metrics.csv --output-dir .\outputs\full\analysis
```

The runner is resumable: completed model/corruption/severity conditions are
skipped. Use `--overwrite` only when intentionally replacing an existing run.

## 5. Run the revision analyses

The CCSB revision adds two analyses requested during peer review:

1. a three-objective Pareto audit and stability-score weight-sensitivity grid;
2. an external natural-shift evaluation on CIFAR-10.1 v6.

The Pareto audit uses the completed CIFAR-10-C metric table and does not rerun
CLIP inference:

```powershell
.\.venv\Scripts\python.exe .\scripts\analyze_revision.py `
  --input .\outputs\full\raw_metrics.csv `
  --output-dir .\outputs\revision\pareto
```

The objective vector maximizes mean accuracy while minimizing accuracy standard
deviation and mean ECE. The paper's prespecified score,
`mean_accuracy - 0.5 * std_accuracy - 0.2 * mean_ECE`, is retained as one
deployment preference rather than treated as a uniquely optimal definition of
robustness. The sensitivity grid evaluates:

- accuracy-SD weight `alpha` in `{0, 0.25, 0.5, 1}`;
- mean-ECE weight `beta` in `{0, 0.1, 0.2, 0.5, 1}`.

To run the independent natural-shift validation:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_cifar10_1.py `
  --data-dir .\data\CIFAR-10.1 `
  --output-dir .\outputs\revision\cifar10_1 `
  --batch-size 64 `
  --num-workers 0 `
  --device cuda

.\.venv\Scripts\python.exe .\scripts\summarize_cifar10_1.py `
  --metrics .\outputs\revision\cifar10_1\metrics.csv `
  --pareto-summary .\outputs\revision\pareto\paper_pareto_summary.csv `
  --output-dir .\outputs\revision\cifar10_1 `
  --bootstrap-samples 10000 `
  --seed 20260728
```

`evaluate_cifar10_1.py` downloads the two public CIFAR-10.1 v6 NumPy files
(approximately 6 MB in total) from the official dataset repository, validates
their shapes and class balance, and records SHA-256 checksums. It evaluates the
same 12 frozen prompts and two OpenAI-pretrained CLIP backbones used in the main
study.

The prompt is selected exclusively from CIFAR-10-C before CIFAR-10.1 is
evaluated. CIFAR-10.1 is never used to tune the score, choose a prompt, or change
the prompt inventory. The summary script reports paired bootstrap confidence
intervals and an exact McNemar test for the selected prompt versus the default.

### Revision-result checkpoints

The checked revision run produced the following results:

- RN50: the prespecified score selected `p10`; four prompts were selected across
  the 20 weight combinations.
- ViT-B/32: the prespecified score selected `p03`; `p03` was selected at all 20
  weight combinations.
- Every LOCO-selected prompt was Pareto-nondominated in all 15 held-out
  corruption folds for both models.
- On CIFAR-10.1 v6, the 12-prompt accuracy range was 7.90 percentage points for
  RN50 and 4.00 points for ViT-B/32.
- The CIFAR-10-C-selected prompts exceeded the default by 0.25 and 0.80
  percentage points, respectively, but both paired 95% confidence intervals
  included zero.

## Laptop-safe settings

- Keep the laptop connected to power and use the dedicated-GPU/high-performance
  mode.
- The default batch size is 64. Increase to 128 only after the pilot succeeds.
- Close GPU-heavy applications before the full run.
- Do not run the two models in parallel.
- If CUDA reports out-of-memory, stop the process and lower `batch_size` in the
  JSON configuration; the runner does not silently retry.

## Outputs

- `raw_metrics.csv`: one row per model, condition, and prompt/method;
- `run_manifest.json`: configuration, versions, GPU, source hash, and timestamps;
- `analysis/prompt_summary.csv`: prompt-level aggregate performance;
- `analysis/condition_summary.csv`: prompt sensitivity per corruption condition;
- `analysis/interaction_decomposition.csv`: prompt, condition, and interaction
  sums-of-squares shares;
- `analysis/statistical_tests.csv`: Friedman tests and Kendall's W;
- `analysis/corruption_aggregated_tests.csv`: conservative Friedman tests after
  averaging over severity within each corruption, with block-bootstrap
  confidence intervals for Kendall's W and global Holm correction;
- `analysis/loco_selection.csv`: leave-one-corruption-out selection results;
- `analysis/loco_selection_summary.csv`: stability-selection comparison;
- `analysis/policy_comparisons.csv`: paired Wilcoxon policy comparisons with
  bootstrap confidence intervals plus within-model and global Holm correction;
- `analysis/figures/*.png`: paper-ready figures at 300 DPI.
- `revision/pareto/prompt_objectives_and_pareto.csv`: three-objective prompt
  aggregates and Pareto-front membership;
- `revision/pareto/weight_sensitivity_grid.csv`: selections over the 20
  coefficient combinations;
- `revision/pareto/loco_pareto_audit.csv`: fold-wise Pareto audit of LOCO
  selections;
- `revision/cifar10_1/metrics.csv`: CIFAR-10.1 metrics for all prompts and the
  probability ensemble;
- `revision/cifar10_1/paired_accuracy_comparisons.csv`: paired accuracy
  intervals and exact McNemar results;
- `revision/cifar10_1/*_predictions.npz`: compact per-example predictions used
  by the paired comparison.

## Reproducibility rules

- Seed: `20260723`.
- Prompts, corruption list, metrics, and stability-score coefficients are frozen
  before seeing results.
- Full inference is treated as environment-sensitive. Accuracy-like metrics
  should match closely on a repeated run, while wall-clock time is not compared.
- `run_manifest.json` records enough information to diagnose environment drift.
- The original stability score remains fixed as
  `mean_accuracy - 0.5 * std_accuracy - 0.2 * mean_ECE`; the revision reports
  Pareto membership and coefficient sensitivity instead of presenting this
  scalarization as uniquely optimal.
- CIFAR-10.1 is an external validation set and is not used for prompt selection.

See [docs/experiment_plan.md](docs/experiment_plan.md) for the preregistered
design and limitations.
