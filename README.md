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

## Reproducibility rules

- Seed: `20260723`.
- Prompts, corruption list, metrics, and stability-score coefficients are frozen
  before seeing results.
- Full inference is treated as environment-sensitive. Accuracy-like metrics
  should match closely on a repeated run, while wall-clock time is not compared.
- `run_manifest.json` records enough information to diagnose environment drift.
- The stability score is fixed as:
  `mean_accuracy - 0.5 * std_accuracy - 0.2 * mean_ECE`.

See [docs/experiment_plan.md](docs/experiment_plan.md) for the preregistered
design and limitations.
