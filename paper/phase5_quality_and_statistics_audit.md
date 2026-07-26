# Writing Quality and Statistical Language Audit

## Automated Checks

| Check | Result | Gate |
| --- | ---: | --- |
| Main-body length, abstract through conclusion, including index terms but excluding references and declarations | approximately 2,840 whitespace-delimited words after stripping Markdown comments | PASS: within 2,700-3,300 |
| English abstract | 176 words | PASS: 150-300 |
| Traditional Chinese abstract | 316 Han characters, punctuation excluded | PASS: 300-500 |
| Em dashes in main body | 0 | PASS |
| Semicolons in main body | 0 | PASS |
| Unique cited sources | 16 | PASS for a short conference paper |
| Citation orphans | 0 | PASS |
| Placeholder citations | 0 | PASS |
| Banned generic AI-style phrase scan | 0 matches | PASS |

## Statistical Claim Audit

| Manuscript claim | Source artifact | Status |
| --- | --- | --- |
| Mean accuracy prompt ranges are 11.66 pp (RN50) and 4.75 pp (ViT-B/32) | `table_2_conservative_prompt_effects.csv` | MATCH |
| Maximum accuracy ranges are 18.79 pp at zoom blur severity 4 and 10.66 pp at glass blur severity 4 | `table_2_conservative_prompt_effects.csv` | MATCH |
| Six prompt-effect tests survive global Holm correction | `corruption_aggregated_tests.csv` | MATCH |
| Kendall's W values and bootstrap intervals | `corruption_aggregated_tests.csv` | MATCH |
| Prompt, condition, and interaction shares | `table_s1_variance_decomposition.csv` | MATCH |
| LOCO accuracy gains and confidence intervals | `table_3_loco_vs_default.csv` | MATCH |
| LOCO ECE increases and confidence intervals | `table_3_loco_vs_default.csv` | MATCH |
| ViT-B/32 AURC improvement | `table_3_loco_vs_default.csv` | MATCH |
| RN50 AURC classified as inconclusive | Wilcoxon global Holm p=.0730 despite a bootstrap mean interval below zero | CONSERVATIVE |
| RN50 selection frequencies p10=13 and p06=2 | `table_s3_selection_frequency.csv` | MATCH |
| ViT-B/32 LOCO equals clean-selected p03 for all corruptions | `table_s3_selection_frequency.csv` | MATCH |

## Interpretation Guardrails

- H1 is described as supported within the evaluated design, not as proof that
  every prompt affects every VLM.
- H2 is explicitly metric-dependent. The manuscript does not call the
  sums-of-squares shares causal variance components.
- H3 is called partially supported, and its joint "without worsening ECE"
  formulation is rejected.
- Statistical significance is paired with Kendall's W, raw percentage-point
  changes, or AURC differences.
- The manuscript explains that the Wilcoxon test and bootstrap interval for
  RN50 AURC target different summaries and retains the prespecified
  non-significant decision.
- LOCO is identified as label-using and is not presented as unsupervised
  test-time adaptation.
- The ViT-B/32 gain is not presented as an adaptive advantage over the
  clean-selected fixed prompt.

## Reproducibility Checks

- `outputs/full/run_manifest.json` reports `status: completed`.
- The manifest records seed 20260723, both OpenAI-pretrained backbones, all 15
  corruptions, five severities, 10,000 images per condition, package versions,
  GPU information, and a source SHA-256 hash.
- `outputs/full/raw_metrics.csv` contains 1,976 rows.
- The primary tests use 15 corruption blocks and 10,000 block-bootstrap
  samples, consistent with the frozen analysis plan.

## Remaining Human Decisions

1. Supply the author list, affiliations, and corresponding-author details.
2. Confirm funding and conflict-of-interest statements.
3. Supply a public archival repository URL.
4. Confirm whether the venue requires or permits the proposed generative-AI
   disclosure wording.
5. Confirm the official paper template and page limit before typesetting.

## Gate Decision

**PASS as a complete first draft.** The manuscript is ready for author content
review. It is not yet submission-ready because metadata, disclosure wording,
the public repository URL, and official IEEE template compliance remain open.
