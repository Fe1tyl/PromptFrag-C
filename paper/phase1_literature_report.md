# Phase 1 Literature Report

## Search Record

- Search date: 2026-07-25
- Scope: zero-shot CLIP prompting, prompt learning and test-time prompt
  adaptation, common-corruption robustness, confidence calibration, selective
  prediction, and the nonparametric procedures used in this study.
- Search strategy: exact-title and topic searches were followed by verification
  against publisher, conference-proceedings, OpenReview, PMLR, JMLR, or
  institutional repository records.
- Inclusion rule: primary research directly supporting the motivation,
  methodological choices, comparison with prior work, or statistical analysis.
- Exclusion rule: secondary summaries were not used as evidentiary sources.
  Indexing pages were used only to locate or cross-check bibliographic metadata.
- Title check: an exact search for "PromptFrag-C" and "Prompt-Induced
  Reliability Drift" did not identify a conflicting academic method or
  benchmark as of the search date. This is not a trademark clearance.

## Literature Synthesis

CLIP established the zero-shot image-classification setting in which natural
language descriptions synthesize class representations. Its reported use of
prompt templates and prompt ensembles makes textual formulation part of the
inference pipeline rather than a purely cosmetic choice. CoOp and CoCoOp then
showed that prompt context can be optimized and that prompt generalization can
fail across unseen classes. More recent methods, including PromptAlign,
Any-Shift Prompting, WATT, Frolic, and Mint, explicitly address distribution
shift or corruption through learned, test-time, or distributional adaptation.
These studies motivate prompt adaptation, but they do not replace a controlled
diagnostic of how frozen paraphrastic templates change multiple reliability
metrics under the same shifted inputs.

The reliability literature provides three complementary outcomes. Accuracy
measures correctness, ECE summarizes confidence calibration, and AURC
summarizes selective risk over coverage levels. Common-corruption benchmarks
make it possible to evaluate these outcomes under standardized visual
degradation. Prior work also shows that distribution shift can damage both
accuracy and calibration, while ECE and other scalar calibration metrics have
known estimation and binning limitations. These results support a
multi-metric design and cautious interpretation rather than a single
"robustness" score.

## Verified Source Map

| ID | Source and verified venue | Direct use in the paper | Persistent identifier |
| --- | --- | --- | --- |
| LIT-01 | A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," ICML, 2021 | CLIP architecture, zero-shot transfer, natural-language class descriptions | https://proceedings.mlr.press/v139/radford21a |
| LIT-02 | K. Zhou et al., "Learning to Prompt for Vision-Language Models," IJCV, 2022 | Prompt wording is consequential; CoOp as learned context | https://doi.org/10.1007/s11263-022-01653-1 |
| LIT-03 | K. Zhou et al., "Conditional Prompt Learning for Vision-Language Models," CVPR, 2022 | CoOp overfitting and conditional prompts for generalization | https://doi.org/10.1109/CVPR52688.2022.01631 |
| LIT-04 | D. Hendrycks and T. Dietterich, "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations," ICLR, 2019 | Common-corruption benchmark rationale and CIFAR-10-C provenance | https://iclr.cc/virtual/2019/poster/731 |
| LIT-05 | Y. Ovadia et al., "Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift," NeurIPS, 2019 | Accuracy and calibration can both degrade under shift | https://proceedings.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html |
| LIT-06 | C. Guo et al., "On Calibration of Modern Neural Networks," ICML, 2017 | Definition and practical importance of confidence calibration and ECE | https://proceedings.mlr.press/v70/guo17a |
| LIT-07 | Y. Ding et al., "Revisiting the Evaluation of Uncertainty Estimation and Its Application to Explore Model Complexity-Uncertainty Trade-Off," CVPR Workshops, 2020 | AURC and limitations of common uncertainty metrics | https://doi.org/10.1109/CVPRW50498.2020.00010 |
| LIT-08 | Y. Geifman and R. El-Yaniv, "SelectiveNet: A Deep Neural Network with an Integrated Reject Option," ICML, 2019 | Risk-coverage interpretation of selective prediction | https://proceedings.mlr.press/v97/geifman19a |
| LIT-09 | J. A. Samadh et al., "Align Your Prompts: Test-Time Prompting with Distribution Alignment for Zero-Shot Generalization," NeurIPS, 2023 | Test-time prompt adaptation under distribution shift | https://doi.org/10.52202/075280-3525 |
| LIT-10 | Z. Xiao et al., "Any-Shift Prompting for Generalization over Distributions," CVPR, 2024 | Prompt learning designed for multiple forms of distribution shift | https://doi.org/10.1109/CVPR52733.2024.01314 |
| LIT-11 | D. Osowiechi et al., "WATT: Weight Average Test Time Adaptation of CLIP," NeurIPS, 2024 | Diverse templates and test-time adaptation evaluated on CIFAR-10-C | https://doi.org/10.52202/079017-1522 |
| LIT-12 | X. Zhu et al., "Enhancing Zero-Shot Vision Models by Label-Free Prompt Distribution Learning and Bias Correcting," NeurIPS, 2024 | Training-free prompt-distribution learning and confidence matching | https://doi.org/10.52202/079017-0064 |
| LIT-13 | W. Bao et al., "Mint: A Simple Test-Time Adaptation of Vision-Language Models against Common Corruptions," NeurIPS, 2025 | Direct evidence that CLIP remains vulnerable to common corruptions | https://proceedings.neurips.cc/paper_files/paper/2025/hash/54df19dc823e2cacff6d26640bac6c10-Abstract-Conference.html |
| LIT-14 | I. Arrieta-Ibarra et al., "Metrics of Calibration for Probabilistic Predictions," JMLR, 2022 | Binning and resolution limitations of scalar calibration metrics | https://www.jmlr.org/papers/v23/22-0658.html |
| LIT-15 | M. Friedman, "The Use of Ranks to Avoid the Assumption of Normality Implicit in the Analysis of Variance," JASA, 1937 | Matched-block rank test | https://doi.org/10.1080/01621459.1937.10503522 |
| LIT-16 | S. Holm, "A Simple Sequentially Rejective Multiple Test Procedure," Scandinavian Journal of Statistics, 1979 | Family-wise multiplicity control | https://www.jstor.org/stable/4615733 |

## Claim-to-Source Mapping

| Manuscript claim | Required source IDs | Boundary on wording |
| --- | --- | --- |
| CLIP performs zero-shot classification through prompt-conditioned text representations | LIT-01 | Describe the mechanism and original evaluation only |
| Prompt formulation can materially affect downstream performance | LIT-01, LIT-02, LIT-03 | Do not infer that all paraphrases are semantically equivalent |
| Recent prompt methods adapt to distribution shift | LIT-09 to LIT-13 | Distinguish learned/test-time methods from the frozen diagnostic used here |
| Common corruptions are a standardized robustness stress test | LIT-04 | Do not equate synthetic corruption with all real-world shift |
| Calibration matters under dataset shift | LIT-05, LIT-06 | ECE is a summary statistic, not complete calibration evidence |
| AURC measures selective risk across coverage | LIT-07, LIT-08 | Lower AURC is better; do not treat it as calibration |
| ECE has binning-related limitations | LIT-07, LIT-14 | State as a limitation, not a reason to discard ECE |
| Friedman and Holm procedures match the analysis plan | LIT-15, LIT-16 | The corruption type is the inferential block |

## Gap Statement Approved for Drafting

Prior work has developed increasingly sophisticated prompt-learning and
test-time adaptation methods for shifted data, including evaluations on
CIFAR-10-C. We found limited direct evidence of a fully crossed, frozen
paraphrase study that jointly measures accuracy, confidence calibration, and
selective risk across corruption type and severity. The manuscript therefore
claims a diagnostic and reporting contribution, not a new adaptation model and
not priority over every possible prompt-robustness study.

## Literature Limitations

- The search was focused and reproducibility-oriented rather than a systematic
  review.
- Several conference-proceedings records do not expose page ranges in their
  landing-page metadata; references retain verified venue, year, DOI, and URL
  without guessing missing pages.
- The rapidly changing 2024-2025 test-time adaptation literature is used to
  position the paper, not to establish exhaustive state-of-the-art coverage.
- No cited source was found to be retracted in the publisher records inspected;
  a formal Retraction Watch database audit remains advisable before submission.
