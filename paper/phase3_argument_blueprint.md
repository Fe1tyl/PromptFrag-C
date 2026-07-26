# Phase 3 Argument Blueprint

## Central Thesis

In zero-shot CLIP evaluation under common corruptions, a frozen class-prompt
template is a consequential reliability factor. Prompt identity changes
accuracy, calibration, and selective risk in architecture- and metric-specific
ways. A stability-weighted selection rule can recover about one percentage
point of accuracy, but its calibration cost prevents it from being described as
a universal robustness improvement.

## Argument Chain

### Argument 1: Prompt identity produces reproducible reliability drift

- **Claim:** The observed differences among the 12 prompts are not adequately
  explained as cell-level noise in this experiment.
- **Evidence:** All six corruption-aggregated model-by-metric Friedman tests
  survive global Holm correction. Kendall's W ranges from 0.362 to 0.907.
- **Practical anchor:** Mean within-condition accuracy ranges are 11.66
  percentage points for RN50 and 4.75 points for ViT-B/32.
- **Warrant:** The matched design changes prompt text while holding image,
  model, preprocessing, and label space constant.
- **Boundary:** The result establishes association within CIFAR-10-C and the
  evaluated prompt panel, not universal semantic instability.

### Argument 2: Prompt sensitivity is architecture- and metric-dependent

- **Claim:** There is no single scalar description of prompt robustness.
- **Evidence:** RN50 has a larger accuracy range and a 56.69% descriptive
  prompt-by-condition share for ECE. ViT-B/32 has a 51.17% prompt main-effect
  share for ECE, while accuracy and AURC are mostly condition-dominated for both
  models.
- **Warrant:** Accuracy, ECE, and AURC summarize distinct aspects of prediction
  behavior.
- **Boundary:** Balanced sums-of-squares shares are descriptive partitions, not
  causal variance components.

### Argument 3: Stability selection creates a reliability trade-off

- **Claim:** The frozen LOCO score improves accuracy but does not improve all
  reliability dimensions.
- **Evidence:** Accuracy increases by 1.03 points for RN50 and 1.28 points for
  ViT-B/32. ECE worsens by 1.93 and 1.00 points. ViT-B/32 AURC improves, while
  RN50 AURC is inconclusive after the prespecified Wilcoxon-Holm analysis.
- **Warrant:** A selector optimized with an accuracy-heavy weighted score
  encodes a deployment preference rather than discovering a universally best
  prompt.
- **Boundary:** ViT-B/32 LOCO always selects p03, the same prompt selected on
  clean data, so it does not demonstrate an adaptive advantage over that fixed
  prompt.

## Counterargument and Response

**Counterargument:** Prompt variation may be an artifact of a small synthetic
benchmark or of deliberately heterogeneous templates, so the result may not
matter for practical deployments.

**Response:** The study does not claim population-wide prompt fragility.
Instead, it demonstrates that even a small, frozen panel of ordinary
class-template paraphrases can materially alter several reliability summaries
when all other inference components are matched. That finding is sufficient to
support a reporting recommendation: robustness studies should disclose the
prompt rule and quantify sensitivity. The external-validity concern is retained
as a limitation and motivates replication on natural shifts, additional
datasets, and newer backbones.

## Hypothesis Decisions

| Hypothesis | Decision | Reason |
| --- | --- | --- |
| H1: prompt identity has a matched effect on accuracy, ECE, and AURC | Supported within the evaluated design | Six of six global-Holm adjusted tests are significant |
| H2: prompt-by-condition interaction is meaningful | Partially supported and metric-dependent | Strong for RN50 ECE, modest for ViT-B/32 ECE, small for accuracy and AURC |
| H3: LOCO improves accuracy and AURC without worsening ECE | Partially supported, joint claim rejected | Accuracy improves, ViT AURC improves, but ECE worsens for both models |

## Writing Constraints

- Use "associated with," "changes," or "drift"; do not claim a causal
  linguistic mechanism.
- Pair every inferential result with an effect size or raw difference.
- Use "statistically significant" only for the prespecified global-Holm result.
- Call RN50 AURC "inconclusive," because the Wilcoxon and bootstrap summaries
  target different functionals.
- State that the LOCO policy uses labeled data from the other corruption types
  and is not an unsupervised test-time adaptation method.
- End with a diagnostic and reporting recommendation, not a claim of a new
  state-of-the-art model.
