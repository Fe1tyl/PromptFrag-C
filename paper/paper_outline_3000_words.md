# Detailed Paper Outline — 3,000 Words

## Working Title

**PromptFrag-C: Prompt-Induced Reliability Drift in Zero-Shot CLIP under Common
Corruptions**

Alternative, slightly more result-forward title:

**Paraphrastic Prompts Shift Zero-Shot CLIP Accuracy and Calibration under
Corruption**

Use the first title unless the literature search reveals an existing method or
benchmark named “PromptFrag-C.”

## Central Argument

Semantically similar class prompts are not a neutral implementation detail in
zero-shot CLIP evaluation. Across two pretrained CLIP architectures and 15
common corruptions, prompt identity is associated with substantial and
model-dependent changes in accuracy, calibration, and selective risk.
Stability-based prompt selection can recover roughly one percentage point of
accuracy, but it also worsens calibration, so prompt selection should be
reported as a multi-objective reliability trade-off rather than a universally
beneficial robustness intervention.

## Claim Status Before Drafting

- **H1 supported:** all six conservative model-by-metric Friedman tests remain
  significant after global Holm correction.
- **H2 supported descriptively, with strong metric dependence:** the
  prompt-by-condition interaction share is especially large for RN50 ECE
  (56.69%), but small for accuracy and AURC; ViT-B/32 ECE is more strongly
  prompt-main-effect dominated.
- **H3 only partially supported:** LOCO stability selection improves accuracy
  for both models and AURC for ViT-B/32, but significantly worsens ECE for both
  models; RN50 AURC is inconclusive under the paired Wilcoxon test.

## Word Budget

The target is **3,000 words including the abstract and excluding references,
tables, figure captions, and declarations**.

| Component | Words | Share |
| --- | ---: | ---: |
| Abstract | 170 | 5.7% |
| 1. Introduction | 430 | 14.3% |
| 2. Related Work | 390 | 13.0% |
| 3. Method | 590 | 19.7% |
| 4. Results | 820 | 27.3% |
| 5. Discussion | 400 | 13.3% |
| 6. Conclusion | 200 | 6.7% |
| **Total** | **3,000** | **100%** |

---

## Abstract — 170 words

### Purpose

Give a complete problem–method–result–implication account without literature
review or broad claims.

### Content Plan

1. **Context and gap (35 words):** zero-shot CLIP evaluations normally select a
   class-prompt template once, although semantically similar templates may
   behave differently under visual corruption.
2. **Method (55 words):** evaluate 12 frozen paraphrastic prompts and a
   probability ensemble on CIFAR-10-C using RN50 and ViT-B/32 CLIP; measure
   accuracy, ECE, and AURC; use corruption-aggregated Friedman tests and LOCO
   stability selection.
3. **Key results (60 words):** report the mean accuracy ranges (11.66 pp RN50;
   4.75 pp ViT-B/32), significant conservative prompt effects, and the paired
   LOCO trade-off: +1.03/+1.28 pp accuracy but +1.93/+1.00 pp ECE.
4. **Conclusion (20 words):** prompt selection can improve recognition while
   degrading calibration, motivating multi-metric reporting.

### Evidence Map

- Main Table II: conservative prompt effects.
- Main Table III: LOCO-versus-default results.

### Transition

End by framing prompt choice as a reliability factor that the Introduction
then motivates and formalizes.

---

## 1. Introduction — 430 words

### Purpose

Establish why prompt sensitivity under visual corruption is a reliability
problem, identify the exact gap, state the research question, and summarize
three bounded contributions.

### Content Plan

- **Problem and significance (about 170 words):** introduce CLIP-style
  zero-shot classification and the use of textual class prompts. Explain that
  a deployed or benchmarked system experiences both linguistic design choices
  and visual distribution shift. A robustness conclusion based on one prompt
  may therefore conflate model behavior with prompt choice. Support the CLIP
  description with `LIT-01`, prompt dependence with `LIT-02`, and corruption
  robustness motivation with `LIT-04`/`LIT-05`.
- **Gap (about 95 words):** distinguish ordinary clean-data prompt comparisons
  from a reliability analysis spanning corruption type, severity, accuracy,
  calibration, and selective risk. Use `LIT-08` to establish the closest known
  work. If the search does not find a directly comparable study, phrase the gap
  as “we found limited evidence” rather than “no prior work exists.”
- **Research question and hypotheses (about 85 words):** state the frozen
  research question in one sentence, followed by compact versions of H1–H3.
  Avoid predicting universal improvement.
- **Contributions (about 80 words):** enumerate: (1) a controlled two-model
  benchmark of 12 paraphrastic prompts over 75 shifted conditions; (2) a
  corruption-block statistical analysis of accuracy, ECE, and AURC; and (3) a
  leakage-controlled LOCO selection evaluation that exposes the
  accuracy–calibration trade-off.

### Evidence and Citation Map

- `LIT-01`, `LIT-02`, `LIT-04`, `LIT-05`, `LIT-08`.
- Table I may be cited in the final contribution paragraph to establish scope.
- Do not present result magnitudes here; reserve them for Section 4.

### Transition Logic

The final paragraph should move from the claimed gap to the bodies of
literature that define the study: prompt sensitivity and reliability under
distribution shift.

---

## 2. Related Work — 390 words

### 2.1 Prompt Dependence in Zero-Shot Vision-Language Models — 200 words

#### Purpose

Position paraphrastic template variation relative to prompt engineering,
prompt ensembling, and learned prompt adaptation.

#### Content Plan

- Summarize how CLIP maps images and prompt-conditioned class text into a shared
  representation (`LIT-01`).
- Review evidence that manual templates and prompt ensembles can change
  zero-shot performance (`LIT-02`).
- Contrast the present diagnostic study with learned prompt methods
  (`LIT-03`): PromptFrag-C does not train parameters and asks whether frozen,
  semantically similar prompts destabilize reliability under shift.
- End with the unresolved issue: a prompt that improves mean accuracy may not
  preserve calibration or selective-risk behavior.

#### Evidence Requirements

At least three verified primary sources. Do not let prompt-learning work crowd
out direct zero-shot template evidence.

#### Transition

Move from prompt-induced variation to the visual and probabilistic reliability
dimensions on which that variation is evaluated.

### 2.2 Reliability under Corruption: Accuracy, Calibration, and Selective Risk — 190 words

#### Purpose

Define the three reliability views and justify the chosen benchmark.

#### Content Plan

- Introduce common-corruption benchmarks and CIFAR-10-C (`LIT-04`) and place
  them within broader distribution-shift evaluation (`LIT-05`, `LIT-12`).
- Define ECE, while noting binning-related limitations that will be revisited
  in Discussion (`LIT-06`).
- Define selective prediction and AURC (`LIT-07`).
- Review the closest evidence on vision-language model calibration or
  robustness (`LIT-08`, `LIT-09`).
- State the synthesis: prior metrics are commonly reported separately, whereas
  the present study tests whether prompt choice changes their joint behavior.

#### Transition

Conclude with a one-sentence bridge: the Method operationalizes this joint
evaluation under a fully crossed, frozen design.

---

## 3. Method — 590 words

### 3.1 Experimental Design and Metrics — 300 words

#### Purpose

Make the study reproducible while keeping implementation details proportional
to a short paper.

#### Content Plan

- **Models and data:** RN50/OpenAI and ViT-B/32/OpenAI zero-shot CLIP on
  CIFAR-10 and CIFAR-10-C. State 15 corruption types, five severities, one clean
  condition, and 10,000 images per condition. Cite `LIT-01` and `LIT-04`.
- **Prompt intervention:** 12 frozen, semantically intended paraphrases of the
  class prompt plus a probability ensemble. Clarify that prompts were fixed
  before viewing full-run results, but linguistic equivalence was not
  human-validated.
- **Inference protocol:** no model fine-tuning; identical images, class labels,
  preprocessing, and model weights across prompt comparisons. State that the
  full grid produces 75 corruption-severity conditions per model.
- **Metrics:** define accuracy, ECE, and AURC as the primary outcomes. Mention
  NLL, Brier score, and risk at 80% coverage only as secondary logged metrics,
  not additional headline endpoints.
- **Reproducibility:** fixed seed 20260723, recorded package/GPU environment,
  deterministic analysis, and raw-output manifest.

#### Evidence and Display Map

- Main Table I: `paper/tables/table_1_experimental_design.csv`.
- Prompt inventory and detailed configuration can move to the supplement.
- Local provenance: `outputs/full/run_manifest.json` and
  `docs/experiment_plan.md`.

#### Transition

End by stating that the fully matched design permits within-corruption prompt
comparisons, leading to the statistical plan.

### 3.2 Statistical Analysis and Stability Selection — 290 words

#### Purpose

Specify the conservative inferential unit, effect sizes, multiplicity control,
and policy comparison without overstating independence.

#### Content Plan

- **Primary prompt-effect test:** average each prompt's five severity results
  within a corruption; use the 15 corruption types as matched blocks in a
  Friedman test, separately for each model and metric. Report Kendall's W and a
  10,000-sample block-bootstrap 95% CI. Apply Holm correction globally across
  six tests. Cite `LIT-10` and `LIT-11`.
- **Sensitivity/descriptive analysis:** retain the 75
  corruption-by-severity-block Friedman analysis as a supplement. Decompose the
  balanced matrix into prompt, condition, and prompt-by-condition
  sums-of-squares shares, labeling these as descriptive rather than causal.
- **LOCO selection:** for each held-out corruption, use the other 14
  corruptions to maximize the frozen score
  `mean accuracy − 0.5×SD accuracy − 0.2×mean ECE`; evaluate all severities of
  the held-out corruption. Compare with default, clean-selected, probability
  ensemble, and oracle policies.
- **Policy inference:** pair LOCO and default results over 15 corruptions, use a
  two-sided Wilcoxon signed-rank test, a paired corruption-block bootstrap CI
  for the mean difference, and global Holm correction across six
  model-by-metric comparisons.

#### Transition

Signal the Results order: first establish prompt-induced drift, then test
whether the frozen selection rule reduces it without creating another
reliability cost.

---

## 4. Results — 820 words

### 4.1 Prompt-Induced Reliability Drift — 430 words

#### Purpose

Answer H1 and H2 using conservative inference plus practically interpretable
ranges and model-specific decomposition.

#### Content Plan

- **Practical magnitude:** across corrupted conditions, the mean
  within-condition accuracy range across prompts is 11.66 percentage points
  for RN50 and 4.75 points for ViT-B/32. The maximum ranges are 18.79 points
  for RN50 at zoom blur severity 4 and 10.66 points for ViT-B/32 at glass blur
  severity 4. ECE ranges are also large: 18.55 and 9.11 points on average.
- **Conservative tests:** all model-by-metric prompt effects survive global
  Holm correction. Report, in compact form, RN50 W values of .904 accuracy,
  .362 ECE, and .819 AURC; ViT-B/32 values of .669, .907, and .664,
  respectively, with their Table II confidence intervals. Emphasize that
  effect ordering changes by model and metric.
- **Interaction pattern:** accuracy and AURC variation is overwhelmingly
  condition-dominated (94.65%/95.60% for RN50 and 98.90%/98.34% for
  ViT-B/32). Calibration differs: RN50 ECE has a 56.69% interaction share,
  whereas ViT-B/32 ECE has a 51.17% prompt main-effect share and 12.53%
  interaction share. State that H2 is strongest for RN50 calibration and much
  weaker for accuracy/AURC.
- **Prompt extrema:** optionally mention that the mean-accuracy gap between the
  best and worst prompts is 11.06 points on RN50 and 3.50 points on ViT-B/32.
  Avoid declaring any prompt universally best because p00 has the lowest ECE
  for both models while different prompts maximize accuracy.

#### Display and Evidence Map

- Main Table II:
  `paper/tables/table_2_conservative_prompt_effects.csv`.
- Main Figure 1: combine
  `outputs/full/analysis/figures/RN50_openai_prompt_accuracy_range.png` and
  `outputs/full/analysis/figures/ViT-B-32_openai_prompt_accuracy_range.png` as
  panels (a) and (b).
- Supplementary Table S1:
  `paper/tables/table_s1_variance_decomposition.csv`.
- Supplementary Table S4: `paper/tables/table_s4_prompt_extrema.csv`.

#### Transition

Conclude that prompt identity changes several reliability dimensions, but this
does not show that selecting a prompt by a frozen rule yields a joint
improvement. Section 4.2 tests that stronger proposition.

### 4.2 Stability Selection Produces an Accuracy–Calibration Trade-off — 390 words

#### Purpose

Answer H3 and separate statistically supported improvements from apparent but
inconclusive changes.

#### Content Plan

- **Accuracy:** LOCO selection improves RN50 accuracy from 47.29% to 48.32%
  (delta +1.03 pp, 95% CI +0.68 to +1.39, global Holm p=.00092) and
  ViT-B/32 from 70.73% to 72.02% (+1.28 pp, +0.90 to +1.66, p=.00073).
- **Calibration:** ECE worsens from 6.16% to 8.10% for RN50 (+1.93 pp,
  +0.91 to +2.84, p=.0105) and from 4.20% to 5.20% for ViT-B/32
  (+1.00 pp, +0.44 to +1.44, p=.0205). Explicitly state that lower ECE is
  better.
- **Selective risk:** ViT-B/32 AURC improves by −.0057
  (95% CI −.0100 to −.0020, p=.0128). RN50 shows a mean delta of −.0046,
  but its Wilcoxon result is not significant after correction (p=.0730), even
  though the bootstrap interval for the mean excludes zero. Explain that the
  two procedures target different summaries and classify the RN50 result as
  inconclusive.
- **Selection identity:** RN50 LOCO selects p10 for 13 corruptions and p06 for
  two, showing genuine held-out-dependent selection. ViT-B/32 always selects
  p03, identical to the clean-selected policy; therefore, its accuracy gain
  over the default prompt is not evidence that adaptive LOCO selection
  outperforms a fixed clean-selected prompt.
- **Hypothesis decision:** H3 is partially supported for accuracy and ViT
  selective risk but rejected as a joint “without worsening ECE” claim.

#### Display and Evidence Map

- Main Table III: `paper/tables/table_3_loco_vs_default.csv`.
- Main Figure 2: combine the two model-specific accuracy–ECE scatterplots from
  `outputs/full/analysis/figures/`.
- Supplementary Tables S2 and S3: policy means and selection frequencies.

#### Transition

Move from the numerical trade-off to its methodological meaning: prompt
selection optimizes a metric bundle whose weights encode a deployment
preference.

---

## 5. Discussion — 400 words

### 5.1 Interpretation and Implications — 230 words

#### Purpose

Explain why prompt effects differ by architecture and metric, and turn the
trade-off into a concrete reporting recommendation.

#### Content Plan

- Relate the model-specific W and decomposition patterns to the idea that text
  templates alter class similarities and confidence geometry, while avoiding a
  mechanistic causal claim not tested by the experiment.
- Explain that accuracy, calibration, and selective risk are not interchangeable:
  selecting for an accuracy-heavy stability score can increase correctness
  while making predicted confidence less calibrated.
- Recommend that zero-shot robustness studies report a prompt panel or
  sensitivity range, identify the prompt-selection rule, and present at least
  one calibration or selective-risk measure alongside accuracy.
- Clarify that the study offers an evaluation protocol and diagnostic result,
  not a new trained model.
- Compare the pattern with verified sources from `LIT-02`, `LIT-08`, and
  `LIT-09` after the literature search.

#### Transition

State that these implications are bounded by the benchmark, models, and prompt
construction used here.

### 5.2 Limitations and Threats to Validity — 170 words

#### Purpose

Bound generalization and disclose design constraints that matter for an IEEE
review.

#### Content Plan

- CIFAR-10-C represents synthetic common corruptions, not natural or semantic
  distribution shifts (`LIT-12`).
- The LOCO selector uses labeled examples from other corruption types and is
  not an unsupervised test-time adaptation method.
- Twelve prompts were intended as paraphrases but were not validated through a
  human semantic-equivalence study.
- Only two OpenAI-pretrained CLIP backbones were evaluated; findings may differ
  for larger architectures, other pretraining sources, or modern
  vision-language models.
- ECE depends on binning choices (`LIT-06`), and the selected stability-score
  coefficients encode a value judgment.
- The 15 corruption types, rather than individual severity cells, are the
  primary inferential blocks; broad population generalization remains limited.

#### Transition

The final sentence should motivate replication over additional datasets,
architectures, natural shifts, and multi-objective prompt-selection rules.

---

## 6. Conclusion — 200 words

### Purpose

Answer the research question directly, restate only supported contributions,
and end with a practical consequence.

### Content Plan

- Reiterate that prompt identity is associated with statistically significant
  and practically meaningful drift in zero-shot CLIP accuracy, calibration, and
  selective risk under common corruptions.
- Summarize the architecture contrast: RN50 shows larger accuracy ranges and
  stronger calibration interactions, whereas ViT-B/32 shows a strong prompt
  main effect on ECE.
- State the selection result precisely: the frozen LOCO score improves accuracy
  by about one percentage point for both models and improves ViT-B/32 AURC, but
  worsens ECE for both.
- Reject the simplistic conclusion that choosing a “robust prompt” uniformly
  improves reliability.
- Close with the actionable recommendation that zero-shot robustness claims
  should disclose prompt sensitivity and evaluate accuracy jointly with
  calibration and selective risk.
- Suggest one future direction: constrained or Pareto-based prompt selection
  evaluated across natural shifts and additional pretrained models.

### Evidence Map

Synthesize Main Tables II and III only; introduce no new numbers or citations.

---

## Planned Displays

| Item | Placement | Function | Source |
| --- | --- | --- | --- |
| Table I | Method 3.1 | Design and evaluation coverage | `paper/tables/table_1_experimental_design.csv` |
| Table II | Results 4.1 | Conservative prompt effects and practical ranges | `paper/tables/table_2_conservative_prompt_effects.csv` |
| Table III | Results 4.2 | LOCO-versus-default trade-offs | `paper/tables/table_3_loco_vs_default.csv` |
| Figure 1 | Results 4.1 | Accuracy-range profile across corruptions for both models | existing prompt-range PNGs |
| Figure 2 | Results 4.2 | Accuracy–ECE prompt trade-off for both models | existing scatterplot PNGs |
| Table S1 | Supplement | Descriptive prompt/condition/interaction shares | `paper/tables/table_s1_variance_decomposition.csv` |
| Table S2 | Supplement | All policy means | `paper/tables/table_s2_policy_summary.csv` |
| Table S3 | Supplement | Selected-prompt frequencies | `paper/tables/table_s3_selection_frequency.csv` |
| Table S4 | Supplement | Prompt extrema | `paper/tables/table_s4_prompt_extrema.csv` |

## Mandatory Back Matter — Excluded from Word Count

Draft these only after author confirmation:

- **Data Availability:** name CIFAR-10/CIFAR-10-C sources and provide the
  eventual code/result repository link.
- **Ethics Statement:** public benchmark images, no human participants, no
  newly collected personal data.
- **Author Contributions (CRediT):** roles TBD.
- **Conflict of Interest:** declaration TBD.
- **Funding:** source or “no external funding” only after confirmation.
- **AI-Use Disclosure:** describe permitted assistance accurately and follow
  the final IEEE/venue policy.

## Phase Gate

This outline completes the `outline-only` workflow stage. Full manuscript
drafting should begin only after:

1. the user approves or edits the outline and 3,000-word allocation;
2. the literature slots are filled with verified sources; and
3. author, funding, repository, and disclosure metadata are confirmed.
