# PromptFrag-C: Prompt-Induced Reliability Drift in Zero-Shot CLIP under Common Corruptions

**Authors:** [TBD]  
**Affiliations:** [TBD]  
**Corresponding author:** [TBD]

## Abstract

Zero-shot vision-language classifiers normally use a fixed textual template to
construct class representations, yet semantically intended paraphrases may
respond differently to visual distribution shift. We evaluated 12 frozen
class-prompt templates and a probability ensemble with OpenAI-pretrained RN50
and ViT-B/32 CLIP on CIFAR-10-C. The fully crossed design covered 15 corruption
types, five severities, 10,000 images per condition, and three primary
outcomes: accuracy, expected calibration error (ECE), and area under the
risk-coverage curve (AURC). Prompt effects were tested after averaging severity
within each corruption, and a leave-one-corruption-out (LOCO) rule selected
prompts using a frozen stability score. The mean within-condition accuracy
range was 11.66 percentage points for RN50 and 4.75 points for ViT-B/32. All
six model-by-metric prompt effects remained significant after global Holm
correction. LOCO improved accuracy by 1.03 and 1.28 points, respectively, but
worsened ECE by 1.93 and 1.00 points. It improved ViT-B/32 AURC, while the
RN50 AURC result was inconclusive. Prompt selection can therefore increase
recognition accuracy without uniformly improving reliability. Corruption
studies should disclose prompt sensitivity and report calibration or selective
risk alongside accuracy.

**Index Terms:** vision-language models, zero-shot classification, distribution
shift, confidence calibration, selective prediction, sensitivity analysis

## 1. Introduction

Contrastive Language-Image Pre-training (CLIP) connects image and text
representations so that natural-language descriptions can define a classifier
without task-specific parameter training [1]<!--ref:radford2021clip--><!--anchor:section:Abstract-->.
For a class label such as *cat*, the inference pipeline embeds a sentence such
as "a photo of a cat" and compares that text representation with an image
representation. The wording surrounding the class name is consequently part of
the classifier. The original CLIP evaluation used hand-designed templates and
template ensembles, while later prompt-learning work showed that context words
can be optimized and that modest wording changes can have substantial effects
on downstream performance [1], [2]<!--ref:radford2021clip--><!--anchor:section:Methods--><!--ref:zhou2022coop--><!--anchor:section:Abstract-->.

Prompt choice becomes a reliability concern when visual inputs also change. A
single template used on clean and corrupted images can make a robustness result
depend jointly on the visual encoder, corruption, and linguistic formulation.
Common-corruption benchmarks were introduced to measure performance under
standardized, non-adversarial image degradations [3]<!--ref:hendrycks2019corruptions--><!--anchor:section:Abstract-->.
Separate work on predictive uncertainty has shown that distribution shift can
degrade both accuracy and the usefulness of model confidence
[4]<!--ref:ovadia2019uncertainty--><!--anchor:section:Abstract-->. Evaluating
only accuracy from one prompt may therefore hide instability in correctness,
calibration, or the model's ability to identify predictions that should be
rejected.

Recent studies have developed learned or test-time prompts for distribution
shift, but we found limited direct evidence of a controlled study that freezes
ordinary paraphrastic templates and measures their joint effect on accuracy,
calibration, and selective risk over corruption type and severity. This paper
asks: **How strongly do paraphrastic prompts interact with corruption type and
severity to affect zero-shot CLIP accuracy, calibration, and selective risk,
and can LOCO stability selection reduce that drift?** We test whether prompt
identity has a matched effect on all three metrics (H1), whether
prompt-by-condition interaction contributes meaningfully to observed variation
(H2), and whether a frozen stability selector improves accuracy and AURC
without materially worsening ECE (H3).

The contributions are threefold. First, we provide a fully matched evaluation
of 12 frozen paraphrastic prompts across two CLIP backbones and 75 corrupted
conditions. Second, we use corruption type, rather than individual images or
severity cells, as the conservative inferential block and report effect sizes
with multiplicity control. Third, we evaluate a leakage-controlled LOCO
selection rule and show that its accuracy gains coincide with a calibration
cost. The study is a diagnostic and reporting contribution, not a new trained
adaptation model.

## 2. Related Work

### 2.1 Prompt Dependence in Vision-Language Models

CLIP uses a text encoder to synthesize class representations, making prompt
design a direct input to zero-shot recognition [1]<!--ref:radford2021clip--><!--anchor:section:Abstract-->.
Context Optimization (CoOp) replaced hand-written context tokens with learned
continuous vectors and reported strong few-shot gains across 11 datasets
[2]<!--ref:zhou2022coop--><!--anchor:section:Abstract-->. Conditional Context
Optimization (CoCoOp) subsequently showed that static learned contexts can
overfit base classes and used image-conditional tokens to improve
generalization [5]<!--ref:zhou2022cocoop--><!--anchor:section:Abstract-->. These
results establish that prompt context affects performance, but their main goal
is adaptation from labeled data rather than reliability diagnosis for frozen
manual paraphrases.

Distribution-aware prompt methods extend this line of work. PromptAlign adapts
multimodal prompts at test time by aligning test and source statistics
[6]<!--ref:samadh2023promptalign--><!--anchor:section:Abstract-->, while
Any-Shift Prompting models relationships between training and test
distributions [7]<!--ref:xiao2024anyshift--><!--anchor:section:Abstract-->.
WATT combines diverse templates, pseudo-label-based updates, and weight
averaging, with evaluations that include CIFAR-10-C
[8]<!--ref:osowiechi2024watt--><!--anchor:section:Abstract-->. Frolic learns a
label-free distribution over prompt prototypes and applies bias correction
[9]<!--ref:zhu2024frolic--><!--anchor:section:Abstract-->. These methods seek
better shifted-domain performance. PromptFrag-C instead holds all model
parameters fixed and asks how much a modest panel of predefined sentences
changes the reliability conclusions themselves.

### 2.2 Reliability under Corruption

CIFAR-10-C applies 15 common corruption types at five severity levels and
supports standardized comparison under synthetic covariate degradation
[3]<!--ref:hendrycks2019corruptions--><!--anchor:section:Benchmark-->. Recent
work specifically reports that CLIP remains vulnerable to common corruptions
and connects degradation to changes in its embedding geometry
[10]<!--ref:bao2025mint--><!--anchor:section:Abstract-->. Such benchmarks do
not represent every natural or semantic shift, but they provide repeatable,
graded stress conditions.

Reliability requires more than correctness. Calibration asks whether stated
confidence corresponds to empirical correctness, and ECE is a widely used
binned summary [11]<!--ref:guo2017calibration--><!--anchor:section:Abstract-->.
Selective prediction instead evaluates whether confidence can support
abstention. AURC aggregates the risk obtained as successively less confident
predictions are excluded, with lower values indicating better selective
behavior [12], [13]<!--ref:ding2020uncertainty--><!--anchor:section:Abstract--><!--ref:geifman2019selectivenet--><!--anchor:section:Abstract-->.
ECE depends on binning and compresses a calibration curve into one number, so
it should not be treated as a complete uncertainty assessment
[12], [14]<!--ref:ding2020uncertainty--><!--anchor:section:Metrics--><!--ref:arrieta2022calibration--><!--anchor:section:Abstract-->.
Our analysis therefore treats accuracy, ECE, and AURC as distinct primary
outcomes.

## 3. Method

### 3.1 Experimental Design and Metrics

We evaluated the OpenAI-pretrained CLIP RN50 and ViT-B/32 backbones on the
CIFAR-10 test set and CIFAR-10-C. Each corrupted condition contains the same
10,000 labeled images. The corruption grid comprised 15 types and five
severities, yielding 75 shifted conditions per model, plus one clean condition.
The tested corruptions included Gaussian, shot, and impulse noise, as well as
defocus, glass, motion, and zoom blur. We also included snow, frost, fog,
brightness, contrast, elastic transformation, pixelation, and JPEG
compression.

For each class, we constructed text using 12 fixed English templates:
`a photo of a {label}.` and 11 semantically intended paraphrases. The complete
inventory was frozen before the full experiment. It includes formulations such
as `a picture showing {label}.`, `this image shows {label}.`, and
`a natural image containing {label}.` The label string and class order were
identical across prompts. A thirteenth policy averaged the 12 prompt-level
probability vectors, but the primary prompt-effect tests use only the 12
individual templates. We did not fine-tune either encoder or estimate
parameters from the held-out test images.

For image \(x_i\), prompt \(p\), and class \(c\), CLIP produces a scaled cosine
similarity \(z_{ipc}\) between the normalized image and text embeddings.
Applying softmax over the ten class scores gives probability vector
\(\hat{\mathbf{p}}_{ip}\). Accuracy is the proportion for which the largest
probability corresponds to the label. ECE uses 15 equal-width confidence bins
and sums the bin-frequency-weighted absolute difference between mean
confidence and accuracy. AURC sorts predictions by maximum softmax confidence,
computes error risk over the retained coverage sequence, and integrates that
curve. Higher accuracy is better, whereas lower ECE and AURC are better. NLL,
Brier score, and risk at 80% coverage were logged as secondary diagnostics but
were not added to the confirmatory endpoint family.

All prompt comparisons use the same images, labels, preprocessing, and
pretrained weights. The seed was 20260723. Inference ran with PyTorch 2.9.1,
CUDA 12.8, and `open_clip_torch` 3.3.0 on an NVIDIA GeForce RTX 5060 Laptop
GPU with 8 GB of memory. The completed manifest contains a source hash and a
detailed environment record, and the analysis is generated deterministically
from 1,976 raw condition-policy rows.

### 3.2 Statistical Analysis and Stability Selection

Treating every severity cell as independent would overstate the effective
replication because severities share a corruption mechanism. For each model,
metric, and prompt, we first averaged the five severity values within each
corruption. The primary prompt-effect analysis then used the 15 corruption
types as matched blocks in a Friedman rank test
[15]<!--ref:friedman1937ranks--><!--anchor:section:Article-->. We report
Kendall's \(W\) and a 95% interval from 10,000 corruption-block bootstrap
samples. Holm's sequential procedure controls the family-wise error rate
globally across the six model-by-metric tests
[16]<!--ref:holm1979multiple--><!--anchor:section:Article-->. A 75-block
corruption-by-severity test is retained only as a sensitivity analysis.

To describe H2, we partitioned the balanced 12 by 75 metric matrix into prompt,
condition, and prompt-by-condition sums of squares. The resulting shares are
descriptive. They are not causal variance components and do not create
additional independent observations.

We also tested a frozen LOCO policy. For each held-out corruption \(j\), every
candidate prompt was scored on all severities from the other 14 corruptions:

\[
S_p=\overline{\mathrm{Acc}}_p-0.5\,\mathrm{SD}(\mathrm{Acc}_p)
-0.2\,\overline{\mathrm{ECE}}_p .
\]

The highest-scoring prompt was evaluated on all five severities of corruption
\(j\). This is a leakage-controlled comparison across corruption types, but it
uses labels from the other corruptions and is not unsupervised test-time
adaptation. We compared LOCO with the default prompt `a photo of a {label}.`
over 15 paired corruption means. For each model and metric, a two-sided
Wilcoxon signed-rank test assessed the paired change, a 10,000-sample paired
corruption bootstrap estimated the 95% interval for the mean difference, and
Holm correction was applied globally across the six comparisons. The
clean-selected, probability-ensemble, and oracle policies were retained as
contextual baselines rather than confirmatory comparisons.

## 4. Results

### 4.1 Prompt-Induced Reliability Drift

Prompt wording produced practically visible changes within the same corrupted
inputs. Across the 75 shifted conditions, the mean difference between the
best- and worst-accuracy prompts within a condition was 11.66 percentage
points for RN50 and 4.75 points for ViT-B/32. The largest RN50 range was
18.79 points under zoom blur at severity 4. The largest ViT-B/32 range was
10.66 points under glass blur at severity 4. ECE was similarly sensitive: its
mean within-condition prompt range was 18.55 points for RN50 and 9.11 points
for ViT-B/32. Thus, a single template could materially change both the apparent
recognition robustness and confidence calibration of a fixed model.

The conservative corruption-aggregated tests supported H1 for every outcome.
For RN50, Kendall's \(W\) was .904 for accuracy (95% CI [.854, .949], global
Holm \(p=1.11\times10^{-25}\)), .362 for ECE ([.261, .615],
\(p=1.01\times10^{-8}\)), and .819 for AURC ([.693, .917],
\(p=6.38\times10^{-23}\)). For ViT-B/32, \(W\) was .669 for accuracy
([.598, .788], \(p=4.46\times10^{-18}\)), .907 for ECE
([.863, .953], \(p=1.08\times10^{-25}\)), and .664 for AURC
([.600, .784], \(p=4.46\times10^{-18}\)). These values show that the
relative prominence of prompt effects depends on the model and metric. RN50
had its largest matched effect for accuracy, whereas ViT-B/32 had its largest
effect for calibration.

The descriptive decomposition refined H2. For RN50 accuracy, 94.65% of the
balanced sum of squares was associated with condition, 4.37% with prompt, and
0.98% with their interaction. RN50 AURC was also condition-dominated
(95.60%), with 3.15% attributed to prompt and 1.25% to interaction. Its ECE
pattern differed sharply: prompt accounted for 29.24%, condition for 14.07%,
and prompt-by-condition interaction for 56.69%. RN50 calibration is therefore
the clearest case in which the effect of wording changed across corruption and
severity.

ViT-B/32 accuracy and AURC were even more condition-dominated, at 98.90% and
98.34%, respectively. Their prompt and interaction shares were each below one
percent. ViT-B/32 ECE instead showed a 51.17% prompt share, a 36.29%
condition share, and a 12.53% interaction share. Prompt wording thus shifted
its calibration more consistently across conditions than it did for RN50.
These partitions support a metric-specific version of H2, not the stronger
claim that interaction is uniformly large.

Aggregate prompt rankings also ruled out a single reliability winner. On
RN50, the mean corrupted accuracy gap between the best and worst templates was
11.06 points, but the default prompt had the lowest mean ECE. On ViT-B/32, the
corresponding accuracy gap was 3.50 points, and the default again had the
lowest mean ECE. Prompts that increased correctness were therefore not
necessarily those that best aligned confidence with empirical accuracy.

### 4.2 Stability Selection and the Accuracy-Calibration Trade-off

The LOCO selector improved accuracy for both models. RN50 mean corrupted
accuracy increased from 47.29% with the default prompt to 48.32%, a difference
of 1.03 percentage points (95% CI [0.68, 1.39], global Holm
\(p=.00092\)). ViT-B/32 accuracy increased from 70.73% to 72.02%, a
1.28-point difference ([0.90, 1.66], \(p=.00073\)). These gains are small
relative to the full within-condition prompt ranges, but they are consistent
across the corruption blocks under the prespecified comparison.

Calibration moved in the unfavorable direction. RN50 ECE rose from 6.16% to
8.10%, an increase of 1.93 points ([0.91, 2.84], \(p=.0105\)). ViT-B/32 ECE
rose from 4.20% to 5.20%, an increase of 1.00 point ([0.44, 1.44],
\(p=.0205\)). Because lower ECE is better, both are statistically supported
calibration deteriorations. The selection score penalized mean ECE, but its
accuracy mean and variability terms still favored prompts with higher ECE.
H3 is therefore rejected as a joint claim that selection improves performance
without worsening calibration.

Selective risk was mixed. ViT-B/32 AURC decreased from .1157 to .1099, a
mean change of -.0057 (95% CI [-.0100, -.0020], global Holm \(p=.0128\)).
RN50 AURC decreased from .3205 to .3159, a mean change of -.0046 with a
bootstrap interval of [-.0083, -.0007], but its corrected Wilcoxon result was
not significant (\(p=.0730\)). The bootstrap interval concerns the mean
paired change, whereas the signed-rank test concerns the paired rank
distribution. We therefore classify RN50 AURC as inconclusive under the
prespecified decision rule rather than selecting the more favorable summary.

The chosen prompt identities further limit the adaptive interpretation. For
RN50, LOCO selected p10 for 13 held-out corruptions and p06 for two, so the
held-out corruption changed the selected prompt in a small subset. For
ViT-B/32, LOCO selected p03 for all 15 corruptions. That is also the prompt
selected using clean data. Its gain over the default is evidence for a better
fixed prompt in this panel, not evidence that corruption-specific adaptation
outperformed the clean-selected policy. Overall, H3 is supported for accuracy
and for ViT-B/32 selective risk, but not for joint reliability.

## 5. Discussion

The results identify prompt text as an experimental factor in zero-shot
corruption evaluation. The larger RN50 accuracy ranges and its strong
prompt-by-condition ECE share suggest that architecture and corruption can
shape how text-conditioned class similarities translate into predictions and
confidence. ViT-B/32 showed smaller accuracy ranges but a highly consistent
prompt ordering for ECE. These observations are compatible with prompt
context altering class-similarity geometry, but the study did not inspect
internal representations or manipulate linguistic features independently.
They should not be interpreted as a tested causal mechanism.

The policy experiment shows why reliability dimensions should remain
separate. An accuracy-heavy score produced roughly one point of accuracy gain
for both models and improved ViT-B/32 AURC, yet it worsened ECE for both.
The selector did not fail computationally. It optimized the preferences encoded
in its coefficients. Calling the selected prompt "more robust" without naming
the target metric would hide that value choice. A deployment that prioritizes
raw correctness may accept the trade-off, while a system that presents
probabilities to a user may not.

Three reporting practices follow. First, zero-shot robustness studies should
publish the exact prompt template and selection rule. Second, they should
evaluate a small prompt panel or report a sensitivity range rather than treating
one convenient template as neutral. Third, accuracy should be accompanied by
at least one calibration or selective-risk measure. Learned and test-time
prompt methods already demonstrate that adaptation can improve shifted-domain
accuracy [6]-[10]<!--ref:samadh2023promptalign--><!--anchor:section:Abstract--><!--ref:xiao2024anyshift--><!--anchor:section:Abstract--><!--ref:osowiechi2024watt--><!--anchor:section:Abstract--><!--ref:zhu2024frolic--><!--anchor:section:Abstract--><!--ref:bao2025mint--><!--anchor:section:Abstract-->.
The present results add that the metric bundle used to judge such improvement
can expose opposing conclusions.

Several limitations bound the findings. CIFAR-10-C contains synthetic common
corruptions and does not represent natural, semantic, label, or compound shifts.
Only two OpenAI-pretrained CLIP backbones and one ten-class dataset were
tested. The 12 prompts were designed as ordinary paraphrases, but semantic
equivalence and linguistic naturalness were not validated by human raters.
LOCO used labeled data from the other corruption types, and its fixed
coefficients encode an unvalidated utility function. ECE depends on binning,
and the 15 corruption types form a small inferential population. Finally, the
probability ensemble and oracle are contextual baselines, not independently
powered confirmatory claims. Replication should include larger VLMs, natural
shifts, preregistered prompt panels, and constrained or Pareto-based selection.

## 6. Conclusion

Across RN50 and ViT-B/32 CLIP, prompt identity was associated with
statistically significant and practically meaningful changes in zero-shot
accuracy, ECE, and AURC under common corruptions. RN50 showed larger accuracy
ranges and a strong prompt-by-condition calibration pattern. ViT-B/32 showed
smaller accuracy drift but a strong prompt main effect on ECE. These
architecture differences reinforce that results from one prompt and one metric
should not be generalized to overall reliability.

A frozen LOCO stability score improved accuracy by 1.03 percentage points for
RN50 and 1.28 points for ViT-B/32. It also improved ViT-B/32 AURC, but
worsened ECE for both models, and the RN50 AURC comparison remained
inconclusive. Moreover, ViT-B/32 LOCO reduced to the same fixed prompt selected
on clean data. The experiment therefore does not support a universal "robust
prompt." It supports a more practical conclusion: prompt sensitivity is part
of the uncertainty of a zero-shot robustness claim. Future evaluations should
disclose the prompt panel and selection objective, preserve the corruption as
the inferential block, and judge accuracy jointly with calibration and
selective risk.

## References

[1] A. Radford *et al.*, "Learning transferable visual models from natural
language supervision," in *Proc. 38th Int. Conf. Machine Learning*, vol. 139,
pp. 8748-8763, 2021. [Online]. Available:
https://proceedings.mlr.press/v139/radford21a

[2] K. Zhou, J. Yang, C. C. Loy, and Z. Liu, "Learning to prompt for
vision-language models," *Int. J. Comput. Vis.*, vol. 130, no. 9, pp.
2337-2348, 2022, https://doi.org/10.1007/s11263-022-01653-1

[3] D. Hendrycks and T. Dietterich, "Benchmarking neural network robustness to
common corruptions and perturbations," in *Proc. 7th Int. Conf. Learning
Representations*, 2019. [Online]. Available:
https://iclr.cc/virtual/2019/poster/731

[4] Y. Ovadia *et al.*, "Can you trust your model's uncertainty? Evaluating
predictive uncertainty under dataset shift," in *Advances in Neural Information
Processing Systems 32*, 2019. [Online]. Available:
https://proceedings.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html

[5] K. Zhou, J. Yang, C. C. Loy, and Z. Liu, "Conditional prompt learning for
vision-language models," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern
Recognition*, pp. 16816-16825, 2022,
https://doi.org/10.1109/CVPR52688.2022.01631

[6] J. A. Samadh *et al.*, "Align your prompts: Test-time prompting with
distribution alignment for zero-shot generalization," in *Advances in Neural
Information Processing Systems 36*, 2023,
https://doi.org/10.52202/075280-3525

[7] Z. Xiao, J. Shen, M. M. Derakhshani, S. Liao, and C. G. M. Snoek,
"Any-shift prompting for generalization over distributions," in *Proc.
IEEE/CVF Conf. Computer Vision and Pattern Recognition*, pp. 13849-13860,
2024, https://doi.org/10.1109/CVPR52733.2024.01314

[8] D. Osowiechi *et al.*, "WATT: Weight average test time adaptation of
CLIP," in *Advances in Neural Information Processing Systems 37*, 2024,
https://doi.org/10.52202/079017-1522

[9] X. Zhu, B. Zhu, Y. Tan, S. Wang, Y. Hao, and H. Zhang, "Enhancing
zero-shot vision models by label-free prompt distribution learning and bias
correcting," in *Advances in Neural Information Processing Systems 37*, 2024,
https://doi.org/10.52202/079017-0064

[10] W. Bao, R. Deng, and J. He, "Mint: A simple test-time adaptation of
vision-language models against common corruptions," in *Advances in Neural
Information Processing Systems 38*, 2025. [Online]. Available:
https://proceedings.neurips.cc/paper_files/paper/2025/hash/54df19dc823e2cacff6d26640bac6c10-Abstract-Conference.html

[11] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of
modern neural networks," in *Proc. 34th Int. Conf. Machine Learning*, vol. 70,
pp. 1321-1330, 2017. [Online]. Available:
https://proceedings.mlr.press/v70/guo17a

[12] Y. Ding, J. Liu, J. Xiong, and Y. Shi, "Revisiting the evaluation of
uncertainty estimation and its application to explore model
complexity-uncertainty trade-off," in *Proc. IEEE/CVF Conf. Computer Vision and
Pattern Recognition Workshops*, pp. 22-31, 2020,
https://doi.org/10.1109/CVPRW50498.2020.00010

[13] Y. Geifman and R. El-Yaniv, "SelectiveNet: A deep neural network with an
integrated reject option," in *Proc. 36th Int. Conf. Machine Learning*, vol.
97, pp. 2151-2159, 2019. [Online]. Available:
https://proceedings.mlr.press/v97/geifman19a

[14] I. Arrieta-Ibarra, P. Gujral, J. Tannen, M. Tygert, and C. Xu, "Metrics
of calibration for probabilistic predictions," *J. Mach. Learn. Res.*, vol.
23, no. 351, pp. 1-54, 2022. [Online]. Available:
https://www.jmlr.org/papers/v23/22-0658.html

[15] M. Friedman, "The use of ranks to avoid the assumption of normality
implicit in the analysis of variance," *J. Amer. Stat. Assoc.*, vol. 32, no.
200, pp. 675-701, 1937,
https://doi.org/10.1080/01621459.1937.10503522

[16] S. Holm, "A simple sequentially rejective multiple test procedure,"
*Scand. J. Stat.*, vol. 6, no. 2, pp. 65-70, 1979. [Online]. Available:
https://www.jstor.org/stable/4615733

## Declarations

### Data and Code Availability

CIFAR-10 and CIFAR-10-C are publicly available benchmark datasets. The
experiment configuration, analysis code, prompt inventory, raw metric table,
run manifest, and derived tables are available in the project workspace. A
public archival repository and permanent URL will be added before the
camera-ready submission.

### Ethics Statement

This study used public image-classification benchmarks and did not involve
human participants, animals, private records, or user interaction. Formal
research-ethics approval was therefore not required.

### Author Contributions

[TBD after the author list is finalized. Use the CRediT taxonomy.]

### Funding

[TBD. Authors must confirm all funding sources before submission.]

### Conflict of Interest

[TBD. Each author must confirm the final statement before submission.]

### Generative AI Disclosure

Generative AI tools assisted with code development, debugging, literature
discovery, and language drafting. The authors designed the study, verified the
sources and numerical results, reviewed all generated content, and remain
responsible for the manuscript. This statement must be reconciled with the
conference's current disclosure policy before submission.
