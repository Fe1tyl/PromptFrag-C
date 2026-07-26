# Paper Configuration Record

## Workflow State

- Workflow: `ars/academic-paper/WORKFLOW.md`
- Operational mode: `full`
- Current phase: Phase 2 outline approved by the user; verified literature,
  argument architecture, and Phase 4 manuscript drafting are authorized.
- Evidence policy: local experimental artifacts may be cited as study results;
  external literature remains represented by source slots until a verified
  literature search is completed.

## Paper Identity

- Working title: **PromptFrag-C: Prompt-Induced Reliability Drift in Zero-Shot
  CLIP under Common Corruptions**
- Paper type: short empirical conference paper
- Target: CCSB 2026, IEEE proceedings
- Discipline: computer science / machine learning / reliable vision-language
  models
- Language: English
- Citation style: IEEE numeric
- Target length: approximately 3,000 words including abstract, excluding
  references, tables, figure captions, and mandatory declarations
- Planned structure: compact IMRaD with at most two subsections per major
  section and no level-three headings

## Frozen Research Question

How strongly do paraphrastic prompts interact with corruption type and severity
to affect zero-shot CLIP accuracy, calibration, and selective risk, and can
leave-one-corruption-out stability selection reduce that drift?

## Frozen Hypotheses and Decision Rules

- H1: Prompt identity has a non-zero matched effect on accuracy, ECE, and AURC.
- H2: Prompt-by-condition interaction accounts for a meaningful share of
  observed reliability variation.
- H3: Leave-one-corruption-out stability selection improves accuracy and AURC
  without materially worsening ECE.
- Primary prompt-effect inference: Friedman test after averaging the five
  severity levels within each corruption, using 15 corruption types as matched
  blocks.
- Effect size: Kendall's W with a 10,000-sample corruption-block bootstrap 95%
  confidence interval.
- Multiplicity: global Holm adjustment across six model-by-metric prompt-effect
  tests and, separately, six LOCO-versus-default policy comparisons.
- Practical reporting: always pair p-values with raw or percentage-point
  differences and confidence intervals.

## Inputs Available

- Frozen experimental plan: `docs/experiment_plan.md`
- Raw full-run metrics: `outputs/full/raw_metrics.csv`
- Run manifest: `outputs/full/run_manifest.json`
- Deterministic analysis artifacts: `outputs/full/analysis/`
- Paper-ready tables: `paper/tables/`
- Existing 300-DPI figures: `outputs/full/analysis/figures/`

## Provisional Metadata

- Authors and affiliations: TBD
- Corresponding author: TBD
- Funding: TBD; do not write “none” until confirmed
- Conflicts of interest: TBD
- CRediT roles: TBD
- Data/code repository URL: TBD
- AI-use disclosure: required before submission; exact venue wording TBD
- Venue page/word limits: not independently verified in this phase
- Domain evidence profile: computer-science/ML profile suggested but not yet
  confirmed by a completed literature search

## Drafting Guardrails

- Do not claim that LOCO selection improves reliability in every dimension.
- Do not treat corruption-severity cells as independent inferential units in
  the primary prompt-effect test.
- Do not describe the balanced sums-of-squares shares as causal variance
  components.
- Do not describe LOCO as an unsupervised deployment method because it uses
  labeled data from other corruption types.
- Do not claim an adaptive-selection advantage for ViT-B/32: its LOCO and
  clean-selected policies choose prompt p03 for all 15 held-out corruptions.
- Do not insert unverified or fabricated references.
