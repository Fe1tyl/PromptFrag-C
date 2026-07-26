# Literature Requirements for the 3,000-Word Paper

No external literature search was executed in the current `outline-only`
request. The outline therefore uses the verified local experiment as its
empirical evidence and reserves the following citation slots. Every slot must
be filled with a real, metadata-verified source before manuscript drafting.

## Minimum Source Set

| Slot | Required evidence | Preferred source type | Planned use |
| --- | --- | --- | --- |
| LIT-01 | Foundational definition and training objective of CLIP | Original peer-reviewed paper or official proceedings paper | Introduction and Method |
| LIT-02 | Evidence that zero-shot CLIP predictions depend on prompt templates or prompt ensembles | Primary empirical paper | Introduction and Related Work |
| LIT-03 | Learnable prompt or prompt-adaptation methods for vision-language models | Primary method paper | Related Work contrast |
| LIT-04 | Definition and construction of CIFAR-10-C/common-corruption benchmarks | Original benchmark paper | Method |
| LIT-05 | Robustness of image classifiers or vision-language models under common corruptions/distribution shift | Primary empirical paper | Motivation and Discussion |
| LIT-06 | Expected calibration error definition, use, and known limitations | Original or authoritative methodological paper | Method and Limitations |
| LIT-07 | Selective prediction and area under the risk-coverage curve | Original or authoritative methodological paper | Method |
| LIT-08 | Prompt robustness or prompt sensitivity under distribution shift | Closest recent primary empirical paper | Gap statement and Related Work |
| LIT-09 | Reliability or confidence calibration of vision-language models | Recent primary empirical paper | Related Work and Discussion |
| LIT-10 | Nonparametric matched-block testing and Kendall's W interpretation | Authoritative statistical source | Statistical Analysis |
| LIT-11 | Multiple-comparison control using Holm's procedure | Original or authoritative statistical source | Statistical Analysis |
| LIT-12 | Limitations of synthetic corruption benchmarks relative to natural shifts | Primary benchmark or survey evidence | Discussion |

## Search Plan for the Next Phase

Recommended databases are IEEE Xplore, ACM Digital Library, Scopus or Web of
Science, and arXiv only for relevant work that has no peer-reviewed version.
Google Scholar may be used for discovery and citation chaining, but metadata
should be verified against the publisher or proceedings record.

Suggested search strings:

1. `("CLIP" OR "vision-language model") AND ("prompt sensitivity" OR
   "prompt robustness" OR "prompt template") AND "zero-shot"`
2. `("CLIP" OR "vision-language model") AND ("common corruption" OR
   "distribution shift") AND (robustness OR reliability)`
3. `("vision-language model" OR CLIP) AND (calibration OR "expected
   calibration error" OR "selective prediction" OR AURC)`
4. `("prompt ensemble" OR "prompt selection") AND CLIP AND robustness`

## Scope and Recency

- Target 12–18 verified references for a 3,000-word conference paper.
- Include foundational sources regardless of year.
- Prefer recent primary work for the closest-gap comparison, while avoiding a
  literature list dominated by unreviewed preprints.
- At least one source must directly address prompt sensitivity under shift; if
  none exists, the manuscript must state that the gap is based on an explicit
  search outcome rather than an unsupported novelty claim.

## Citation Integrity Rule

The identifiers `LIT-01` through `LIT-12` are planning labels, not citations.
They must not appear in the submitted paper. No factual statement attributed to
prior work may be drafted until its source is retrieved and checked.
