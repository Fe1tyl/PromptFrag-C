# Citation Audit Report

## Summary

| Metric | Result |
| --- | ---: |
| Citation source-marker occurrences | 26 |
| Unique cited sources | 16 |
| Reference-list entries | 16 |
| Orphan in-text citations | 0 |
| Orphan references | 0 |
| Reference sequence | [1] through [16], in order of first appearance |
| Format errors remaining | 0 detected |
| Missing verified DOIs | 0 |
| Sources from 2021-2026 | 9/16 (56.25%) |
| Sources older than ten years | 2/16 (12.5%) |
| Self-citation ratio | Pending author list |

## Compliance Findings

- The manuscript uses IEEE numeric in-text citations and a numbered reference
  list ordered by first appearance.
- Every source in the reference list is cited in the body, and every cited
  source has a corresponding reference entry.
- The manuscript contains hidden source and section anchors immediately after
  visible citations. These comments support provenance checks and do not appear
  in rendered Markdown.
- All DOIs exposed by the verified publisher or conference records are included
  as `https://doi.org/` URLs. Sources without a publisher-assigned DOI retain a
  direct proceedings or journal URL.
- The two sources older than ten years are the foundational Friedman matched
  rank test and Holm multiplicity procedure. Their use is methodological rather
  than a substitute for current domain literature.
- Background and related-work claims carry citations. Uncited paragraphs in
  Method, Results, Discussion, and Conclusion describe this study's design,
  results, interpretation, or limitations.

## DOI and Source Verification

| Ref. | DOI status | Verification status |
| ---: | --- | --- |
| [1] | No publisher DOI identified | PMLR record verified |
| [2] | 10.1007/s11263-022-01653-1 | Verified against journal metadata |
| [3] | No publisher DOI identified | ICLR proceedings record verified |
| [4] | No publisher DOI identified | NeurIPS proceedings record verified |
| [5] | 10.1109/CVPR52688.2022.01631 | CVF record and DOI metadata cross-checked |
| [6] | 10.52202/075280-3525 | NeurIPS proceedings record verified |
| [7] | 10.1109/CVPR52733.2024.01314 | CVF and institutional records cross-checked |
| [8] | 10.52202/079017-1522 | NeurIPS proceedings record verified |
| [9] | 10.52202/079017-0064 | NeurIPS proceedings record verified |
| [10] | No DOI displayed by proceedings | NeurIPS proceedings record verified |
| [11] | No publisher DOI identified | PMLR record verified |
| [12] | 10.1109/CVPRW50498.2020.00010 | CVF and DOI metadata cross-checked |
| [13] | No publisher DOI identified | PMLR record verified |
| [14] | No publisher DOI identified | JMLR record verified |
| [15] | 10.1080/01621459.1937.10503522 | Journal DOI metadata verified |
| [16] | No DOI identified | Journal metadata and JSTOR record verified |

## Corrections Made

| Location | Issue | Resolution |
| --- | --- | --- |
| Reference [5] | CVPR DOI was uncertain during planning | Verified and set to `10.1109/CVPR52688.2022.01631` |
| Reference [7] | CVPR DOI absent from the CVF landing-page BibTeX | Cross-checked against the University of Amsterdam repository and added |
| Reference [12] | Workshop page displayed inconsistent page metadata in its short listing | Used the verified IEEE page range 22-31 and DOI |
| Entire reference list | Mixed availability of DOI and proceedings URLs | Applied a uniform IEEE rule: DOI when available, otherwise direct primary URL |

## Items Requiring Author Review

1. The self-citation ratio cannot be assessed until the author list is final.
2. A formal Retraction Watch database screen was not available in this
   workflow. No retraction or expression-of-concern notice appeared on the
   publisher records inspected.
3. The final IEEE template may impose a different URL presentation or author
   truncation rule. Re-run format validation after the manuscript is placed in
   the official conference template.

## Gate Decision

**PASS for drafting.** The current manuscript has zero citation orphans, a
complete verified DOI set, and consistent IEEE numbering. The three
author-dependent checks above remain for the submission-stage audit.
