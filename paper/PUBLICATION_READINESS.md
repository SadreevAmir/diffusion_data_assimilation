# Publication readiness audit

Audit date: 2026-08-15

Publication status: NOT_READY_FOR_HUMAN_REVIEW

## Required scientific blockers

1. **Case-level uncertainty is absent from the paper package.** The aggregate
   mechanism result is internally consistent, but the draft has no date-level
   paired uncertainty interval or sensitivity summary. This can be computed
   from the existing compact per-case table with
   `paper/make_case_level_artifacts.py`; no new sampling is justified.
2. **Publication displays are incomplete.** The manuscript now contains an
   aggregate evidence table, but no provenance-backed case-level figure or
   frozen figure/table artifact exists in the worktree.
3. **Scholarly context is incomplete.** The draft has no bibliography or inline
   citations for finite-ensemble CRPS, ensemble calibration, generative data
   assimilation, or the application protocol.
4. **The manuscript is not yet a submission-form document.** It lacks author,
   venue/template, data/code availability, ethics/conflict statements, and a
   reproducibility checklist tailored to the intended venue.

## Audit findings that are not blockers to the narrow claim

- The abstract, results, claim ledger, research plan and handoff agree on the
  selected fold scales and all aggregate calibration metrics.
- `paper/REPRODUCIBILITY.md` freezes the compact input contract, deterministic
  generation command, expected aggregate reconciliation values and a numeric
  provenance-failure threshold.
- The claim is correctly limited to a validation-set mechanism result from one
  checkpoint, one seed, 40 dates and ten members.
- The clipping caveat is explicit: the bounded RMSE change is not presented as
  intrinsic improvement of the pre-clipping ensemble center.
- The manuscript does not claim deterministic-method superiority, complete
  tail calibration, conditional calibration or fieldwise coverage.
- No additional server experiment is needed to address the current blockers.

## Fastest defensible closure path

1. Retrieve the existing compact per-case analysis table and run
   `paper/make_case_level_artifacts.py` to produce a paired case-level
   uncertainty summary and one compact distribution-of-deltas figure. The
   generator enforces the frozen 40-case protocol, unique case identifiers when
   present, finite metric values, deterministic resampling, and records the
   input SHA-256 digest in the summary.
2. Freeze those outputs in the repository with a small generation script and
   provenance note, then cite them from the Results section.
3. Add and verify the bibliography and venue-required submission metadata.
4. Re-audit every quantitative manuscript sentence against the claim ledger
   and compact artifacts. Only then replace the status above with the
   controller-required ready status and record remaining limitations.

## External-only follow-ups

- Authors must select the target venue and supply author/affiliation, funding,
  conflict-of-interest and data-release details.
- Any claim beyond the present validation mechanism result requires genuinely
  independent evidence and is outside this readiness audit.
