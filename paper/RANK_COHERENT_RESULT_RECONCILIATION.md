# Rank-coherent result reconciliation

Status: PRE_RESULT_NO_TRUSTED_MODE

This is a fail-closed downstream publication contract, not scientific evidence
and not authorization to invent a mode. Reconciliation may begin only from the
same `load_and_validate_combined` operation over the exact controller-visible
admission JSON and decision-bearing compact directory.

## Mutually exclusive scientific branches

`RECONCILED_POSITIVE` requires literal `true` for proper score, finite-ensemble
reliability, boundary, spatial/physical, operational and `overall_eligible`.
Any false mandatory family requires `RECONCILED_NEGATIVE`; improvements in
another family cannot compensate and no post-result tuning is permitted.

## Atomic publication update map

Prepare complete replacements for `PAPER_DRAFT.md`, `CLAIM_LEDGER.md`,
`REPRODUCIBILITY.md`, `PUBLICATION_READINESS.md` and this file. Each replacement
must contain exactly one identical canonical `RANK_COHERENT_RESULT` line derived
by `canonical_marker`; `publish_reconciliation` reruns that same combined
admission and derivation itself, rejects an absent, duplicated or different line
before writing, and uses prepare-first replacement with rollback across all five
surfaces.

`rank_coherent_publication_renderer.py` removes the remaining manual content
transfer. It reruns combined admission, reads experiment/candidate identity from
the admitted metadata, renders every aggregate effect size with both compact
uncertainty intervals, and emits the five literal family decisions. A negative
branch names every failed family and states that cross-family compensation and
blocker closure are forbidden. The resulting complete replacements remain
subject to `publish_reconciliation`; renderer output alone does not write files.

## Fail-closed consistency rules

- `canonical_marker` invokes combined admission itself and constructs a marker
  only from the identities it returns after re-reading both exact inputs.
- The marker binds `reviewed_mode`, experiment and candidate identifiers, both
  admitted SHA-256 identities, the exact 40-case/10-member envelope, all five
  family Booleans and their literal conjunction as `overall_eligible`.
- A pre-result state contains no `RANK_COHERENT_RESULT` line. A reconciled state
  contains the identical line on all five surfaces.
- A negative result does not close the eligible-calibration blocker. A positive
  result closes it only when every mandatory family is true; neither branch
  changes unrelated minimum-tier evidence.
- Rollback covers caught filesystem errors during replacement. As in
  `atomic_publish.py`, this is not a journaled guarantee after abrupt process or
  operating-system termination.
