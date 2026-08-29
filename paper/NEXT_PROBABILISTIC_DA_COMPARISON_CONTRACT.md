# Frozen probabilistic DA comparison contract

Status: `FROZEN_NOT_EXECUTABLE`

Local runner-parity guard: `paper/probabilistic_da_contract_oracle.py` validates
the exact compact manifest, observation-output identity, localization grid,
assimilation procedure, purged fold selection and decision thresholds. Its
negative fixtures are in `paper/test_probabilistic_da_contract_oracle.py`.
Passing this oracle does not make the contract executable and does not admit a
trusted mode; it is a fail-closed prerequisite for subsequent runner review.

## Publication role

This contract freezes the missing probabilistic DA comparator without claiming
that a trusted runner or result exists. The comparator is a conventional LETKF,
not a calibration of generated members and not a candidate for the learned-joint
no-compensation gate. Its role is to test whether the learned-joint ensemble is
competitive with an assimilation method that updates an explicit finite
background ensemble under the same sparse observations. The LETKF method
identity follows Hunt, Kostelich and Szunyogh (2007),
doi:10.1016/j.physd.2006.11.008; that source does not establish the validity,
skill or fairness of this project's common-information comparison.

## Common-information and case contract

- Score exactly the same forty development cases used by
  `joint_full_condition_validation_2022`, in the same chronological order, with
  ten members and the already sealed truth, grid, masks and area weights.
- For every case, the LETKF and learned-joint methods must receive the identical
  observation values, observation locations, observation-error covariance,
  validity mask and assimilation time. The observation-operator output must be
  hash-identical before either update is run.
- The LETKF background ensemble must be produced by the project's conventional
  forecast/background pipeline without truth, generated members or
  learned-joint outputs. Its checkpoint/configuration digest and all ten member
  hashes must be recorded before scoring.
- No truth from a scored block may enter parameter selection, normalization,
  missing-value handling, localization, inflation or quality control.

Failure of any identity or leakage check is `COMPARATOR_INVALID`, not a negative
scientific result.

## Frozen algorithm and leakage-safe selection

Use a deterministic square-root local ensemble transform Kalman filter with all
state updates performed in the physical SIC variable and projected to `[0,1]`
only after the complete analysis update. The observation-error covariance is
the sealed covariance used by the common observation contract; it is never
estimated from verifying truth.

Use five contiguous eight-date scoring blocks and remove the three nearest cases
on each temporal side from each training set; the purge is non-circular. In each
fold, choose one localization-radius/inflation pair from the fixed Cartesian set

`radius_km in {50,100,200,400}` and `inflation in {1.00,1.05,1.10,1.20}`.

Minimize training-date mean fair CRPS. Resolve exact ties by smaller inflation,
then smaller radius. A pair is infeasible if any training analysis is non-finite,
if its mean leaves `[0,1]` before the final projection, or if the analysis-update
solver reports a numerical failure. Apply the selected pair unchanged to the
held-out block. Do not add adaptive inflation, parameter interpolation, regime
subdivision, observation thinning, covariance taper families, stochastic
perturbations or a second selection objective after reading results.

## Metrics and falsifiable decision

Compare held-out LETKF with both the raw learned-joint ensemble and the single
3D-Var analysis using the same date aggregation. Report fair and ordinary CRPS,
randomized-rank histogram and absolute uniformity decision, attainable `M=10`
central coverage and width, ensemble-mean RMSE, IIEE, ice-area and extent errors,
edge error, and at least one patchwise variogram discrepancy. Uncertainty uses
paired dates; a non-overlapping four-date-block interval is sensitivity only.
Pixels are never treated as independent replicates.

The pre-result prediction is that LETKF will be a useful probabilistic DA
comparator if all of the following hold jointly:

1. `COMPARATOR_VALID` passes all common-information, finiteness and fold-leakage
   checks for all forty cases and ten members;
2. LETKF fair CRPS is no more than `1.10` times raw learned-joint fair CRPS;
3. LETKF passes the frozen absolute randomized-rank uniformity criterion; and
4. LETKF ensemble-mean RMSE is no more than `1.02` times 3D-Var RMSE.

If all four conditions pass, label the result `PROBABILISTIC_DA_USEFUL`. Otherwise
label it `PROBABILISTIC_DA_NEGATIVE`, naming every failed condition. Either valid
outcome closes the minimum-tier evidence row; invalid execution leaves it
`MISSING`. Success does not establish an eligible learned-joint calibration, and
failure rejects only this exact small-ensemble LETKF contract rather than the
probabilistic DA family.

## Publication-readiness closure binding

Readiness closure requires a valid trusted execution and exactly one decision-bearing outcome: `PROBABILISTIC_DA_USEFUL` or `PROBABILISTIC_DA_NEGATIVE`.
It also requires the same exact compact-record SHA-256 identity in the
manuscript, claim ledger, readiness audit and reproducibility handoff; an outcome
label without that four-surface identity closes nothing.
Contract availability or `COMPARATOR_INVALID` leaves the normative comparison row `MISSING`.

## Execution and compact evidence boundary

All computation and analysis must run on the server. Retrieval is
`summary_only`: one JSON decision summary, one 120-row case-metric CSV (forty
cases for each of the three exact methods), and one compact JSON manifest
containing SHA-256 hashes of the exact summary and CSV bytes, fold selections
and invariant checks. `probabilistic_da_adapter_parity.py` loads exactly these
three files, invokes `validate_manifest`, `validate_case_rows` and
`evaluate_decision`, verifies both payload hashes, and rejects a claimed outcome
that differs from the recomputed frozen label. No fourth admission artifact is
permitted.
The controller-facing `validate_probabilistic_da_admission.py` CLI hashes the
exact filenames and bytes of all three artifacts with length-delimited framing,
runs the complete semantic adapter, and repeats the directory digest afterward.
Any payload or directory-membership substitution during admission fails closed;
the emitted JSON binds the recomputed outcome to the admitted directory SHA-256.
The controller must retain that identity and use the downstream consumer:

```bash
python3 paper/validate_probabilistic_da_admission.py COMPACT_DIRECTORY > ADMISSION.json
python3 paper/reconcile_probabilistic_da_admission.py \
  ADMISSION.json EXPECTED_COMPACT_DIRECTORY_SHA256
```

The second command rejects an outcome without the exact expected
`compact_directory_sha256`; an outcome string alone is never reconciled.
The probabilistic-DA evidence row can close only after combined admission and
downstream reconciliation reproduce the controller-retained directory digest
from the separately reviewed trusted runner.
No raw backgrounds, analyses, observations or truths are retrieved. No currently
admitted trusted mode implements this contract, so it must not be proposed under
an invented identifier. Admission requires a separately reviewed runner that
implements this literal contract and fails closed on every identity check.
