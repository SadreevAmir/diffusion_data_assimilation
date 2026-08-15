# Reproducibility handoff for the frozen mechanism result

This handoff covers the 40-case development-period mechanism analysis and the
completed fixed-contract joint calibration audit. Neither requires raw
ensembles in the publication worktree.

## Frozen inputs and procedure

- Analysis unit: date-level case.
- Ensemble size: 10.
- Cross-fitting: five deterministic date-stratified folds, with 32 selection
  cases and eight held-out cases per fold.
- Transform: `mean + scale * (member - mean)`, followed by clipping only for
  bounded scoring.
- Selection objective: case-mean fair CRPS.
- Scale grid: `1.0, 1.1, ..., 4.0`.
- Selected fold scales: `2.6, 2.7, 2.8, 2.8, 2.6`.
- Paired uncertainty: completed by trusted mode
  `validation_existing_calibration_paired_uncertainty` with source
  `joint_existing_ensemble_calibration_audit_valid`. It pairs all 40 dates and
  compares raw with all three fixed candidates without reading raw ensembles.
  The manuscript cites only fields present in its returned compact summary;
  aggregate means are not used to synthesize pairs.

## Joint-gate audit trail

The completed audit evaluates 40 cases with ten members and fixed folds,
candidate grid and tie randomization. The global-spread candidate passes the
aggregate proper-score thresholds but fails the predeclared no-compensation
gate. Reconciliation anchors are: established-ice Brier score `0.0569726083`
raw and `0.0581998581` corrected; exact-one member mass `0.0090418817` raw and
`0.1648320723` corrected; mean IIEE `0.0796004071` raw and `0.0839797890`
corrected; edge disagreement `0.0351291198` raw and `0.0366783519` corrected;
absolute ice-extent error `0.0461536485` raw and `0.0507920924` corrected.
These values support rejection, not reconstruction of paired case differences.

The subsequent fixed purged hurdle-isotonic/ECC-Q audit reuses the same raw
ensemble through a reviewed server-side compact contract. It uses five
contiguous eight-date holdouts with a three-case purge, a fixed-grid hurdle
isotonic bounded distribution fitted only outside each purged holdout, and
ten-member ECC-Q reconstruction from the raw rank template. There are no
runtime tuning parameters. All 40 cases are finite and the reported ECC
rank-order violation count is zero. Reconciliation anchors for the candidate
are fair CRPS `0.0855814526`, ordinary CRPS `0.0977137292`, ensemble-mean RMSE
`0.2684209181`, exact-zero mass absolute error `0.0134507165`, exact-one mass
absolute error `0`, randomized-rank discrepancy per point-case `0.0053082831`,
mean IIEE `0.1833850772`, edge disagreement `0.3058432880`, and energy-score RMS
`0.1979851836`. The authoritative compact summary reports paired date and
non-overlapping four-case-block intervals; the latter remains a post-hoc
temporal sensitivity. These anchors document a rejected method and must not be
used to reconstruct case-level pairs.

The mean-preserving projected-spread audit also reuses the same 40 server-side
ensembles and the already frozen cross-fitted scales. For each ten-member pixel
distribution it applies an exact capped-simplex projection with no runtime
tuning parameters. Reconciliation anchors are fair CRPS `0.0563000911`,
ordinary CRPS `0.0610710199`, ensemble-mean RMSE `0.1903366044`, mean IIEE
`0.0796004071`, randomized-rank discrepancy per point-case `0.0148130296`,
established-ice Brier score `0.0596498653`, avoidable exact-one mass `0`,
upper-cap mass `0.1674381130`, and maximum mean error `3.87e-16`. The trusted
summary reports fair-CRPS date CI `[-0.00297734, -0.00139088]` and non-overlap
four-case-block CI `[-0.00340517, -0.000862635]`. These anchors support a
rejected sufficiency hypothesis: proper scores and mean-field invariants pass,
but boundary, inner-order and member-spatial families fail. They must not be
used to reconstruct case-level pairs.

The final fixed mean-preserving open-logit audit inherits the projected
candidate's exact zero mask and the frozen transferred modulo-five scales. It
replaces hard capping by an open-logit member transform and solves a per-pixel
intercept to preserve the raw ensemble mean; the transferred scales are not
claimed as cross-fitted optima for this transform family. Reconciliation
anchors are fair CRPS `0.0557378026`, ordinary CRPS `0.0604237707`,
ensemble-mean RMSE `0.1903366044`, randomized-rank discrepancy per point-case
`0.0117638969`, inner-order attainable-coverage error `0.1483434796`,
established-ice Brier score `0.0591071355`, mass above `0.999`
`0.0719361803`, upper-logit-clamp mass `5.14e-7`, upper-cap mass `0`, and
maximum mean error `4.57e-16`. All 40 cases are finite with zero strict
rank-order violations. The authoritative gate rejects the candidate because
boundary and member-spatial families fail despite the other four families
passing. This is the terminal fixed diagnostic; epsilon, transform bounds,
zero mask and transferred scales must not be tuned after this result.

## Compact-artifact contract

Two distinct compact contracts must not be conflated. The trusted 40-row
spread-only corrected-case table has an audited contract containing
`analysis_fair_crps`, `analysis_crps`, `analysis_spread_skill_ratio`, and
`analysis_coverage_90`, but no corresponding `raw_*` fields. The aggregate
summary separately reports corrected and raw means. Those means are sufficient
for aggregate reconciliation but cannot identify paired case differences.

The completed joint-audit table instead has 160 long-form rows, with
`target_date`, `fold`, `method` and the full proper-score, rank, boundary and
spatial diagnostic family. It includes raw and corrected methods, so a trusted
server analysis can join them by `target_date` and compute paired date-level
uncertainty plus a circular contiguous four-date-block temporal sensitivity.
The reviewed CPU mode compares raw with all three fixed candidates and never
reads raw ensembles. Its completed compact summary is the authoritative source
for the paired intervals cited here and in the manuscript; the worktree
intentionally does not copy the per-case table or raw ensembles.

Accordingly, `paper/make_case_level_artifacts.py` is a guarded server-analysis
generator with two explicit input layouts. Its `--long-form` layout consumes
the completed joint-audit table directly, filters the exactly named raw and
global-spread methods, and pivots only on unique ISO `target_date`/`method`
pairs. It requires exactly 40 dates and both methods on every date, checks the
proper-score, deterministic, boundary and spatial case means against the
trusted aggregate anchors within `1e-10`, and only then writes paired date and
contiguous-block intervals. The legacy two-table layout remains available for
the earlier wide contract. Neither layout uses row order for pairing. Outputs
from this local generator are not cited: the manuscript intervals come from
the separately completed trusted compact uncertainty package, whose fixed
resampling contract is authoritative.

For the completed contract, the reviewed server invocation is:

```bash
python3 paper/make_case_level_artifacts.py per_case_metrics.csv \
  --long-form --date-column target_date --block-length 4 \
  --summary case_level_uncertainty.json \
  --figure case_level_fair_crps.svg
```

The default method labels match the audited contract; any override is recorded
in the invocation. The block length is fixed at four cases by the predeclared
gate. The generator sorts by parsed dates, rejects duplicates and malformed
dates, and reports paired case bootstrap and paired circular contiguous-block
bootstrap intervals. This prevents arbitrary CSV row order from being treated
as temporal adjacency. In the legacy layout, the corrected table is positional
and the raw table is supplied through `--raw-csv`.

## Required reconciliation checks

Before citing the generated outputs, verify that their case means reproduce the
trusted aggregate summary to ordinary floating-point tolerance:

| Metric | Raw | Corrected |
|---|---:|---:|
| Fair CRPS | 0.0584905850 | 0.0556896736 |
| Ordinary CRPS | 0.0621082810 | 0.0611013421 |
| Spread-skill ratio | 0.7240662110 | 1.0614835127 |
| 90% interval diagnostic | 0.5050835783 | 0.8787749039 |

Treat a mismatch larger than `1e-10` in any listed case mean as a provenance or
schema failure: do not publish the artifact and do not repair it by rounding or
manual editing. The guarded generator enforces this check before creating its
output directories or writing JSON/SVG artifacts. Paired intervals and
improved-case counts belong to the separately completed trusted compact
uncertainty package, not to this generator invocation or the aggregate anchors.

## Scope and audit trail

The aggregate Figure 1 is generated without raw ensembles:

```bash
python3 paper/make_calibration_summary_figure.py \
  aggregate_case_mean_metrics.json \
  paper/figures/calibration_summary.svg
```

The generator requires a one-row JSON list, `num_cases == 40`, and finite raw
and corrected values for both CRPS variants, spread-skill and all four interval
diagnostics. It fails closed on a missing key, changed case count or non-finite
value. Figure 1 is descriptive: it neither manufactures paired intervals nor
adds unavailable spatial/physical evidence.

The claim-led joint-gate Figure 2 is generated from the completed projected-
spread aggregate without raw ensembles:

```bash
python3 paper/make_joint_gate_figure.py \
  aggregate_case_mean_metrics.json paper/figures/joint_gate_summary.svg
```

The generator requires the exact reviewed candidate identifier, both full-
region method rows, 40 finite cases per method and `overall_eligible=false`.
It reports candidate/raw ratios and fixed per-metric limits; it does not form a
composite score or recompute the gate. The checked-in SVG is a compact rendering
of the trusted summary and can be regenerated only after these checks pass.

The compact CSV and aggregate JSON retain their trusted manifests outside the
paper narrative. Venue metadata, author statements and data-release decisions
remain external inputs.

The publication package itself has a local fail-closed integrity audit:

```bash
python3 paper/check_publication_artifacts.py
```

It requires the manuscript, claim ledger, research plan, reproducibility handoff
and readiness audit; resolves every manuscript figure within `paper/`; parses
linked SVG files as XML; checks contiguous numbered references; and rejects a
ready status with nonempty scientific blockers or a not-ready status without
explicit blockers. It also requires the final fixed diagnostic's fair-CRPS,
established-ice Brier and near-upper-bound anchors, together with the frozen
no-post-hoc-tuning decision, in the manuscript, claim ledger, reproducibility
handoff and readiness audit. This catches a partially updated evidence chain;
it validates package integrity, not scientific eligibility.
