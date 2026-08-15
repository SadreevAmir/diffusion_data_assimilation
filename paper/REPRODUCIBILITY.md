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
for the paired intervals cited here; the worktree intentionally does not copy
the per-case table or raw ensembles.

Accordingly, `paper/make_case_level_artifacts.py` is a guarded server-analysis
generator with two explicit input layouts. Its `--long-form` layout consumes
the completed joint-audit table directly, filters the exactly named raw and
global-spread methods, and pivots only on unique ISO `target_date`/`method`
pairs. It requires exactly 40 dates and both methods on every date, checks the
proper-score, deterministic, boundary and spatial case means against the
trusted aggregate anchors within `1e-10`, and only then writes paired date and
contiguous-block intervals. The legacy two-table layout remains available for
the earlier wide contract. Neither layout uses row order for pairing, and none
of the interval outputs is yet cited.

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

The compact CSV and aggregate JSON retain their trusted manifests outside the
paper narrative. Venue metadata, author statements and data-release decisions
remain external inputs.
