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
- Paired uncertainty: not reported because the audited compact contract lacks
  raw case-level fields; aggregate means are not used to synthesize pairs.

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

The trusted 40-row corrected-case table has an audited contract containing
`analysis_fair_crps`, `analysis_crps`, `analysis_spread_skill_ratio`, and
`analysis_coverage_90`, but no corresponding `raw_*` fields. The aggregate
summary separately reports corrected and raw means. Those means are sufficient
for aggregate reconciliation but cannot identify paired case differences.

Accordingly, `paper/make_case_level_artifacts.py` is retained as a guarded
generator for two richer future compact contracts: one raw case table and one
corrected case table. Both must carry the same unique ISO-date keys and the four
metrics below. The generator joins only by date, rejects unequal date sets and
never uses row order. It is not executable from the currently available table
and none of its outputs are cited. Do not synthesize raw case values from
aggregate means.

The corrected table is the positional input and the raw table is supplied via
`--raw-csv`. The generator requires an explicit ISO-date column through `--date-column`,
sorts cases by that column, rejects duplicate or malformed dates, and reports
both a paired case bootstrap and a paired circular contiguous-block bootstrap.
The block length is mandatory through `--block-length`, recorded in the JSON
output, and must be fixed scientifically before inspecting interval results. This
prevents an arbitrary CSV row order from being treated as temporal adjacency.

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
output directories or writing JSON/SVG artifacts. No paired interval or
improved-case count is part of the present evidence package.

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
