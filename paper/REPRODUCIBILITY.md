# Reproducibility handoff for the frozen mechanism result

This handoff covers only the 40-case development-period mechanism analysis in
the paper. It consumes the compact per-case table from the completed trusted
cross-fitted spread run; it does not require or permit raw ensembles.

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
- Paired uncertainty: deterministic percentile bootstrap over the 40 cases,
  with 20,000 replicates and seed `20220815`.

## Compact-artifact generation

The compact table is not currently present in this worktree, and the aggregate
summary does not establish its row-level schema. Before running the generator,
inspect the retrieved CSV header and require these exact paired fields:

- `analysis_fair_crps`, `raw_analysis_fair_crps`;
- `analysis_crps`, `raw_analysis_crps`;
- `analysis_spread_skill_ratio`, `raw_analysis_spread_skill_ratio`;
- `analysis_coverage_90`, `raw_analysis_coverage_90`.

Do not claim that local generation is executable until this header check passes.
If it passes, place the trusted compact table at
`paper/artifacts/per_case_metrics.csv`, then run:

```bash
python3 paper/make_case_level_artifacts.py \
  paper/artifacts/per_case_metrics.csv \
  --summary paper/artifacts/case_level_summary.json \
  --figure paper/figures/fair_crps_case_deltas.svg
```

The generator itself repeats the column check and rejects an unexpected case
count, duplicate case identifiers when that column is present, non-finite
metrics, and an undersized bootstrap.
It creates output directories as needed and records the input SHA-256 digest in
the JSON summary.

## Required reconciliation checks

Before citing the generated outputs, verify that their case means reproduce the
trusted aggregate summary to ordinary floating-point tolerance:

| Metric | Raw | Corrected |
|---|---:|---:|
| Fair CRPS | 0.0584905850 | 0.0556896736 |
| Ordinary CRPS | 0.0621082810 | 0.0611013421 |
| Spread-skill ratio | 0.7240662110 | 1.0614835127 |
| 90% interval diagnostic | 0.5050835783 | 0.8787749039 |

Then record the paired interval and improved-case count in the manuscript and
claim ledger. Treat a mismatch larger than `1e-10` in any listed case mean as a
provenance or schema failure: do not publish the artifact and do not repair it
by rounding or manual editing.

## Scope and audit trail

The JSON summary and SVG are derived publication artifacts. The compact CSV is
the evidence input and must retain its trusted retrieval manifest outside the
paper narrative. The paper must not describe the paired case bootstrap as an
independent-period uncertainty estimate. Venue metadata, author statements and
data-release decisions remain external inputs.
