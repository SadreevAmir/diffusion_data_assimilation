# Publication readiness audit

Audit date: 2026-08-15

Publication status: NOT_READY

Required scientific blockers: case/block uncertainty, strong SIC calibration
baselines, spatial/physical preservation, clean checkpoint and frozen independent
evaluation

## Scientific readiness decision

The validation-set mechanism result is internally auditable, but it is not a
submission-ready strong domain/SciML paper. Narrowing the claim to one global
spread correction does not satisfy the minimum tier frozen in
`paper/RESEARCH_PLAN.md`. The earlier readiness decision is withdrawn.

The existing evidence supports a useful mechanism diagnosis: the learned-joint
ten-member ensemble is globally underdispersed and cross-fitted anomaly scaling
improves fair CRPS without moving the pre-clipping center. It does not yet
support spatial preservation, uncertainty of the score improvement, comparison
with strong SIC postprocessors, or independent generalization.

The global scaling is therefore frozen only as a reproducible reference
baseline, not as the selected calibrated-ensemble method. The joint stop/go gate
in `paper/RESEARCH_PLAN.md` now requires simultaneous proper-score,
finite-ensemble rank/coverage, boundary-mass and spatial/physical evidence.

## Minimum-tier gap audit

| Required element | Current evidence | Readiness consequence |
|---|---|---|
| Clean checkpoint and frozen independent evaluation | One legacy checkpoint and reused development dates | Blocking; method may be frozen, but the generalization claim is not tested |
| Exact deterministic comparison | Background and learned-joint aggregate RMSE/IIEE are available; no independent frozen comparison is claimed | Blocking for the main comparison table |
| Correct finite-ensemble diagnostics | Fair and ordinary CRPS, spread-skill, four coverage diagnostics and center invariance are reported | Satisfied for the narrow mechanism claim |
| Strong SIC calibration baselines | Only raw ensemble and a failed affine-logit transform are evaluated | Blocking; at least one boundary-aware distributional baseline and one rank-preserving or conformal baseline are required |
| Case/block uncertainty | The corrected compact table lacks paired raw fields | Blocking; aggregate means cannot identify paired date-level uncertainty |
| Spatial/physical preservation | Raw learned-joint IIEE is available, but calibrated IIEE, edge/area/extent or multivariate scores are absent | Blocking; a marginal score gain alone is insufficient |
| Publication figures | A reproducible aggregate mechanism figure is cited; uncertainty and spatial/physical figures remain unavailable | Partial; paired uncertainty and spatial/physical comparisons must be visualized after admissible compact artifacts exist |
| References and reproducibility | Core scoring and generative-model references plus a guarded reconciliation contract are present | Partial; baseline and sea-ice verification references must accompany the missing experiments |

## Fastest defensible next evidence

The next server analysis should reuse the existing learned-joint ensemble and
emit a compact paired date-level table containing both raw and corrected values
for fair CRPS and physical preservation metrics. Date and contiguous temporal
block bootstrap intervals must be computed on the server; pixels must never be
resampled as independent units. The same analysis should evaluate predeclared
boundary-aware and rank-preserving/conformal baselines if an implemented trusted
mode supports them.

The implemented `validation_existing_ensemble_calibration_audit` is now the
fastest admissible next analysis. Its reviewed server contract reuses the
existing learned-joint ensemble and fixes folds, random seed, calibration
candidates, proper scores, randomized-rank diagnostics, boundary diagnostics
and spatial diagnostics. It should be run before any new sampling. Repeating a
spread-only mode would reproduce an already known aggregate result and would
not close the minimum-tier blockers.

## Safe autonomous work completed or still possible

- The frozen method, aggregate reconciliation values, clipping caveat and
  negative affine-logit finding are recorded consistently.
- Unsupported paired intervals and improved-case counts are not reconstructed.
- A guarded artifact generator exists for separate compact raw and corrected
  tables. It requires the same explicit ISO-date keys in both inputs and a
  predeclared block length, then emits paired case-bootstrap and circular
  contiguous-block intervals; unequal date sets, malformed dates and duplicates
  fail closed. Before writing any artifact it also requires all four case means
  to reproduce the trusted aggregate summary within `1e-10`.
- Once an admissible compact paired table is returned, the guarded generator,
  claim ledger, result figure and manuscript uncertainty paragraph can be
  updated without accessing raw ensembles.
- A fail-closed generator for the aggregate CRPS, spread-skill and coverage
  figure is present and documented; it uses only the trusted one-row summary
  and does not imply paired uncertainty or spatial preservation.

## External boundary

The immediate next step is the allowlisted existing-ensemble calibration audit;
its result determines whether a candidate passes the predeclared joint gate or
whether the calibration claim must stop or be revised. The frozen independent
evaluation remains locked until explicit authorization after the pre-test
evidence package and clean checkpoint are complete.
