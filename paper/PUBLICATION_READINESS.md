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

## Minimum-tier gap audit

| Required element | Current evidence | Readiness consequence |
|---|---|---|
| Clean checkpoint and frozen independent evaluation | One legacy checkpoint and reused development dates | Blocking; method may be frozen, but the generalization claim is not tested |
| Exact deterministic comparison | Background and learned-joint aggregate RMSE/IIEE are available; no independent frozen comparison is claimed | Blocking for the main comparison table |
| Correct finite-ensemble diagnostics | Fair and ordinary CRPS, spread-skill, four coverage diagnostics and center invariance are reported | Satisfied for the narrow mechanism claim |
| Strong SIC calibration baselines | Only raw ensemble and a failed affine-logit transform are evaluated | Blocking; at least one boundary-aware distributional baseline and one rank-preserving or conformal baseline are required |
| Case/block uncertainty | The corrected compact table lacks paired raw fields | Blocking; aggregate means cannot identify paired date-level uncertainty |
| Spatial/physical preservation | Raw learned-joint IIEE is available, but calibrated IIEE, edge/area/extent or multivariate scores are absent | Blocking; a marginal score gain alone is insufficient |
| Publication figures | No generated result figure is cited by the manuscript | Blocking; uncertainty and spatial/physical comparisons must be visualized after admissible compact artifacts exist |
| References and reproducibility | Core scoring and generative-model references plus a guarded reconciliation contract are present | Partial; baseline and sea-ice verification references must accompany the missing experiments |

## Fastest defensible next evidence

The next server analysis should reuse the existing learned-joint ensemble and
emit a compact paired date-level table containing both raw and corrected values
for fair CRPS and physical preservation metrics. Date and contiguous temporal
block bootstrap intervals must be computed on the server; pixels must never be
resampled as independent units. The same analysis should evaluate predeclared
boundary-aware and rank-preserving/conformal baselines if an implemented trusted
mode supports them.

The currently implemented cross-fit spread modes do not, by their audited
compact contract, provide those baselines or paired physical metrics. Repeating
the same mode would reproduce an already known aggregate result and would not
close the minimum-tier blockers. A new mode cannot be invented in a proposal;
the trusted implementation must first extend an allowlisted analysis contract.

## Safe autonomous work completed or still possible

- The frozen method, aggregate reconciliation values, clipping caveat and
  negative affine-logit finding are recorded consistently.
- Unsupported paired intervals and improved-case counts are not reconstructed.
- A guarded artifact generator exists and fails closed when paired raw columns
  are absent. It now requires an explicit ISO-date column and predeclared block
  length, then emits both paired case-bootstrap and circular contiguous-block
  intervals; malformed or duplicate dates fail closed.
- Once an admissible compact paired table is returned, the guarded generator,
  claim ledger, result figure and manuscript uncertainty paragraph can be
  updated without accessing raw ensembles.

## External boundary

The immediate boundary is not editorial review. It is the absence of an
allowlisted server-analysis contract that can return the required paired
uncertainty, baseline and spatial/physical evidence from existing ensembles.
The frozen independent evaluation remains locked until explicit authorization
after the pre-test evidence package and clean checkpoint are complete.
