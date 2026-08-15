# Publication readiness audit

Audit date: 2026-08-15

Publication status: NOT_READY

Required scientific blockers: case/block uncertainty, strong SIC calibration
baselines, spatial/physical preservation, clean checkpoint and frozen independent
evaluation

## Scientific readiness decision

The validation-set mechanism result and completed joint calibration audit are
internally auditable, but this is not a
submission-ready strong domain/SciML paper. Narrowing the claim to one global
spread correction does not satisfy the minimum tier frozen in
`paper/RESEARCH_PLAN.md`. The earlier readiness decision is withdrawn.

The evidence supports a useful mechanism diagnosis: the learned-joint
ten-member ensemble is globally underdispersed and cross-fitted anomaly scaling
improves fair CRPS without moving the pre-clipping center. The completed audit
rejects spatial preservation; uncertainty of the score improvement, comparison
with strong SIC postprocessors, and independent generalization remain absent.

The global scaling is therefore frozen only as a reproducible reference
and negative mechanism baseline, not as the selected calibrated-ensemble
method. The joint stop/go gate has been applied: proper scores improve, but
boundary and spatial/physical preservation fail. Established-ice Brier score
worsens by 2.15%, exact-one member mass rises to 0.164832 despite zero truth
mass, mean IIEE worsens by 5.50%, and edge disagreement worsens by 4.41%.

## Minimum-tier gap audit

| Required element | Current evidence | Readiness consequence |
|---|---|---|
| Clean checkpoint and frozen independent evaluation | One legacy checkpoint and reused development dates | Blocking; method may be frozen, but the generalization claim is not tested |
| Exact deterministic comparison | Background and learned-joint aggregate RMSE/IIEE are available; no independent frozen comparison is claimed | Blocking for the main comparison table |
| Correct finite-ensemble diagnostics | Fair and ordinary CRPS, spread-skill, four coverage diagnostics and center invariance are reported | Satisfied for the narrow mechanism claim |
| Strong SIC calibration baselines | Only raw ensemble and a failed affine-logit transform are evaluated | Blocking; at least one boundary-aware distributional baseline and one rank-preserving or conformal baseline are required |
| Case/block uncertainty | The earlier 40-row spread-only table lacks paired raw fields; the later 160-row joint-audit contract contains `target_date` and `method`. The implemented trusted paired-uncertainty mode has been proposed, but no compact summary has been retrieved | Blocking until the returned date-paired intervals are reconciled; the circular four-date-block interval is a temporal sensitivity, not a predeclared pass/fail gate |
| Spatial/physical preservation | The joint audit reports calibrated IIEE, edge, area/extent and variogram diagnostics; the candidate exceeds the IIEE and edge tolerances | Blocking for selection of global scaling, but resolved as an auditable negative mechanism result |
| Publication figures | A reproducible aggregate mechanism figure is cited; uncertainty and spatial/physical figures remain unavailable | Partial; paired uncertainty and spatial/physical comparisons must be visualized after admissible compact artifacts exist |
| References and reproducibility | Core scoring and generative-model references plus a guarded reconciliation contract are present | Partial; baseline and sea-ice verification references must accompany the missing experiments |

## Fastest defensible next evidence

The completed joint audit closes the previous diagnostic unknown and rejects
global anomaly scaling under the frozen gate. The already proposed trusted CPU
analysis is the fastest route to paired uncertainty for the four audited fixed
methods; it requires no new sampling. After that summary, the remaining
minimum-tier scientific evidence is not another tuning point: it is a
predeclared strong SIC postprocessing comparison containing at least one
boundary-aware distributional baseline and one rank-preserving or conformal
baseline. No currently implemented trusted mode provides that baseline family,
so proposing an invented mode would be invalid. Repeating either spread-only
analysis or the joint audit would not change the decision.

## Safe autonomous work completed or still possible

- The frozen method, aggregate reconciliation values, clipping caveat and
  negative affine-logit finding are recorded consistently.
- Unsupported paired intervals and improved-case counts are not reconstructed.
- The completed joint-audit schema is now distinguished from the earlier
  spread-only schema: its 160 long-form rows contain `target_date` and `method`
  for four methods. This makes a server-side paired analysis possible in
  principle, but does not make an interval available in the publication
  worktree.
- The guarded artifact generator now accepts the completed 160-row long-form
  contract directly. It pivots exactly named methods by unique ISO
  `target_date`, requires both methods on all 40 dates and the predeclared
  four-case block length, and emits paired case-bootstrap and circular-block
  intervals only after proper-score, boundary and spatial means reproduce the
  trusted aggregate anchors within `1e-10`. Malformed dates, duplicate pairs,
  incomplete method pairs and provenance mismatches fail closed.
- The claim ledger, manuscript and readiness audit now record the failed joint
  gate and its boundary/spatial mechanism without accessing raw ensembles.
- The manuscript evidence table reports the boundary and spatial failures next
  to the proper-score gains, including the improving lag-1/lag-2 variogram
  errors that are explicitly barred from compensating for failed mandatory
  families. This prevents the aggregate reliability figure from being read as
  the complete scientific decision.
- A fail-closed generator for the aggregate CRPS, spread-skill and coverage
  figure is present and documented; it uses only the trusted one-row summary
  and does not imply paired uncertainty or spatial preservation.

## External boundary

The joint gate decision is complete: global spread scaling is rejected as the
paper's calibrated-ensemble method. The immediate external scientific boundary
is implementation and review of a trusted strong-baseline analysis contract.
The paired date/block CPU analysis of the existing compact audit is implemented
and already proposed, but its summary has not yet returned; local reconstruction
is intentionally excluded. The currently implemented modes cannot supply
boundary-aware distributional and rank-preserving/conformal comparators. The
frozen independent evaluation
also remains locked until explicit authorization after the pre-test evidence
package and clean checkpoint are complete.
