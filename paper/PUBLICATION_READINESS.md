# Publication readiness audit

Audit date: 2026-08-16

Publication status: NOT_READY

Required scientific blockers: an eligible spatially preserving calibration,
clean checkpoint and frozen independent evaluation

## Independent package re-audit

The 2026-08-16 independent re-audit checked the manuscript, claim ledger,
research plan, reproducibility handoff, readiness declaration, both linked SVG
figures and the local fail-closed checker. It found and corrected two stale
baseline-inventory statements: the claim ledger still described the evidence
as if only the hurdle-isotonic/ECC-Q comparator had completed, and the research
plan still marked zero/one-inflated Beta or EMOS-like postprocessing as missing.
Both now record the completed, rejected ZOIB-EMOS/ECC-Q result while retaining
the genuinely missing conformal, probabilistic-DA and independent deterministic
comparisons. The checker now requires these corrected inventory anchors.

The re-audit does not change the scientific decision. All available trusted
calibration routes are negative controls, no method passes the common gate, and
the minimum strong domain/SciML tier still lacks a clean checkpoint and frozen
independent evaluation. Publication status therefore remains `NOT_READY`.

## Scientific readiness decision

The validation-set mechanism result and completed joint calibration audit are
internally auditable, but this is not a
submission-ready strong domain/SciML paper. Narrowing the claim to one global
spread correction does not satisfy the minimum tier frozen in
`paper/RESEARCH_PLAN.md`. The earlier readiness decision is withdrawn.

The evidence supports a useful mechanism diagnosis: the learned-joint
ten-member ensemble is globally underdispersed and cross-fitted anomaly scaling
improves fair CRPS without moving the pre-clipping center. The completed audit
rejects spatial preservation. Paired date uncertainty now supports the fair-CRPS
improvement, and the post-hoc four-date-block sensitivity agrees; comparison
with strong SIC postprocessors and independent generalization remain absent.

The global scaling is therefore frozen only as a reproducible reference
and negative mechanism baseline, not as the selected calibrated-ensemble
method. The joint stop/go gate has been applied: proper scores improve, but
boundary and spatial/physical preservation fail. Established-ice Brier score
worsens by 2.15%, exact-one member mass rises to 0.164832 despite zero truth
mass, mean IIEE worsens by 5.50%, and edge disagreement worsens by 4.41%.

The targeted exact capped-simplex follow-up is also complete. It preserves the
raw ensemble mean and every audited mean-field/physical invariant to numerical
precision and retains a robust fair-CRPS gain, but remains ineligible. Boundary,
inner-order and member-spatial families fail: established-ice Brier worsens,
upper-cap mass reaches `0.167438`, and local/member variogram tolerances are
exceeded. Thus mean shift is ruled out as the cause of the remaining tradeoff;
exact mean preservation is not sufficient for a jointly reliable spatial
ensemble.

The final fixed open-logit diagnostic removes the capped projection's hard
upper-cap mass and passes proper-score, finite-ensemble reliability,
mean-field-invariant and operational families. It also improves fair CRPS to
`0.0557378` and inner-order error to `0.148343`. It is still ineligible:
established-ice Brier is `0.0591071` versus `0.0569726` raw, mass above `0.999`
is `0.0719362` versus `0.0108617`, and member-spatial safety fails. Boundary
and spatial damage therefore persist without a hard cap; the fixed diagnostic
is closed and no post-hoc tuning is admissible.

The frozen ZOIB-EMOS/ECC-Q baseline is now also complete and rejected. It
finishes all 40 cases with converged folds and zero ECC rank-order violations,
and sharply improves exact-boundary mass errors and randomized ranks. However,
fair CRPS is `0.0585570` versus `0.0584906` raw, ordinary CRPS worsens to
`0.0649912`, inner-order attainable error worsens to `0.253947`, established-
ice Brier worsens to `0.0642805`, and every spatial/physical criterion fails.
Only operational validity passes at family level. This closes the frozen
parametric boundary-aware route without post-hoc tuning; it does not supply an
eligible calibrated ensemble.

## Minimum-tier gap audit

| Required element | Current evidence | Readiness consequence |
|---|---|---|
| Clean checkpoint and frozen independent evaluation | One legacy checkpoint and reused development dates | Blocking; method may be frozen, but the generalization claim is not tested |
| Exact deterministic comparison | Background and learned-joint aggregate RMSE/IIEE are available; no independent frozen comparison is claimed | Blocking for the main comparison table |
| Correct finite-ensemble diagnostics | Fair and ordinary CRPS, spread-skill, four coverage diagnostics and center invariance are reported | Satisfied for the narrow mechanism claim |
| Strong SIC calibration baselines | Raw, affine-logit, global spread, purged hurdle-isotonic/ECC-Q, exact mean-preserving projected spread, open-logit desaturation and frozen ZOIB-EMOS/ECC-Q are evaluated; all postprocessors fail at least one mandatory family | The predeclared zero/one-inflated parametric gap is resolved as a strong negative control; no candidate satisfies the common gate, and conformal or probabilistic-DA comparisons remain absent |
| Case/block uncertainty | Trusted 40-date paired analysis reports fair-CRPS delta `-0.00280091`, date CI `[-0.00413219, -0.00135419]`, and circular four-date-block CI `[-0.00502644, -0.000290285]`; ordinary-CRPS intervals cross zero | Satisfied for the narrow mechanism claim; the block interval is a post-hoc temporal sensitivity, not a predeclared pass/fail gate |
| Spatial/physical preservation | The joint audit reports calibrated IIEE, edge, area/extent and variogram diagnostics; the candidate exceeds the IIEE and edge tolerances | Blocking for selection of global scaling, but resolved as an auditable negative mechanism result |
| Publication figures | Reproducible aggregate reliability and joint-gate figures plus manuscript uncertainty and mechanism tables are cited | Satisfied for the present mechanism claims; broader baseline and independent-evaluation figures depend on the blocking evidence |
| References and reproducibility | Core scoring, ECC and IIEE references plus a guarded reconciliation contract are present | Partial; method-specific references must accompany any still-missing baseline experiments |

## Fastest defensible next evidence

The completed joint audit rejects global anomaly scaling under the frozen gate,
and the trusted CPU result closes paired uncertainty without new sampling. The
fixed purged hurdle-isotonic bounded distribution with ECC-Q now supplies the
previously missing boundary-aware/rank-preserving mechanism comparison. It
repairs boundary masses and randomized ranks but fails proper-score and
spatial/physical families by large margins. The remaining minimum-tier gap is
therefore broader independent baseline coverage and a candidate that can
satisfy the no-compensation gate, not another tuning point. The completed
ZOIB-EMOS/ECC-Q baseline closes the parametric boundary-aware route: boundary
masses and ranks improve, but proper-score, inner-order, boundary and spatial
families fail. No additional implemented mode supplies a mechanistically
distinct eligible family.
Repeating completed analyses would not change the decision.
The final open-logit result additionally rules out hard upper-cap saturation as
the sole cause of failure. It improves the capped candidate's proper score and
inner-order reliability but still fails boundary and member-spatial families.

## Safe autonomous work completed or still possible

- The frozen method, aggregate reconciliation values, clipping caveat and
  negative affine-logit finding are recorded consistently.
- Paired intervals and improved-case counts are cited only from the completed
  trusted compact analysis, never reconstructed from aggregate means.
- The completed joint-audit schema is now distinguished from the earlier
  spread-only schema: its 160 long-form rows contain `target_date` and `method`
  for four methods. The completed trusted server analysis uses this contract;
  its compact intervals are cited without copying the case table into the
  publication worktree.
- The guarded artifact generator now accepts the completed 160-row long-form
  contract directly. It pivots exactly named methods by unique ISO
  `target_date`, requires both methods on all 40 dates and the predeclared
  four-case block length, and emits paired case-bootstrap and circular-block
  intervals only after proper-score, boundary and spatial means reproduce the
  trusted aggregate anchors within `1e-10`. Malformed dates, duplicate pairs,
  incomplete method pairs and provenance mismatches fail closed.
- The claim ledger, manuscript and readiness audit now record the failed joint
  gate and its boundary/spatial mechanism without accessing raw ensembles.
- The fixed purged hurdle-isotonic/ECC-Q result is recorded as a negative
  mechanism ablation: boundary-mass and rank diagnostics improve, while paired
  proper-score, IIEE, edge and energy diagnostics reject method selection.
- The exact mean-preserving projected-spread result is recorded across the
  manuscript, claim ledger and reproducibility trail. It isolates a robust
  fair-CRPS/rank gain from failed boundary, inner-order and member-spatial
  families without conflating the result with a shifted mean field.
- The final fixed open-logit result is recorded across manuscript, claim ledger
  and reproducibility trail. It removes hard upper-cap saturation and passes
  finite-ensemble reliability, while its failed boundary and member-spatial
  families close calibration development without post-hoc parameter changes.
- The frozen ZOIB-EMOS/ECC-Q result is recorded as a fifth negative mechanism:
  operational validity and marginal boundary/rank repair do not compensate for
  failed proper-score, inner-order, established-ice and spatial families.
- The claim ledger no longer describes the baseline evidence as raw plus
  affine-logit only, and the manuscript contributions now expose both sides of
  the boundary/rank-versus-spatial tradeoff without treating one failed
  construction as exhaustive baseline coverage.
- The manuscript evidence table reports the boundary and spatial failures next
  to the proper-score gains, including the improving lag-1/lag-2 variogram
  errors that are explicitly barred from compensating for failed mandatory
  families. This prevents the aggregate reliability figure from being read as
  the complete scientific decision.
- A fail-closed generator for the aggregate CRPS, spread-skill and coverage
  figure is present and documented; it uses only the trusted one-row summary
  and does not imply paired uncertainty or spatial preservation.
- A second fail-closed generator and checked-in figure expose the central joint-
  gate decision as candidate/raw ratios. They place the fair-CRPS gain beside
  boundary and spatial diagnostics without averaging across mandatory families.
- The manuscript now cites the primary ECC and IIEE references at first use;
  any future baseline family must add its own method-specific citation rather
  than inheriting authority from these references.
- A local fail-closed publication audit now checks required paper files,
  manuscript figure links, SVG parseability, contiguous numbered references and
  consistency between publication status and the blocker declaration. It also
  requires unique contiguous claim-ledger IDs and the decision-bearing rows of
  all five manuscript evidence tables, so dropping a negative mechanism result
  cannot silently leave the narrative intact. This is an integrity check and
  does not relax any scientific blocker.

## External boundary

The joint gate decisions are complete: global spread scaling, purged
hurdle-isotonic/ECC-Q, exact mean-preserving projected spread, fixed open-logit
desaturation and frozen ZOIB-EMOS/ECC-Q are rejected as
the paper's calibrated-ensemble method. The immediate external scientific
boundary is authorization of a clean checkpoint and the already frozen
independent evaluation, or specification and review of a genuinely new
mechanistically distinct calibration contract. All currently implemented
boundary/rank comparators have been completed and rejected.
The paired date/block CPU analysis of the existing compact audit is complete;
its compact result has been reconciled into the evidence chain, while local
reconstruction remains intentionally excluded. The currently implemented modes
cannot supply another independent strong baseline family beyond the completed
negative comparators. The frozen independent evaluation
also remains locked until explicit authorization after the pre-test evidence
package and clean checkpoint are complete.
