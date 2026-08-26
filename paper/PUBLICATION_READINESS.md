# Publication readiness audit

Audit date: 2026-08-27

Publication status: NOT_READY

Required scientific blockers: an eligible spatially preserving calibration and
the remaining minimum-tier comparisons

External primary evidence state: RECONCILED_NEGATIVE

Controller readiness: NOT_READY
Scientific primary reconciliation: COMPLETE_NEGATIVE
Autonomous publication work remaining: YES

## Decision-bearing readiness blocker matrix

This table is normative for the current `NOT_READY` decision. Its evidence
states are derived from the minimum-tier audit rather than editorial judgment;
none of the four rows may be silently removed or treated as compensating for
another row.

| Required blocker | Normative source | Current evidence state | Exact closure condition |
|---|---|---|---|
| Eligible spatially preserving calibration | `RESEARCH_PLAN.md` | MISSING_ELIGIBLE_RESULT | One frozen candidate passes every mandatory no-compensation gate family |
| Conformal intervals | `MINIMUM_TIER_COMPARISON_AUDIT.md` | MISSING | The corresponding normative comparison row becomes decision-bearing after valid trusted execution |
| Probabilistic DA baseline such as EnKF/LETKF | `MINIMUM_TIER_COMPARISON_AUDIT.md` | MISSING | The corresponding normative comparison row becomes decision-bearing after valid trusted execution |
| Independent-strength deterministic background and 3D-Var | `MINIMUM_TIER_COMPARISON_AUDIT.md` | PRESENT_DEVELOPMENT_ONLY | A frozen common-information comparison supplies evidence at the manuscript's required independent strength |

## Unique cross-artifact closure routes

Each blocker has exactly one frozen contract and one insertion route into the
manuscript and normative claim ledger. A valid negative result may close a
baseline-family evidence row where its contract says so, but cannot close the
eligible-calibration row. An invalid execution closes nothing. These routes do
not authorize an unimplemented mode or another independent evaluation.

| Required blocker | Frozen contract | Manuscript target | Claim-ledger target |
|---|---|---|---|
| Eligible spatially preserving calibration | `NEXT_RANK_COHERENT_CONTRACT.md` | PAPER_DRAFT.md:Section 6 mechanism result and no-compensation decision | CLAIM_LEDGER.md:new decision-bearing calibration claim |
| Conformal intervals | `NEXT_CONFORMAL_BASELINE_CONTRACT.md` | PAPER_DRAFT.md:Section 3 frozen outcome matrix | CLAIM_LEDGER.md:C17 plus a decision-bearing conformal claim |
| Probabilistic DA baseline such as EnKF/LETKF | `NEXT_PROBABILISTIC_DA_COMPARISON_CONTRACT.md` | PAPER_DRAFT.md:Section 3 frozen outcome matrix | CLAIM_LEDGER.md:C17 plus a decision-bearing probabilistic-DA claim |
| Independent-strength deterministic background and 3D-Var | `FROZEN_EVALUATION_HANDOFF.md` | PAPER_DRAFT.md:Section 3 common-information comparison | CLAIM_LEDGER.md:C9 and C17 evidence-strength transition |

The 2026-08-27 key-claim audit now derives the two `MISSING` baseline families
from `MINIMUM_TIER_COMPARISON_AUDIT.md` and requires both to remain explicit in
Abstract, Contributions and Conclusion. The manuscript now contains a dedicated
Conclusion with the same non-compensation boundary. Four focused fixtures accept
the current text and reject omission from Abstract, promotion in Contributions,
or removal of Conclusion. This closes an editorial fail-closed gap without
creating either missing result; status remains `NOT_READY`.

The 2026-08-27 claim-status consistency audit derives every `Unknown` and
`Rejected` row directly from the normative claim ledger. The unified checker
now requires every unknown claim to remain in the unsupported-scope row and
outside the empirical evidence table, while every rejected claim must have a
decision-bearing manuscript presentation with an explicit negative
interpretation. Three negative fixtures reject promotion of an unknown claim,
loss of a rejected claim and neutralization of its decision language. This
closes the declared cross-artifact consistency gap without adding scientific
evidence; status remains `NOT_READY`.

The 2026-08-27 limitation-traceability audit maps all eight distinct Section 7
limitations to exact claim-ledger rows. The two genuine evidence absences point
to the existing `Unknown` records C9 and C10 rather than being promoted to
results. `LIMITATION_TRACEABILITY.md` is now required by the unified checker,
and three negative fixtures reject a missing limitation, a changed claim mapping
or a missing evidence-absence status. This closes the declared limitation-audit
gap without changing any scientific result; status remains `NOT_READY`.

The 2026-08-27 reference-traceability audit independently checked all six
bibliography records against primary publisher or conference metadata and maps
each item to one scope-limited manuscript statement in
`REFERENCE_TRACEABILITY.md`. It found two overextensions and corrected them:
references 1--2 no longer implicitly source stochastic interpolants or the
project's spatial resolution, and the fair-score sentence no longer presents a
study-design choice as a universal requirement. Four negative regression
fixtures now reject a missing row, changed identity or missing empirical-claim
boundary. The unified checker requires the artifact and suite. This closes the
declared bibliographic gap without changing any empirical result; status remains
`NOT_READY`.

The 2026-08-27 figure-generator audit found that finite compact inputs could
still reach undefined rendering paths: jointly zero calibration values produced
a zero plotting scale, and a zero raw joint-gate metric produced an undefined
ratio. Both generators now reject those inputs explicitly, reject negative
metrics consistently, and have deterministic compact-fixture regression tests.
The unified checker requires and executes that suite. This closes a concrete
figure-reproducibility gap but adds no scientific result, so publication status
remains `NOT_READY`.

The 2026-08-27 pre-result interpretation audit now freezes in `PAPER_DRAFT.md`
all four valid joint outcomes of the two outstanding minimum-tier contracts.
Each conformal and probabilistic-DA result closes only its exact evidence row;
invalid execution leaves that row `MISSING`, and no combination can establish
learned-joint calibration eligibility or compensate for a failed gate family.
`REPRODUCIBILITY.md` records the same logical boundary and the unified checker
requires the decision labels, all four combinations and the non-compensation
interpretation. This prevents post-hoc narrative promotion but supplies neither
missing result, so publication status remains `NOT_READY`.

The 2026-08-27 conformal-baseline design audit freezes the fastest outstanding
minimum-tier comparison in `NEXT_CONFORMAL_BASELINE_CONTRACT.md`. It uses five
contiguous purged folds, the raw ten-member sea-ice-area range, the exact
finite-sample split-conformal quantile and pre-result coverage/width stop-go
criteria. The unified checker fails if these anchors or the explicit
`FROZEN_NOT_EXECUTABLE` boundary are weakened. Because no admitted trusted mode
implements this literal contract, the conformal row remains `MISSING` and no
experiment is proposed under an invented identifier.

The 2026-08-27 minimum-tier comparison audit now maps all nine required baseline
families to a concrete manuscript presentation and an existing compact
publication source. The unified checker requires the exact family/status map,
rejects duplicate or missing rows, verifies every source stays inside `paper/`,
and retains exactly two explicit `MISSING` families: conformal intervals and a
probabilistic DA comparator. It separately labels the deterministic comparison
as development-only. This makes the remaining baseline blocker exact and
fail-closed, but supplies neither missing experiment nor eligible calibration;
publication status remains `NOT_READY`.

The 2026-08-27 empirical-traceability negative-fixture audit now exercises the
production validator directly. Four focused tests accept the current map and
independently reject missing empirical-claim coverage, a presentation without a
named section or figure, and a compact-source path that escapes `paper/`. The
unified checker requires and executes this suite, so its fail-closed guarantee
is regression-tested rather than inferred from the successful current tree.
This closes a reproducibility gap but creates neither a scientific result nor
an admitted trusted mode, so publication status remains `NOT_READY`.

The 2026-08-27 empirical-traceability audit now links every empirical claim row
to both a concrete manuscript table, figure or named results paragraph and an
existing compact publication source. The unified checker requires exact,
duplicate-free coverage of C3--C8 and C11--C39, requires a concrete presentation
anchor in every row, and rejects sources that are missing or escape `paper/`.
This prevents an editorial revision from retaining a numerical or decision
claim after its presentation or compact evidence unit has disappeared. It does
not add a scientific result or an admitted trusted mode, so status remains
`NOT_READY`.

The 2026-08-27 negative-fixture audit now proves the immutable-identity boundary
fail-closed rather than relying only on a successful current-tree check. A
focused suite mutates each of the four frozen runner, prototype-test, oracle and
contract files independently and requires the unified checker to reject every
single mismatch; it separately rejects a handoff table that documents the wrong
digest. The unified publication audit executes this suite on every run. This
closes the remaining local provenance-test gap but creates neither a reviewed
trusted mode nor a scientific result, so publication status remains `NOT_READY`.

The 2026-08-26 immutable-identity audit found that the controller handoff still
recorded a stale SHA-256 for `rank_coherent_reference.py` after the admission
validator was added. The handoff now records the current digest, and the unified
publication checker computes and verifies all four frozen runner, test, oracle
and contract digests and requires the same values in the handoff table. This
closes a concrete reviewed-mode provenance risk but does not create an admitted
mode or scientific result; publication status remains `NOT_READY`.

The 2026-08-26 mechanism-table audit added a single family-level matrix for the
late frozen mechanisms that support the manuscript's central failure-map claim.
It distinguishes complete family passes from partial metric improvements and
marks the locked-MC-dropout families `NR`, because only its trusted overall
Boolean is present locally. This makes the negative comparison auditable without
inventing unavailable effect sizes or failure causes. It improves manuscript
traceability but does not change `NOT_READY` or remove any scientific blocker.

The 2026-08-26 table-integrity audit made the publication checker fail closed
on malformed Markdown evidence tables. It now verifies header/separator widths,
non-empty headers and body cells, constant row widths, non-empty bodies, at
least eight manuscript tables and exactly one normative claim-ledger table.
This guards the rendered evidence and claim map against silent column drift.
It is a publication-quality improvement only: it does not create a trusted
rank-coherent mode, recover missing quantitative dropout fields or change
`NOT_READY`.

The 2026-08-26 blocker-inventory audit corrected a stale minimum-tier row that
still described independent evaluation as absent. The single frozen independent
primary is complete and negative for its candidate; it does not establish
successful generalization of an eligible calibration. The remaining blockers
are therefore an eligible development calibration, broader independent
comparison coverage and the exact deterministic comparison, not absence of the
already reconciled primary. The unified checker now rejects the superseded row
and requires this distinction. This consistency repair does not change
`NOT_READY`.

The 2026-08-26 scope audit corrected one stale pre-primary sentence in the
manuscript. The contribution section had said that independent evaluation was
outside the present claim even though the reconciled negative independent
primary is now a central result. It now distinguishes that completed
falsification of one frozen candidate from the still unsupported claim of
successful independent generalization. The unified checker requires the new
scope statement and rejects the superseded wording. This removes an internal
contradiction but does not change `NOT_READY` or any scientific gate.

The idle-state audit also froze a contingent method in
`NEXT_RAW_MEMBER_REWEIGHTING_CONTRACT.md` rather than waiting for integration of
the first rank-coherent runner. It tests probability-mass error among unchanged
complete raw scenarios by purged analog truth-rank reweighting. Exact copied
fields and boundary masks are hard invariants; the unchanged rank, proper-score,
mean-field spatial/physical and operational families remain no-compensation
requirements. This is pre-result design evidence only, is activated solely by a
negative completed whole-field transport gate, and does not create a trusted
mode or change `NOT_READY`.

The 2026-08-26 independent publication-readiness rerun completed the unified
fail-closed audit and the focused rank-coherent handoff suites against the
current worktree. The audit passed with 44 required files, two figures, six
references and all 39 claim-ledger rows traced; the five prototype tests and
the 13-test adapter-parity suite also passed. `git diff --check` found no
whitespace errors. This confirms internal consistency and executable handoff
integrity, but it does not create decision-bearing scientific evidence. The
status therefore remains `NOT_READY`: the rank-coherent construction still
lacks a separately reviewed literal trusted mode and the minimum-tier
scientific blockers below remain unchanged.

The 2026-08-23 independent table audit found one stale scope sentence after the
reconciled primary had been added: the evidence section still said that no
independent-period study was presented as a contribution. The manuscript now
states the narrower, accurate scope and includes a compact independent-primary
table with fair CRPS, absolute normalized rank, all three member-semivariogram
errors and the negative no-compensation decision. The unified checker requires
the table and conclusion, preventing a later edit from retaining the proper-score
gain while silently dropping the rank or spatial failures. This closes a
publication-consistency gap but does not change `NOT_READY` or scientific
eligibility.

The independent manuscript-to-ledger audit found that the ledger declared
itself normative, but the manuscript contained no executable claim-level
traceability map. `PAPER_DRAFT.md` now maps every empirical and scope section to
all 39 ledger rows, including rejected and unknown claims, and the unified
checker fails on a missing, duplicated or untraced claim ID. This prevents a
later edit from silently promoting an unsupported statement while leaving the
ledger unchanged. It improves publication auditability but does not remove the
scientific blockers below.

The next autonomous mechanism is now frozen in
`NEXT_RANK_COHERENT_CONTRACT.md`. It transports complete member-anomaly fields
selected by purged forecast-only analogs, targets case-level rank dispersion,
uses training-only feasibility, and retains the unchanged no-compensation gate.
It is not yet a scientific result and has no implemented trusted mode; the
contract explicitly prevents pixelwise marginal tuning or reopening the closed
independent evaluation.

The local engineering handoff is now executable: `rank_coherent_reference.py`
checks the exact five purged folds, training-only six-feature standardization,
ten-neighbor forecast-only selection, deterministic date/member/distance tie
ordering, the exact `valid`, 40-case, ten-member, stride-five development
envelope, finite shape-compatible complete-field rank pairing, bounded normalized
truth ranks, frozen alpha selection, sole runner
parameter, full spatial candidate construction, physical bounds and the `1e-10`
per-pixel capped-simplex mean invariant on synthetic inputs. It now also rejects
missing or non-boolean mandatory gate families and any `overall_eligible` value
that is not their exact conjunction. The publication
handoff also fails closed on incomplete or reordered fold selections, any mismatch
between `selected_alpha` and `no_positive_feasible_alpha`, rank-target counts
other than ten uses of 40, projection fractions outside `[0,1]`, non-finite
diagnostics, missing or extra diagnostic fields, and maximum projection mean error
above `1e-10`. These checks make fold selection, target balance and bounded
projection accounting executable rather than narrative-only requirements. The
same oracle now rejects aggregate deltas that do not equal candidate minus
raw, paired summaries that do not reconcile with the same aggregate metric set,
non-finite or unordered uncertainty intervals, member-spatial decisions that do
not follow their frozen absolute tolerances, and family flags that differ from
their complete criterion conjunctions. Thus the full numeric compact handoff is
fail-closed without presuming a positive scientific outcome. The publication
audit requires, parses and executes this oracle, and fails closed unless it
prints the exact success sentinel. This is implementation evidence only:
until a separately reviewed trusted mode executes the full server-side gate,
the method remains untested and publication status remains `NOT_READY`.

The 2026-08-23 publication-readiness re-audit found that this new executable
handoff was still outside the unified fail-closed checker. The checker now
requires the prototype runner, its synthetic suite, and the exact controller
handoff; parses both Python files; executes all five synthetic tests; executes
the independent rank-coherent oracle; and verifies immutable handoff anchors,
including the recorded runner/test digests, sole scientific argument, four-file
compact output contract, and non-decision-bearing boundary. The unified audit
now covers 26 required files and fails if the handoff is removed, syntactically
damaged, behaviorally broken, or narratively loosened. This closes a concrete
reproducibility gap but does not change scientific readiness: the prototype
still cannot produce decision-bearing evidence without the separately reviewed
trusted full-gate integration.

`RANK_COHERENT_RUNNER_REVIEW_CHECKLIST.md` now supplies the missing independent
integration admission test. It traces the sole-parameter interface, provenance,
fold/purge and tie parity, complete-field construction, frozen alpha selection,
projection invariant, compact evidence, full gate, negative fixtures and a
synthetic server dry run. Every item is mandatory and any waiver is `NO_GO`;
only a recorded independently reviewed implemented mode can make a proposal
admissible. This is engineering readiness, not scientific evidence, so the
publication status remains `NOT_READY`.

The controller handoff now also has an exact fail-closed admission-record
schema. It requires the literal reviewed mode, reviewed publication commit,
runner/contract/synthetic SHA-256 digests, exact test command and sentinel,
`decision_bearing_validation=PASS`, and an empty deviations list. Placeholders,
missing fields or deviations remain `NO_GO`. This removes ambiguity from the
next controller-to-proposal transition without asserting that the review or the
scientific experiment has occurred; status remains `NOT_READY`.

That admission schema is now enforced by `validate_admission_record` in the
dependency-free oracle. Its self-test accepts one structurally complete record
and rejects a placeholder mode, malformed identities, a failed decision-bearing
validation and any deviation. This does not assert that a controller record
currently exists.

The explicit file boundary is now executable through
`validate_rank_coherent_admission.py`: it reads only the path supplied by the
controller, requires a JSON object, delegates the exact schema decision to the
same oracle and emits only `reviewed_mode=<literal reviewed mode>` on success.
Focused tests cover both the literal-mode success path and fail-closed rejection
of a non-object record or any deviation. No controller-visible admission record
is present in this worktree, so this closes handoff ambiguity but neither admits
an experiment nor changes `NOT_READY`.

The rank-coherent handoff now includes an executable directory-level parity
fixture and a frozen adapter replacement specification. The oracle validates all
four compact files together, reconciles all 80 per-case metric rows with their
aggregate means, and fails on status/gate, schema, envelope or decision-bearing
role drift. This closes the local adapter-parity preparation task, but it remains
implementation evidence only: no separately reviewed controller mode exists,
so no experiment is proposed and publication status remains `NOT_READY`.

The frozen independent primary is now reconciled from the exact permitted CPU
truth-normalization recovery
`external_2024_calendar_global_bias_confirm48_primary_retry2`; all superseded
`retry1` metrics are excluded. All 48 cases, 480 member records, signed identities, input/raw/truth
seals, hashes and candidate-before-truth ordering pass. The scientific decision
is negative: `overall_eligible=false`. Proper scores, absolute rank reliability,
truth-relative boundary calibration and spatial/physical preservation fail;
only operational validity passes. Fair CRPS worsens from
`0.05541808434196047` raw to `0.061833300537408264` candidate. Normalized mean
rank also worsens from `0.23610946912844127` raw to approximately `0.225`
candidate, farther from `0.5`. No resubmission, retuning or second external
evaluation is admissible.

The success definition has been amended fail-closed in
`AMENDED_PRIMARY_EVALUATION_CONTRACT.md`. A good randomized rank histogram is
an absolute requirement, high-SIC behaviour is compared with truth at
`q={0,.15,.90,.95,.99}`, and `>=.999`/exact-one masses are encoding diagnostics
rather than boundary acceptance criteria. Consequently, no historical
`overall_eligible` from the superseded gate is final success. The trusted
controller deployed and audited the exact amended primary and gate digests, and
the single signed chain has now completed through its permitted recovery. Its
four compact artifacts establish both admissible provenance and the reconciled
negative scientific outcome reported above.

## Current decision-bearing evidence

The complete compact result for the final frozen development fallback,
`crossfit_iid_calendar_global_bias_mixture_valid`, is now reconciled. It is
negative under the unchanged joint gate: `overall_eligible=false`. Reliability,
boundary, spatial/physical and operational families pass, but proper scores do
not. Fair CRPS improves only 0.35% (`0.0584905850` to `0.0582878868`), below the
3% threshold, and both the paired date interval
`[-0.00100456, 0.000700186]` and four-case-block sensitivity interval
`[-0.00139050, 0.00129663]` include zero. Ordinary CRPS worsens slightly. The
frozen iid mixture therefore closes as a negative mechanism result; it does not
authorize post-hoc changes to probability, bandwidth, residuals, folds or seeds.

All currently selectable frozen development calibration mechanisms have now
failed at least one mandatory family. The paper can defend a broad negative
mechanism map and a fail-closed evaluation protocol, but it cannot claim an
eligible calibrated ensemble. Publication status remains `NOT_READY`; the
remaining scientific gap cannot be erased by editorial narrowing of the minimum
paper tier.

The package is not waiting on clean-checkpoint training. Measured throughput
makes that previously frozen three-seed route a months-long construction. The
single frozen latent-temperature construction at scale `1.30` has been
recovered from the completed internal sampler without GPU recomputation, and
its complete compact payload from `latent_temperature_1p30_gate_retry1` has
been reconciled. The result is negative: `overall_eligible=false`; reliability
passes, while proper-score, boundary and spatial/physical families fail. The
mechanistically distinct frozen locked-MC-dropout route is also complete. Its
wrapper-only schema failure was recovered without GPU recomputation, and the
dependent trusted compact gate identifies the exact frozen candidate
`locked_mc_dropout_p010_final_ema_ensemble` with `overall_eligible=false`.
That negative decision is the contractual prerequisite that activated the final
cross-fitted iid calendar global-bias fallback; no dropout probability, layer
set, mask construction or refresh rule was changed after the result.

An independent evidence-unit audit found that the dedicated dropout
reconciliation file still declared a pre-result state even though the exact
candidate and negative overall decision had already been admitted elsewhere.
It now distinguishes the controller-recorded Boolean decision from the still
missing quantitative publication payload. Raw/candidate proper scores, paired
intervals and explicit mandatory-family decisions are not present in this
worktree, so no family-specific or effect-size claim is admitted for this
mechanism. This is a publication-reconciliation blocker, not a reason to repeat
sampling or to tune the frozen construction.

This ordering does not waive the minimum publication tier. The negative latent-
temperature result activated only the already predeclared locked-MC-dropout
fallback, unchanged; it did not authorize another temperature or retrospective
tuning. The completed negative dropout result closes only this frozen candidate.
The subsequently completed primary supplies independent negative evidence for
one frozen candidate, but not an eligible calibration, a publication-grade
independent deterministic comparison or the missing broader baseline comparisons.

## Independent package re-audit

The 2026-08-16 independent re-audit checked the manuscript, claim ledger,
research plan, reproducibility handoff, readiness declaration, both linked SVG
figures and the local fail-closed checker. It found and corrected two stale
baseline-inventory statements: the claim ledger still described the evidence
as if only the hurdle-isotonic/ECC-Q comparator had completed, and the research
plan still marked zero/one-inflated Beta or EMOS-like postprocessing as missing.
Both now record the completed, rejected ZOIB-EMOS/ECC-Q result while retaining
the genuinely missing conformal and probabilistic-DA comparisons and the
insufficient development-only deterministic comparison. The checker now requires
these corrected inventory anchors.

A subsequent independent integrity pass also closed a reproducibility hole in
the checker itself. The audit now requires all four publication generators,
parses each as Python before accepting the package, and checks semantic text and
numeric anchors in both checked-in SVG figures in addition to XML well-formedness.
Consequently, a missing or syntactically damaged generator, or a well-formed but
stale replacement figure, fails the publication audit instead of silently
passing on filename alone.

The re-audit does not change the scientific decision. All completed trusted
calibration routes are negative controls and no method passes the common gate.
The frozen independent primary is complete but negative; the minimum strong
domain/SciML tier still lacks an eligible method and the required independent
deterministic and broader baseline comparisons. Publication status therefore
remains `NOT_READY`.

The manuscript contribution inventory was also reconciled with the completed
mechanism suite. It no longer labels the evidence as five negative mechanisms
while later sections report topology-preserving transport, analog residual
dressing, guidance mixing and two coherent-offset constructions. The revised
contribution states the distinct failure modes and explicitly avoids claiming
that the implemented suite exhausts all calibration families. The local checker
now fails if this scope correction is lost.

The title and abstract were independently tightened to match that evidentiary
scope. They now present an audit of reliability rather than implying that a
reliable calibrated ensemble has already been obtained, summarize the full
mechanistically diverse negative suite, and state explicitly that the current
contribution is a failure map and fail-closed protocol. The checker requires
these scope anchors so a later edit cannot silently restore a positive method
claim before an eligible result exists.

The reproducibility handoff was then audited section by section. A misplaced
`Compact-artifact contract` heading had made the latent-temperature compact
result appear under the analog-residual section even though all numeric content
was correct. The result is now colocated with its recovery provenance, the
compact-artifact heading immediately precedes the two schema contracts, and the
checker enforces both semantic section order and absence of latent-temperature
evidence from the analog-residual section.

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

The claim-level consistency audit distinguishes evidence absence from evidence
strength. Exactly two baseline-family result rows are `MISSING`: conformal and
probabilistic DA. The deterministic row is `PRESENT_DEVELOPMENT_ONLY`, so it is
not a third missing family, but it still blocks the main comparison table because
no independent frozen deterministic comparison is claimed. Closing either of
the two missing rows cannot upgrade that deterministic evidence or make a
learned-joint calibration eligible.

| Required element | Current evidence | Readiness consequence |
|---|---|---|
| Checkpoint and independent generalization evidence | One legacy checkpoint plus one completed frozen independent primary that is negative for its candidate | Blocking for a successful generalization claim: the completed primary falsifies one candidate but does not establish an eligible calibration or broader checkpoint/seed robustness |
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

The final frozen topology-preserving stratified-transport contract has now been
executed. It leaves boundary atoms and
both memberwise ice-event masks unchanged, preserves the pixelwise ensemble
mean, and expands only within raw physical strata. Purged training folds select
among six frozen strengths under proper-score, reliability and stricter
member/local-spatial feasibility constraints. The held-out result is negative:
boundary and spatial/physical families pass with exact masks and maximum mean
error at most `5e-13`, but all three reliability criteria, both fair-CRPS
criteria and all-fold scale feasibility fail. `overall_eligible=false`; this is
negative mechanism evidence, not a selected calibration. Raw and candidate
fair CRPS are both `0.0584905850`, and raw and candidate spread-skill are both
`0.7240662110`; the conclusion therefore does not rest only on threshold labels.

The exact frozen purged analog-residual contract has now been executed and is
rejected. It improves all three finite-ensemble reliability criteria, but fair
CRPS worsens by 9.94%, ordinary CRPS by 8.62%, mean IIEE by 35.3% and extent
absolute error by 76.4%. Clipping mass reaches `0.243596` at zero and `0.102705`
at one, while maximum ensemble-mean displacement reaches `0.753021`. Thus whole
historical residual fields transfer useful rank diversity together with unsafe
bias, boundary atoms and spatial error; no analog parameter is retuned.

The subsequent frozen guidance-mixture contract has also been executed. It
draws two common-random-number members at each of five already evaluated
independent-CFG weight pairs and is operationally valid, but it fails the
proper-score, finite-ensemble reliability, boundary and spatial/physical
families. The mechanism is therefore closed without post-hoc changes to its
weights or member allocation. The clean-checkpoint deep ensemble frozen in
`NEXT_GENERATIVE_METHOD_CONTRACT.md` is retained only as an archived
reproducibility contract. It is not selectable as a fast fallback: measured
throughput implies months for the frozen three-seed training construction, and
no sampling or gate stage from that route is an active scientific dependency.

The frozen coherent-member-offset contract has also been executed and rejected.
Its purged training folds select amplitude `0.0`, leaving fair CRPS, ranks and
coverage exactly equal to raw. Boundary, mean-field, member-spatial and
operational families pass, but proper-score and finite-ensemble-reliability
families fail and `overall_eligible=false`. This closes global coherent offsets
without post-hoc changes. The checkpoint-trajectory EMA sampling and gate jobs
have completed for all 40 validation cases, but the publication worktree still
lacks a reconciled compact gate payload. Job completion alone is not scientific
evidence. The base gate id and separately completed retry id remain distinct
evidence units and must not be merged or substituted; the manuscript therefore
makes no positive or negative EMA claim.

The projection-free slack-limited ablation is also complete and rejected.
Although it creates no boundary or spatial damage, every pixel is blocked, the
selected and effective amplitudes are `0.0`, and all candidate metrics equal
raw on all 40 dates. Proper-score and finite-ensemble-reliability families fail
and `overall_eligible=false`. This closes projection as the sole explanation
for the coherent-offset failure; no slack, amplitude or fold retuning follows.

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
- The topology-preserving mechanism was frozen before execution in
  `NEXT_BASELINE_CONTRACT.md`, including exact projection semantics, leakage-safe
  selection, failure interpretations, server-side inputs, compact return schema
  and a one-parameter trusted runner interface. Its completed compact result is
  reconciled as a negative mechanism test without post-hoc changes.
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
- A local fail-closed publication audit now checks required paper files and
  parses all seven executable publication/reference scripts as Python,
  manuscript figure links, SVG parseability, contiguous numbered references and
  complete use of every listed reference. It requires consistency among
  publication status, the blocker declaration and the frozen-evaluation handoff,
  as well as unique contiguous claim-ledger IDs and the decision-bearing rows of
  all five manuscript evidence tables. Dropping a citation, negative mechanism
  result or active external boundary therefore cannot silently leave the
  narrative intact. It also checks semantic numeric anchors in both linked SVGs,
  rather than accepting arbitrary well-formed XML at the expected path. This is
  an integrity check and does not relax any scientific blocker.
- The five decision-bearing files were reconciled atomically after the external
  primary. The checker now requires their shared reconciled-negative state and
  decision anchors, so a partial narrative update fails locally instead of
  mixing pre-result and post-result claims.

## External boundary

## Reconciled external-primary result

The sealed raw stage `external_2024_calendar_global_bias_raw48_primary_v1` and
the exact permitted CPU recovery are complete. The four compact artifacts have
been reconciled atomically under
`paper/EXTERNAL_PRIMARY_RESULT_RECONCILIATION.md`. Their authoritative decision
is negative, so there is no active external evidence job and no admissible
resubmission, substitution, retuning or second evaluation.

Reconciliation is fail-closed. A positive primary claim requires verified signed
contract identity, input/raw/truth seals, every recorded hash, formation of the
complete candidate before scoring truth is opened, `overall_eligible=true`, and
explicit passes for proper-score, finite-ensemble reliability, boundary,
spatial/physical and operational families. Failure of any integrity check or
mandatory family is a negative primary result; no development result may
compensate for it. Claim C39 now records the reconciled negative decision from
the sole admissible compact payload; it cannot be reverted to a pending outcome
or strengthened into a positive generalization claim.

The preflight for this boundary is now explicit in
`paper/FROZEN_EVALUATION_HANDOFF.md`. It requires controller-attested immutable
checkpoint, dataset, code and environment identities; an implemented reviewed
runner; aligned deterministic-comparator cases; frozen seeds; and a single-pass
evaluation. It also fixes the summary-only return schema and forbids retuning
after confirmatory evidence. This closes an autonomous reproducibility gap but
does not constitute authorization or change `NOT_READY`.

The completed development joint gate decisions are negative: global spread
scaling, purged hurdle-isotonic/ECC-Q, exact mean-preserving projected spread,
fixed open-logit desaturation, frozen ZOIB-EMOS/ECC-Q, topology-preserving stratified transport
and purged analog-residual dressing are rejected as
the paper's calibrated-ensemble method. All currently implemented
mechanistically distinct development calibration routes have been completed and
rejected under the common gate; repeating or post-hoc retuning them would not
create confirmatory evidence. This does not invalidate the separately frozen
external primary above: that chain was independent of development-gate outcomes
and has now been reconciled exactly once as a negative result.
The paired date/block CPU analysis of the existing compact audit is complete;
its compact result has been reconciled into the evidence chain, while local
reconstruction remains intentionally excluded. The currently implemented modes
cannot supply another independent strong baseline family beyond the completed
negative comparators. No second external evaluation is authorized, and no
external primary remains active.

The guidance-mixture result is now complete and rejected. Its construction is
operationally valid, but all three proper-score criteria, two reliability
criteria, and the established-ice and exact-one boundary criteria fail. This is
a mechanism failure, not permission to retune weights or member allocation.

Both checkpoint-trajectory EMA stages have now completed, but completion
metadata do not contain the decision-bearing compact gate. Until one exact EMA
gate payload is reconciled in this worktree, the manuscript admits neither a
positive nor a negative EMA claim. The terminal-safe clean-checkpoint sampling
retry was cancelled rather than completed, so neither it nor its dependent CPU
gate is evidence in flight. It must not be described as queued or used to defer
the next autonomous mechanism test. Moreover, the measured clean-training
throughput makes a fresh three-seed training cycle a months-long route rather
than a fast fallback. Its frozen construction remains a reproducibility
contract, neither a scientific result nor a selectable next experiment or
active dependency.

The frozen latent-temperature result is complete and negative. Its reconciled
gate reports `overall_eligible=false`: fair CRPS worsens by `0.0046271592` with
date CI `[0.0014355657, 0.0075956683]`; reliability passes, but proper-score,
boundary and spatial/physical families fail. No second temperature is admissible.
The predeclared locked-MC-dropout chain is complete and negative under its
unchanged gate; it is no longer an active dependency or the next mechanism.
The closed sequence remains immutable: no second temperature may be
selected post hoc, and the dropout contract may not be altered.

The frozen pair remains specified in `NEXT_LATENT_TEMPERATURE_CONTRACT.md`.
The latent-temperature internal sampler completed all forty cases, and the
wrapper-only failure was recovered without GPU recomputation as
`latent_temperature_1p30_sampling_retry1`. Its dependent unchanged CPU gate is
complete and reconciled as the negative result above. The predeclared locked-
MC-dropout internal sampler completed the exact 40-by-10 outputs, and its outer
`cases_file=cases.json` schema failure was finalized on server CPU without GPU
resampling. The dependent compact gate then recorded the exact frozen candidate
`locked_mc_dropout_p010_final_ema_ensemble` with `overall_eligible=false`.
This closes the dropout mechanism and activates the already frozen iid calendar
global-bias fallback reported above; no second temperature, different dropout
probability or different mask construction is admissible post hoc.
