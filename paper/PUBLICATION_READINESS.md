# Publication readiness audit

The 2026-08-30 independent adapter-boundary audit found that the immutable
`ReviewedModeInventory` was tested in isolation but the executable
`construct_case` path still accepted a caller-supplied `admission=GO` dictionary
directly.  That bypass meant registration was not actually a dispatch
precondition.  The adapter now resolves the requested literal mode from the
registered immutable inventory before request validation or construction and
rejects both an empty inventory and a direct admission dictionary.  The focused
15-test adapter/review boundary and `git diff --check` pass.  This closes a local
integration bypass only: it neither records an independent trusted review nor
creates a controller-visible literal mode or scientific result.  Publication
status therefore remains `NOT_READY`.

The 2026-08-30 raw-member implementation audit added a separate deterministic
runner surface and focused tests for the frozen interface, purged folds,
training-only analog selection, complete-field copying and fail-closed input
validation.  The admission regression now checks this runner against the
independent construction oracle instead of treating the oracle as the runner.
An independent follow-up found that admission bound the synthetic-result digest
without validating its contents; the validator now requires exact equality to
the runner's deterministic dry-run record and explicitly rejects decision-bearing
or project-data metric output.  The focused 24-test boundary passed.  The
unified audit passed with 88 required files while retaining `NOT_READY`: no
independent review record, controller-visible literal mode or scientific result
is inferred from the local synthetic check.

The 2026-08-30 full executable publication audit independently ran test
discovery across the complete publication package before invoking the unified
artifact checker.  All 276 discovered tests completed successfully; eight
numerical fixtures were explicit dependency-only skips because NumPy is absent
from the minimal local runtime.  The subsequent unified checker passed with 86
required files, two figures, eight references and 40 claims at `NOT_READY`, and
`git diff --check` passed.  This broader snapshot supersedes the narrower
39-test snapshot below as the current executable audit, while leaving its
historical result intact.  It supplies no missing coverage--occurrence compact
record and changes no family decision, claim or blocker state.

The 2026-08-30 publication-readiness rerun independently searched the disposable
worktree for the two decision-bearing coverage--occurrence
`gate_decision.json` records and found neither payload.  It then executed the
unified publication checker against the unchanged evidence boundary; the audit
passed with 86 required files, two figures, eight references and 40 claims at
`NOT_READY`.  A focused 39-test boundary covering joint occurrence admission,
claim status, compact schemas, empirical traceability, figure generation,
limitations and references also passed.  These checks confirm that the current
manuscript and handoff fail closed rather than inferring family outcomes from
the two `summary_ready` scheduler states.  The fastest scientific transition
remains delivery and atomic admission of the two existing immutable compact
records; rerunning either completed calculation would add no evidence.

The 2026-08-29 coverage-admission reproducibility audit found that the
decision-bearing joint validator, its regression suite and its reconciliation
handoff existed but were not mandatory members of the unified publication
inventory; the suite was also absent from the exact documented regression
command.  All three files are now required and the suite is part of the
executable handoff.  This closes a deletion/drift gap at the current atomic
admission boundary only: it supplies neither missing compact payload, changes
no family decision and leaves publication status `NOT_READY`.  The unified
audit passed with 86 required files, two figures, eight references, 40 claims
and 30 regression suites after this inventory expansion.

The 2026-08-29 independent executable publication audit rechecked the current
manuscript, claim ledger, figures, tables, compact-evidence guards and
reproducibility boundary without relying on scheduler status.  The unified
checker passed with 83 required files, two figures, eight references and 40
claims at `NOT_READY`.  A broader test discovery executed 272 tests across the
publication package; all executable paths passed and eight numerical fixtures
were explicit dependency-only skips because NumPy is absent from the minimal
local runtime.  This new snapshot confirms internal consistency but supplies no
missing compact payload and no eligible scientific result.  The publication
status and blocker matrix therefore remain unchanged.

The 2026-08-29 decision-bearing evidence audit independently searched the
publication worktree for compact outputs from the two completed
coverage--occurrence jobs and found no result payload, metadata bundle or
reconciliation artifact for either job.  A scheduler status of `summary_ready`
therefore remains operational evidence only and cannot supply candidate
identity, five-family decisions, paired uncertainty or invariant checks.  The
same snapshot passed the unified checker with 83 required files, two figures,
eight references and 40 claims at `NOT_READY`; a focused 129-test boundary over
claim status, compact schemas, empirical traceability, figure generation,
limitations, references, minimum-tier comparisons and raw-member review also
passed.  This closes the current editorial-consistency audit but does not admit
either missing compact result, authorize an unimplemented mode or change any
scientific blocker.  The next scientific transition remains fail closed: admit
and reconcile those exact compact payloads if they become controller-visible;
if both are negative, consume only a controller-visible `GO` record for the
already frozen raw-member reweighting mechanism.

The 2026-08-29 independent publication-readiness trigger audit reran the
documented unified checker from the current worktree and obtained the exact
verdict `83 files, 2 figures, 8 references, 40 claims, status=NOT_READY`.
Manual cross-checking of `PAPER_DRAFT.md`, `CLAIM_LEDGER.md`, both publication
figures and `REPRODUCIBILITY.md` found no promotion of the reconciled negative
primary, no use of the superseded normalization result and no disagreement in
the four normative evidence rows.  The remaining gaps are scientific rather
than editorial: no jointly eligible calibration, no conformal comparison, no
probabilistic-DA comparison and no independent-strength deterministic
comparison.  The permitted trusted-mode inventory contains no executable mode
for the three missing comparison rows, and the raw-member reweighting route
still lacks a controller-visible reviewed literal mode.  This rerun therefore
adds current-snapshot audit evidence but does not authorize an experiment or
change publication status.

The 2026-08-29 raw-member admission-to-proposal audit found that the validator
checked the review record, runner, frozen contract and synthetic-result
identities but discarded those bindings from its successful CLI payload. That
made the downstream requirement for exact proposal bindings dependent on a
second, unbound file read. The admission payload now atomically emits the review
record SHA-256, publication commit and all three reviewed artifact SHA-256
identities with the literal mode and `admission=GO`; a focused regression fixture
requires exact equality. This closes a local substitution surface only. It does
not create a reviewed mode, authorize a proposal or change any scientific
evidence state; publication status remains `NOT_READY`.

The 2026-08-29 independent cross-surface rerun checked the current manuscript,
claim ledger, both publication figures, compact-evidence handoff and documented
reproducibility entry point against the normative blocker matrix.  The unified
checker passed with 83 required files, two figures, eight references and 40
claims, while the manual comparison found no promotion of a negative result and
no disagreement in evidence status.  The four decision-bearing gaps remain
unchanged: no jointly eligible calibration result, no conformal comparison, no
probabilistic-DA comparison and no independent-strength deterministic
comparison.  A raw-member reweighting proposal also remains fail closed until
one controller-visible review record binds a literal implemented mode and
passes the documented admission validator without deviations.  This audit
therefore strengthens the readiness provenance but supplies no new scientific
result; publication status remains `NOT_READY`.

The 2026-08-29 current-snapshot provenance audit found that the unified checker
already required and passed 83 publication files and 29 regression suites, but
the undated `Current independent audit snapshot` and its trigger-specific rerun
still described the superseded 79-file inventory as current. Those two current
descriptions and the immediately preceding inventory-expansion summary now
report 83 files; dated 79-file entries remain unchanged as audit history. No
scientific result, blocker state or readiness decision changed. Publication
status remains `NOT_READY`.

The 2026-08-29 reproducibility-inventory audit found that the newly completed
fail-closed raw-member reweighting review validator and its regression suite
were described by the handoff checklist but were not mandatory inputs to the
unified publication audit. Both executable files and the checklist are now
required publication artifacts, and the review suite is part of the exact
documented regression command. The unified audit passed with 83 required files,
two figures, eight references, 40 claims and 29 suites. This closes a
deletion/drift gap in the autonomous handoff
only; it does not create a reviewed trusted mode, authorize an experiment or
change any scientific evidence state. Publication status remains `NOT_READY`.

The 2026-08-29 independent publication-readiness rerun exposed two stale
fail-closed bindings in the current committed state.  Claim `C40` was present
in the ledger and manuscript result subsection but absent from the manuscript's
claim and empirical-evidence traceability tables, causing eleven downstream
negative-path tests to stop at the wrong invariant.  The frozen rank-coherent
contract also had a newer actual SHA-256 than the immutable manifest, handoff
and checker recorded, while its execution and reconciliation files had already
advanced to completed-but-unadmitted states that the unified checker and
renderer did not recognize.  The traceability tables, exact digest bindings and
unreconciled lifecycle-state guards are now synchronized.  No result, metric,
threshold or gate decision changed, and completion still cannot be interpreted
as scientific admission without the exact combined admission inputs.  The
focused 99-test defect boundary and subsequent 14-test renderer/claim/manifest
boundary passed.  The unified checker now passes with 79 required files, two
figures, eight references and 40 claims at `NOT_READY`; `git diff --check`
passes.  The four scientific blocker rows below remain unchanged.

The 2026-08-29 joint spread--occurrence result review confirms 40/40 completed
cases and `overall_eligible=false`. The candidate aggregate improves
date-balanced rank total variation from `0.3566938377` to `0.1092186644` and
fair CRPS from `0.0584905850` to `0.0575343944`, but absolute rank total
variation and maximum-bin adequacy fail; proper-score, boundary and
spatial/physical families also fail. The compact payload is not admitted as a
quantitative publication evidence unit because its `gate.candidate_method`
names a different method than its sole candidate aggregate. This fail-closed
provenance decision leaves publication status `NOT_READY` and activates the
already frozen raw-member probability-reweighting contract without authorizing
an unimplemented mode or any scale/threshold retuning.

The 2026-08-29 independent-deterministic closure-binding audit found that all
four normative publication surfaces already required one compact-record
SHA-256 identity, while `FROZEN_EVALUATION_HANDOFF.md` required only aligned
comparator evidence and unspecified immutable identities. The handoff now
requires the same exact compact-record identity in the manuscript, claim
ledger, readiness audit and reproducibility handoff, and states that neither a
promotion label nor a non-identical provenance-identity collection closes the
row. No scientific result or evidence state changed:
`independent_deterministic` remains `PRESENT_DEVELOPMENT_ONLY` with compact
identity `NONE`.

The 2026-08-29 probabilistic-DA closure-binding audit found that the frozen
contract required a valid trusted execution and one decision-bearing outcome,
but did not itself require the compact-record SHA-256 identity already mandated
by the manuscript, claim ledger, this readiness audit and reproducibility
handoff. `NEXT_PROBABILISTIC_DA_COMPARISON_CONTRACT.md` now requires that exact
four-surface identity and states that an outcome label alone closes nothing.
No method, threshold, result or evidence status changed: `probabilistic_da`
remains `MISSING` with compact identity `NONE`.

The final 2026-08-29 family-matrix provenance pass checked locked MC dropout
and the iid calendar global-bias mixture.  For locked MC dropout,
`PAPER_DRAFT.md`, `CLAIM_LEDGER.md` C38, `REPRODUCIBILITY.md` and
`LOCKED_MC_DROPOUT_RESULT_RECONCILIATION.md` consistently report only the
trusted candidate identity and `overall_eligible=false`; none invents an
unavailable family decision or effect size.  For the iid mixture, the
manuscript, C37 and the reproducibility handoff agree that only the proper-score
family fails, on fair CRPS `0.0584905850` to `0.0582878868`, paired date
interval `[-0.00100456, 0.000700186]`, and ordinary CRPS `0.0621082810` to
`0.0623775052`, while reliability, boundary, spatial/physical and operational
families pass.  No provenance correction was warranted.  This audit adds no
scientific evidence, does not promote the unavailable quantitative dropout
payload and leaves publication status `NOT_READY`.

The 2026-08-29 key-claim language audit independently compared the Abstract,
Contributions, Limitations and Conclusion with all four normative blocker rows
and the negative compact evidence.  It found that the most visible sections
named the two wholly missing comparison families but left the deterministic
background/3D-Var comparison's development-only evidence strength implicit.
The Abstract, Contributions and Conclusion now state that boundary explicitly;
no result, metric or blocker state changed.  The focused minimum-tier,
limitation-traceability and claim-status suite passed 96 tests.  The subsequent
unified checker passed with 79 required files, two figures, eight references and
39 claims at status `NOT_READY`, and `git diff --check` passed.

The 2026-08-29 remaining-blocker language audit found one concrete asymmetry:
Abstract and Conclusion named all three incomplete minimum-tier comparison
rows, while Limitations named only the independent 3D-Var gap. Limitations now
also states that conformal intervals and a probabilistic DA baseline such as
EnKF/LETKF are missing, that the minimum strong domain/SciML baseline tier is
incomplete, and that comparison completeness cannot compensate for a failed
calibration-gate family. This is an editorial traceability correction only; all
four normative evidence states and publication status remain unchanged.

The 2026-08-29 conformal closure-binding audit found a narrower fail-open
asymmetry in the frozen contract itself. The manuscript, claim ledger,
readiness matrix and reproducibility handoff already required one identical
compact-record SHA-256 identity, but the contract's closure paragraph named
only valid trusted execution and an outcome label. The paragraph now requires
the same exact identity on all four publication surfaces and explicitly states
that a label alone closes nothing. No method, threshold, result or normative
evidence state changed; the conformal row remains `MISSING`.

The 2026-08-29 end-to-end reproducibility audit executed the complete two-command
handoff documented in `REPRODUCIBILITY.md`, rather than relying on static command
inspection.  The documented 27-suite list exactly matches
`REQUIRED_REGRESSION_SUITES` in order and membership, and the subsequent unified
checker completed with 79 required files, two figures, eight references and 39
claims at status `NOT_READY`.  No missing or stale executable step was found.
This establishes local handoff executability only: it supplies no new scientific
result, closes none of the four normative blocker rows and does not change the
publication status.

The 2026-08-29 cross-surface evidence audit independently traced all four
normative blocker states through `PAPER_DRAFT.md`, `CLAIM_LEDGER.md`,
`MINIMUM_TIER_COMPARISON_AUDIT.md`, both linked figures and the compact-result
values cited for the reconciled primary decision.  The scientific states and
reported fair-CRPS and normalized-rank values agree.  The audit did find one
publication-metadata mismatch: `CLAIM_LEDGER.md` still reported its audit date
as 2026-08-23 despite containing the later C39 reconciliation and subsequent
cross-artifact checks.  Its `Last audited` field is now 2026-08-29.  This fixes
provenance freshness only; it creates no scientific evidence, closes none of
the four blocker rows and leaves publication status `NOT_READY`.

The dependency-limited score-aware boundary was rerun independently on
2026-08-27 with the four documented reference, admission, compact-directory and
reconciliation modules.  It ran 51 tests: 43 passed and eight were explicit
NumPy-only skips.  Five skips are numerical reference-oracle fixtures; the
other three are admission-divergence fixtures that exercise the same numerical
training-row parity.  This distinction supersedes the ambiguous shorthand
"five NumPy-dependent checks": five is the unresolved oracle count, while eight
is the skip count for this complete four-module boundary.  The exact same
worktree then passed the mandatory unified audit with 79 required files, two
figures, eight references and 39 claims.  NumPy remains unavailable in the
installed local runtime, so none of the eight skipped paths is represented as
executed evidence.  All decision-bearing admission remains fail closed, and
the scientific status stays `NOT_READY`.

The independent predictor-construction audit now includes numerical oracle
fixtures for every one of the thirteen ordered score-aware descriptors on a
non-square, spatially weighted grid.  The oracle uses explicit scalar loops for
weighted means and both directional semivariograms rather than the reference
helpers.  Separate fixtures independently reconstruct the spatially weighted
MAE-plus-0.25-MSE target and prove that changing held-out truth leaves all
forecast-only predictors bitwise unchanged while changing the truth-derived
targets.  In the minimal local runtime these three numerical fixtures are
dependency-only skips because `numpy` is unavailable; the remaining focused
score-aware suite passed 30 tests with eight total dependency-only skips.  The
mandatory unified audit, which includes this module, passed with 79 required
files, two figures, eight references and 39 claims.  The fixtures close a local
test-design gap but do not create trusted scientific evidence, so publication
status remains `NOT_READY`.

An independent 109-test publication-surface rerun covered claim-status,
figure-generation, limitation, reference, minimum-tier, empirical-traceability,
immutable-identity, score-aware reconciliation and score-aware reference
contracts. All executable tests passed; five NumPy-dependent score-aware tests
were skipped because NumPy is unavailable in each installed local Python
runtime. The manuscript, claim ledger, both linked figures and their tabulated
negative decisions therefore agree with the current evidence matrix. This is
local consistency evidence only and does not change any blocker state.

The score-aware runner admission boundary now independently exercises both the
positional cross-fitting semantics and exact training-row construction that the
compact validator expects. The reference exposes the frozen forty-case,
five-by-eight contiguous folds with non-circular purge three and builds each
retained `(case_id, member_index)` row from the exact thirteen forecast-only
predictors and the fixed weighted MAE-plus-MSE target. Admission compares row
identities, predictor values and targets for three separated folds. Negative
runner fixtures prove that sorting identifiers, retaining a purged or held-out
case, permuting member predictor columns or changing the target coefficient
fails closed. The focused score-aware suite passed 48 tests with five
dependency-only skips, and the unified audit passed with 79 required files, two
figures, eight references and 39 claims. This is runner-parity evidence only:
it neither creates a trusted mode nor supplies an eligible scientific result,
so the readiness status remains unchanged.

The probabilistic-DA downstream boundary now has an executable consumer:
`reconcile_probabilistic_da_admission.py` accepts the combined admission JSON
only alongside the controller-retained `compact_directory_sha256`. Regression
fixtures prove that a decision-bearing outcome without that exact digest, or
with a different digest, is rejected. This closes the local admission-to-
reconciliation identity gap without changing the scientific evidence row or
the not-ready status below.

Audit date: 2026-08-30

Publication status: NOT_READY

Required scientific blockers: an eligible spatially preserving calibration and
the remaining minimum-tier comparisons

External primary evidence state: RECONCILED_NEGATIVE

Controller readiness: NOT_READY
Scientific primary reconciliation: COMPLETE_NEGATIVE
Autonomous publication work remaining: YES

The latest reproducibility audit added the probabilistic-DA directory adapter,
combined admission CLI and their negative fixtures to the mandatory publication
inventory. The adapter
requires exactly three compact files, verifies SHA-256 bindings for the summary
and 120-row three-method case table, runs all three semantic oracle stages, and
rejects a claimed decision label that differs from the recomputed outcome.
The combined CLI additionally repeats a framed digest over every filename and
payload after semantic validation and rejects in-admission substitution of any
payload or directory member. This closes a local admission-parity
gap only: no trusted mode has been admitted and the normative evidence row and
publication status remain unchanged.
The unified audit passed with 83 required files after the subsequent inventory
expansion;
earlier lower counts below are retained only as dated audit history.

## Current independent audit snapshot

The current audit passed with 86 required files, two parseable linked figures,
eight traceable references and 40 contiguous claim-ledger records. The
immutable rank-coherent package separately
passed its 13-file manifest check and all 26 focused prototype, adapter-parity,
combined-admission and manifest fixtures. These current counts supersede the
smaller historical inventory counts recorded later in this chronology; those
older counts remain as audit history rather than descriptions of the current
inventory.

This rerun confirms local integrity only. The rank-coherent contract still has
no literal reviewed trusted mode, and the conformal and probabilistic-DA
contracts remain frozen but non-executable. No decision-bearing scientific row
therefore changes, and publication status remains `NOT_READY`.

The publication-readiness trigger was then independently repeated against this
exact worktree state. The complete 30-suite command documented in
`REPRODUCIBILITY.md` ran 276 tests: 268 passed and eight NumPy-dependent tests
were explicit dependency-only skips. The unified checker then passed with 86
required files, two parseable linked figures, eight traceable references and 40
contiguous claim-ledger records; `git diff --check` reported no whitespace
errors. This records the trigger-specific verification rather than relying on
the earlier controller summary. It remains local integrity evidence only: none
of the four normative blocker states changes, no experiment is authorized, and
the status remains `NOT_READY`.

The same audit found that the newly executable conformal oracle, CPU runner
prototype and synthetic suite were not yet members of the mandatory publication
inventory or the documented regression command. All three source artifacts and
the suite are now required by the unified checker. This makes deletion or
synthetic-parity drift fail closed, but it is runner-parity evidence only: no
literal trusted mode has been admitted, no conformal result exists and the
corresponding minimum-tier row remains `MISSING`.

The conformal handoff now also has an explicit fail-closed admission-record
boundary. The validator binds an independently supplied literal mode and trusted
synthetic result to the exact frozen contract digest and rejects extra fields,
placeholders, non-PASS validation or any deviation. Focused fixtures cover the
valid record, contract substitution, non-object input and a non-empty deviation.
This removes ambiguity from the next trusted review but does not manufacture a
reviewed mode, authorize a proposal or change the `MISSING` evidence state.

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

The 2026-08-27 closure-condition audit found that the checker required all four
conditions to be non-empty but did not bind their exact scientific meaning. A
plausible edit could therefore replace the eligible-calibration requirement to
pass every mandatory no-compensation family with improvement in only one
family. The checker now compares every condition with its exact normative value.
Four independent negative fixtures prove that non-empty substitutions fail
closed: single-family improvement cannot replace joint eligibility, contract
availability cannot replace a decision-bearing trusted conformal or
probabilistic-DA result, and development-only documentation cannot replace
independent-strength common-information evidence. This strengthens the local
mapping from every remaining blocker to its verifiable closure artifact; it adds
no scientific result and leaves status `NOT_READY`.

The subsequent 2026-08-27 cross-artifact audit binds those same closure
conditions to the literal stop/go semantics in `RESEARCH_PLAN.md`,
`NEXT_CONFORMAL_BASELINE_CONTRACT.md`,
`NEXT_PROBABILISTIC_DA_COMPARISON_CONTRACT.md` and
`FROZEN_EVALUATION_HANDOFF.md`. The checker now fails closed if any required
binding is absent or weakened. One parameterized negative test mutates each of
the four artifacts independently, and a coverage test fixes the complete file
set. The targeted suite passed 51 tests; the full documented regression
contract passed 166 tests with two expected skips because `numpy` is absent in
the minimal local environment. The unified audit again passed with 61 files,
two figures, eight references and 39 claims. This is stronger consistency
evidence, not a blocker closure or a new scientific result; status remains
`NOT_READY`.

The following 2026-08-27 decision-surface audit closes the remaining local
fail-open gap. The checker previously verified only generic section and claim
anchors in `PAPER_DRAFT.md` and `CLAIM_LEDGER.md`; those anchors could survive
while one route-specific transition was weakened. It now requires all four
exact transition rules on both publication surfaces: the conformal and
probabilistic-DA rows require valid compact evidence for one frozen matrix
outcome, independent deterministic evidence requires the frozen
common-information comparison, and eligible calibration requires a pass in
every mandatory no-compensation family. Eight independent negative fixtures
weaken one transition on one surface at a time and fail closed. This is
cross-artifact consistency evidence only: no new scientific result exists, no
blocker closes and publication status remains `NOT_READY`.

The 2026-08-28 admission-to-transition audit distinguishes the three remaining
minimum-tier handoffs instead of treating admission as one generic event. A
passing conformal admission record binds only the reviewed implementation and
contains neither a scientific outcome nor a compact-result digest, so it cannot
close that row. The probabilistic-DA row requires combined semantic admission
and downstream reconciliation against the controller-retained compact-directory
digest. The independent deterministic row has no local admission fixture and
requires aligned compact comparator evidence with the frozen immutable
identities. The unified checker now binds these exact route-specific boundaries,
and three negative fixtures weaken them independently and fail closed. This
closes a reproducibility handoff ambiguity only; all scientific blocker states
remain unchanged and publication status remains `NOT_READY`.

## Independent audit snapshot

The publication-readiness trigger was independently rerun against the current
worktree on 2026-08-27. The fail-closed command
`python3 paper/check_publication_artifacts.py` audit passed with 61 required files,
two parseable linked figures, eight traceable references and 39
contiguous claim-ledger records. A separate unfinished-text scan found no
`TODO`, `TBD`, `FIXME` or `PLACEHOLDER` marker in the Markdown publication
artifacts. The worktree was clean before this audit note was added.

This is a consistency and reproducibility result, not a scientific result. It
does not change any evidence state in the blocker matrix and cannot support
`READY_FOR_HUMAN_REVIEW`. The next decision-bearing development action remains
fail-closed admission of the frozen rank-coherent mechanism: only a
controller-visible record with a literal reviewed mode and a trusted synthetic
directory accepted with `decision_bearing=True` may authorize one full CPU
proposal. Until then, no implemented negative baseline is repeated and no mode
identifier is invented.

The independent status-derivation audit found that the unified checker formerly
validated the normative blocker matrix and the top-level readiness declaration
separately. A two-line edit could therefore claim `READY_FOR_HUMAN_REVIEW` and
`Required scientific blockers: none` while retaining `MISSING`,
`MISSING_ELIGIBLE_RESULT` or `PRESENT_DEVELOPMENT_ONLY` states in the matrix.
The checker now rejects that contradiction explicitly, and a negative fixture
proves that the false promotion fails closed. This strengthens the completion
guard only; all four normative blockers remain open and status stays
`NOT_READY`.

The same independent trigger reran the decision-bearing handoff subset rather
than relying only on the unified checker's aggregate success. All 29 focused
tests covering the rank-coherent prototype, adapter parity, admission record,
immutable manifest and contingent raw-member reweighting oracle passed. The
negative paths include envelope and threshold drift, non-finite source data,
compact-output disagreement, an unbound contract digest, manifest mutation and
path traversal. This establishes a reproducible fail-closed local handoff at
the frozen contract boundary; it does not assert that a trusted server mode has
been admitted, does not authorize an experiment proposal and does not change
`NOT_READY`.

The independent rerun also found that the newly frozen raw-member reweighting
oracle and its fail-closed tests were described by the contract but were not
members of the required publication inventory or documented regression suite.
Both files and the test module are now mandatory in the unified checker and in
the reproducibility command. This prevents the construction oracle from being
silently removed or drifting untested while the prose contract remains. The
change strengthens a contingent trusted-runner handoff only; it supplies no
decision-bearing result and leaves publication status `NOT_READY`.

A subsequent executable parity audit found that this oracle stopped at selected
rank positions and therefore did not encode the contract's final mapping through
the held-out raw-member case-mean ordering. That distinction is decision-relevant
when member order differs from mean order or case means tie. The oracle now
performs the exact mean-then-member-index ordering, reports rank positions
separately from source raw-member indices, rejects malformed or non-finite means,
and tests a permuted ordering containing a tie. This closes a future trusted-
runner parity gap only; it does not activate the contingent method or change
`NOT_READY`.

The score-aware contingent-method audit found that its frozen prose contract had
no executable reference oracle.  The new oracle fixes training-only
standardization, the unpenalized-intercept ridge SVD formula, stable risk
weights, deterministic risk/index ties, systematic inverse-CDF selection and
all required multiplicity diagnostics.  Focused negative fixtures reject wrong
member counts, non-finite risks and degenerate training predictors; extreme
risk ranges remain finite.  The suite and both source artifacts are now required
by the unified publication checker.  This is an admission prerequisite only:
there is still no literal trusted mode, no scientific result and no change to
`NOT_READY`.

The subsequent compact-output parity audit closes the remaining directory-level
handoff gap for that contingent mechanism.  The new fail-closed validator
requires exactly four compact files, recomputes each case's risk weights,
systematic selections, multiplicities, unique-member count and ESS, reconciles
copy/mask and paired-score aggregates, and binds `overall_eligible` to the
conjunction of all five mandatory gate families.  Negative fixtures cover
weight, ESS, aggregate, uncertainty, invariant, gate and extra-file drift.  This
is still only an admission prerequisite: no literal trusted mode or scientific
result exists, and publication status remains `NOT_READY`.

The integrated v2 admission audit closes the subsequent time-of-check gap.  A
single length-framed SHA-256 now binds the exact names and bytes of all four
compact files alongside the runner, frozen contract and independent reference.
The combined validator performs semantic parity, full compact cross-file parity
and a final digest recheck in one process.  Integration fixtures reject both a
pre-admission semantically valid reserialization and substitution immediately
after semantic admission.  This strengthens evidence admission only: a literal
trusted mode and an eligible scientific result are still absent, so status
remains `NOT_READY`.

The publication handoff now also has a fail-closed score-aware result
reconciliation template. It reserves the exact admission identities, separates
positive and negative branches, and requires one atomic update across the
manuscript, claim ledger, readiness audit, reproducibility record and the
reconciliation file itself. The template is a mandatory publication artifact;
its current `PRE_RESULT_NO_TRUSTED_MODE` state cannot be mistaken for evidence
or authorization. This closes an autonomous result-transfer gap only: no
literal trusted mode or scientific result exists, and all normative blockers
remain open.

The subsequent reconciliation audit strengthened that atomic marker from a
decision label into an evidence-bound record. Every one of the five publication
files must now repeat the same exact `compact_directory_sha256` and the literal
Booleans for `proper_score`, `reliability`, `boundary`, `spatial_physical` and
`operational`; `overall_eligible` must equal their conjunction and the selected
branch must agree. The marker now also binds the exact
`admission_record_sha256`, `completed_cases=40` and `ensemble_size=10`; both a
single-file substitution and mutually consistent incomplete counts fail closed.
Thirteen focused fixtures pass, including substitutions of the compact digest,
admission identity, completed counts and a single family decision. The full
audit still passes with 61 required files, two figures, eight references and 39
claims, and still derives `NOT_READY` from the unchanged blocker matrix.

The next end-to-end provenance audit removes the remaining manual identity
transfer. Combined admission now emits the SHA-256 of the exact admission JSON
and compact directory that survived the same fail-closed operation, and it
rejects replacement of either input before returning. An integration fixture
uses those emitted identities to construct the canonical reconciliation marker;
a separate negative fixture substitutes the admission JSON during admission.
The focused admission/reconciliation suite passes 26 tests, the unified audit
again passes with 61 required files, two figures, eight references and 39
claims, and `git diff --check` reports no whitespace errors. This closes a
reproducibility handoff gap only; it does not supply an eligible calibration or
either missing minimum-tier comparison, so publication status remains
`NOT_READY`.

The rank-coherent path now closes the analogous time-of-check gap before a
trusted mode can be proposed. Its admission CLI accepts both the exact
controller-visible JSON and the decision-bearing four-file compact directory,
applies the frozen record validator and directory oracle in one process, emits
their SHA-256 identities with the literal `reviewed_mode`, and rereads both
inputs before returning. Independent negative fixtures replace the compact
directory bytes or admission JSON after the initial check and both fail closed.
This is implementation-integrity evidence only: there is still no admitted
literal mode or eligible result, so every scientific blocker and publication
status remain unchanged.

The downstream rank-coherent result transition is now executable and
fail-closed. `canonical_marker` invokes combined admission over the exact JSON
and compact directory, then binds its returned identities to all five literal
gate-family decisions and their conjunction. `publish_reconciliation` requires
the identical canonical marker in complete replacements for the manuscript,
claim ledger, reproducibility record, readiness audit and reconciliation record
before one prepare-first, rollback-protected publication; the publisher derives
the marker again rather than trusting caller text. Negative fixtures
prove that a partial surface transition performs no writes and that an injected
failure after replacement begins restores all five prior surfaces. This adds no
trusted mode or scientific result; status remains `NOT_READY`.
The expanded unified audit passed with 64 required files after the executable
reconciliation fixtures were added.

The compact-to-publication transition is now renderer-backed rather than
caller-authored. After rerunning combined admission, the renderer extracts the
compact-bound experiment and candidate identifiers, all aggregate effect sizes,
both paired uncertainty intervals and the five literal family decisions. It
creates complete replacements for all five reconciliation surfaces and, for a
negative branch, names every failed family while explicitly preserving the
blocker. Four focused fixtures cover successful rendering, a missing proper-score
effect size, caller text that understates a failed family and an attempted
negative-branch blocker closure. This closes a local result-transfer gap only;
there is no new decision-bearing result and publication status remains
`NOT_READY`.
The unified audit passed with 66 required files, two figures, eight references
and 39 claims after the renderer and its fixtures became mandatory.

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

## Minimum-tier decision-bearing evidence guard

These three closure routes remain fail-closed until an identical compact record
identity is admitted on every normative publication surface. `NONE` explicitly
forbids a decision-bearing result claim or independent-strength promotion.

| Route | Evidence status | Compact evidence record SHA-256 | Allowed presentation |
|---|---|---|---|
| `conformal` | `MISSING` | `NONE` | `PRE_RESULT_ONLY` |
| `probabilistic_da` | `MISSING` | `NONE` | `PRE_RESULT_ONLY` |
| `independent_deterministic` | `PRESENT_DEVELOPMENT_ONLY` | `NONE` | `DEVELOPMENT_ONLY` |

| `eligible_calibration` | `MISSING_ELIGIBLE_RESULT` | `NONE` | `BLOCKED` |

The positive eligible-calibration row is separately bound to one compact record
across manuscript, claim ledger, readiness and reproducibility. The current
missing state forbids a decision-bearing eligibility claim.

The `conformal` blocker may close only when one frozen matrix outcome and its exact compact record SHA-256 identity match the manuscript and claim ledger; an outcome label alone closes nothing.
The `probabilistic_da` blocker may close only when one frozen matrix outcome and its exact compact record SHA-256 identity match the manuscript and claim ledger; an outcome label alone closes nothing.
The `independent_deterministic` blocker may close at independent strength only when the frozen common-information evidence and its exact compact record SHA-256 identity match the manuscript and claim ledger; a promotion label alone closes nothing.
The `eligible_calibration` blocker may close only when every mandatory no-compensation family passes and the exact compact record SHA-256 identity matches the manuscript and claim ledger; an eligibility label alone closes nothing.

The 2026-08-27 closure-route audit now makes those three rows executable
publication guards rather than prose-only destinations. The checker requires
the same status, compact-record identity and allowed presentation in the
manuscript, claim ledger, readiness audit and normative minimum-tier audit. It
also confines the four conformal/probabilistic-DA outcome labels to the explicit
pre-result matrix. Three negative fixtures reject an unsupported conformal
decision, an unsupported probabilistic-DA decision and an independent-strength
promotion of the development-only deterministic comparison. The focused suite
passes 41 tests; the unified audit still passes with 61 required files, two
figures, eight references and 39 claims, and status remains `NOT_READY`.

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

The 2026-08-27 reference-traceability audit independently checked all eight
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

The independent publication-readiness rerun found that the two frozen missing
baseline contracts still lacked method-specific primary citations. The
manuscript and contracts now cite the split-conformal construction of Lei et
al. (2018) and the LETKF construction of Hunt et al. (2007), while
`REFERENCE_TRACEABILITY.md` explicitly prevents either citation from being
treated as project-specific evidence. The checker now requires all eight exact
identities and support mappings, and a negative fixture removes the new LETKF
row. This closes a concrete literature-traceability gap without supplying either
missing baseline result or an eligible calibration; status remains `NOT_READY`.

The 2026-08-27 figure-generator audit found that finite compact inputs could
still reach undefined rendering paths: jointly zero calibration values produced
a zero plotting scale, and a zero raw joint-gate metric produced an undefined
ratio. Both generators now reject those inputs explicitly, reject negative
metrics consistently, and have deterministic compact-fixture regression tests.
The unified checker requires and executes that suite. This closes a concrete
figure-reproducibility gap but adds no scientific result, so publication status
remains `NOT_READY`.

The 2026-08-27 executable-handoff audit adds a strict sidecar validator before
every `SERVER_ONLY` compact consumer. It binds one exact producer experiment,
one exact artifact basename and the artifact bytes through lowercase SHA-256;
missing hash entries, digest mismatches, extra entries and producer substitution
fail closed. Four regression fixtures exercise the accepted contract and the
three principal provenance failures. This closes the executable sidecar gap but
does not create a missing scientific result, so publication status remains
`NOT_READY`.

The 2026-08-27 end-to-end CLI audit removes the remaining bypass around that
validator: each of the three `SERVER_ONLY` consumers now requires and validates
its producer-bound manifest inside the consumer process before payload parsing,
directory creation or output writes. Six negative paths cover a mismatched
manifest and a schema-invalid payload for each consumer and require that no
partial SVG or JSON remains. This closes an executable atomicity/provenance gap
without adding scientific evidence, so publication status remains `NOT_READY`.

The 2026-08-27 output-publication audit replaces direct final-path writes in all
three consumers with same-directory temporary files and `os.replace`. The
two-output case-level path prepares and `fsync`-s both artifacts before the
first replacement. A fault-injection fixture makes preparation of the second
artifact fail and verifies that both prior destinations remain byte-identical
and no temporary files survive. This is an explicit prepare-stage guarantee,
not a claim of a filesystem-wide transaction across multiple replacements, and
it does not change the scientific `NOT_READY` status.

The 2026-08-27 replacement-phase audit strengthens that helper with
same-directory backups and reverse-order rollback. Fault injection at the
second artifact's final replacement now proves that both prior targets are
restored byte-for-byte; a separate mixed existing/new-target fixture proves
that rollback also removes a target absent on entry. Both paths require that no
temporary or backup file survives. The guarantee is explicitly limited to
caught filesystem exceptions: no durable journal or recovery claim is made for
abrupt process or operating-system termination. This closes the known
replacement-phase consistency gap without changing any scientific result, so
publication status remains `NOT_READY`.

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

The 2026-08-29 family-source traceability audit found that the guidance-mixture
row in the manuscript matrix and C32 correctly recorded a spatial/physical
family failure, while the compact-source handoff in `REPRODUCIBILITY.md`
enumerated only the proper-score, reliability and boundary failures. The
handoff now also records that every spatial/physical criterion failed, matching
the already reconciled readiness evidence without adding an effect size or
changing `overall_eligible=false`. This closes a family-specific provenance gap
but does not change `NOT_READY` or remove any scientific blocker.

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

The 2026-08-27 independent publication-readiness rerun completed the unified
fail-closed audit and the full documented regression contract against the
current worktree. The audit passed with 57 required files, two figures, six
references and all 39 claim-ledger rows traced; all 92 regression tests passed.
The compact-payload subset now explicitly rejects a wrong 160-row envelope,
duplicate method/date identities, candidate substitution, duplicate full-region
rows and a non-Boolean `overall_eligible`. `git diff --check` found no whitespace
errors. This confirms internal consistency and executable handoff integrity, but
it does not create decision-bearing scientific evidence. The status therefore
remains `NOT_READY`: the rank-coherent construction still lacks a separately
reviewed literal trusted mode and the minimum-tier scientific blockers below
remain unchanged.

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

The 2026-08-27 admission-package audit closed a remaining mixed-version review
path. `rank_coherent_admission_manifest.json` now binds the frozen contract,
runner prototype, reference/schema oracle, adapter specification and parity
oracle, review checklist, admission CLI and all three focused suites by exact
SHA-256. `validate_rank_coherent_manifest.py` rejects missing or changed files,
schema drift and path traversal, and the unified checker requires and executes
its regression suite. This strengthens implementation admission only; no
trusted mode or decision-bearing scientific result exists, so status remains
`NOT_READY`.

The subsequent independent packet audit found that a valid manifest could still
enumerate an arbitrary non-empty subset and did not bind the controller handoff
or the manifest validator and its tests. The validator now requires the
exact 13-file allowlist, including the handoff, validator and negative fixtures; any
missing or extra manifest entry fails before digest checks can be treated as a
complete-package result. The handoff records the same count and fail-closed
semantics, and the unified checker protects those statements. This closes a
real mixed-package admission path but does not create a trusted mode or a
scientific result; publication status remains `NOT_READY`.

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

The minimum-tier future-state reconciliation is now executable rather than
prose-only. The unified validator admits both frozen positive and negative
outcomes for `conformal` and `probabilistic_da`, plus positive and negative
independent-strength deterministic outcomes, only when `PAPER_DRAFT.md`,
`CLAIM_LEDGER.md` and this readiness record carry one identical 64-hex compact
record identity. The comparison audit remains a frozen pre-result inventory;
it cannot be rewritten into result evidence. Focused fixtures exercise all six
valid transitions and reject a partial surface update, a mismatched digest and
a label-only transition. This closes the previously identified executable
future-state gap but supplies no new scientific evidence: the current guard
rows and blocker matrix remain unchanged, so publication status remains
`NOT_READY`.

The subsequent blocker-transition audit makes those future states operationally
complete. For each of the six admitted outcomes, the checker now requires the
matching readiness blocker row to carry that exact outcome state and derives
`Publication status` from the entire blocker matrix. Focused negative fixtures
reject an unchanged blocker row, a blocker closed with the wrong outcome and a
premature ready declaration while any normative blocker remains unresolved.

The joint-transition audit now exercises the only admissible terminal state as
one transaction across all four normative blockers. It accepts
`READY_FOR_HUMAN_REVIEW` only when every compact-evidence surface agrees, every
readiness row carries its exact closure outcome, `Required scientific blockers`
is literally `none`, and the frozen evaluation handoff no longer reports an
active result. Focused negative fixtures independently reopen one row, retain a
stale blocker declaration, and retain an active handoff. Thus no collection of
partial closures can be mistaken for publication readiness, while the present
scientific rows and `NOT_READY` status remain unchanged.
Thus compact identity alone cannot create an editorially inconsistent closure;
the current evidence rows are unchanged and status remains `NOT_READY`.

The following independent publication audit found that the three minimum-tier
compact identities were atomically guarded in the manuscript, claim ledger and
readiness record but were absent from the reproducibility handoff. The exact
three-row guard is now repeated in `REPRODUCIBILITY.md`, and the unified checker
requires it to match the manuscript atomically. Focused fixtures accept the
current missing/development-only state and reject an omitted identity row. This
closes a reproducibility synchronization gap only: no compact result is created,
all four scientific blockers remain open and status remains `NOT_READY`.

The 2026-08-29 continuation audit checked the first two remaining family-level
matrix rows against their declared compact publication source. For topology-
preserving stratified transport, `PAPER_DRAFT.md`, `CLAIM_LEDGER.md` C30 and
`REPRODUCIBILITY.md` agree that proper-score and finite-ensemble-reliability
families fail while boundary, spatial/physical and operational families pass.
For purged analog-residual dressing, the same three surfaces agree that only
finite-ensemble reliability and operational validity pass. Their quoted fair-
CRPS anchors also reconcile exactly. No provenance correction was warranted;
this records a completed negative audit, does not add evidence or alter either
gate decision, and leaves publication status `NOT_READY`.

The subsequent 2026-08-29 provenance audit checked both coherent-offset rows.
For the projected coherent member offset, `PAPER_DRAFT.md`, `CLAIM_LEDGER.md`
C33 and `REPRODUCIBILITY.md` agree that training-only selection chooses the
null amplitude, raw and candidate fair CRPS are both `0.0584905850`, ranks and
coverage are unchanged, proper-score and finite-ensemble-reliability families
fail, and boundary, spatial/physical and operational families pass. For the
projection-free slack-limited ablation, all three surfaces agree that every
pixel is blocked, selected and effective amplitudes are `0.0`, every paired
metric and interval equals raw, the same two scientific families fail, and the
same three families pass. Both rows retain `overall_eligible=false`. No
provenance correction was warranted; this negative audit adds no scientific
evidence, authorizes no retuning and leaves publication status `NOT_READY`.

The next 2026-08-29 family-level provenance audit checked the guidance-mixture
and latent-temperature rows. For guidance mixture, the manuscript matrix,
`CLAIM_LEDGER.md` C32 and the corrected compact handoff in
`REPRODUCIBILITY.md` agree that proper-score, finite-ensemble-reliability,
boundary and spatial/physical families fail, operational validity passes and
`overall_eligible=false`. For latent temperature, `PAPER_DRAFT.md`,
`CLAIM_LEDGER.md` C35, `LATENT_TEMPERATURE_RESULT_RECONCILIATION.md` and the
reproducibility handoff agree on raw/candidate fair CRPS
`0.0584905850`/`0.0631177443`, paired delta `0.0046271592`, date interval
`[0.0014355657, 0.0075956683]`, reliability and operational passes, and
proper-score, boundary and spatial/physical failures. No additional provenance
correction was warranted. The audit adds no evidence, permits no retuning and
leaves publication status `NOT_READY`.

The 2026-08-29 rank-first reconciliation adds
`joint_rank_coherent_rank_first_valid` to the manuscript, claim ledger and
reproducibility handoff. All 40 cases complete and operational validity passes,
but `overall_eligible=false`: absolute rank adequacy, proper-score, boundary and
spatial/physical families fail. Fair CRPS worsens by `0.0053490344` with date
interval `[0.0026379728, 0.0085435566]`; exact-one mass and variogram damage
prohibit compensation by the relative rank improvement. This closes the fixed
mechanism without post-result tuning. No eligible calibration has been obtained,
so publication status remains `NOT_READY` and the next mechanistically distinct
contract remains active work.

The subsequent pre-result audit adds
`RAW_MEMBER_REWEIGHTING_RESULT_RECONCILIATION.md` to the mandatory publication
inventory. It freezes mutually exclusive positive and negative interpretations,
the atomic five-file update surface, exact completed counts and the prohibition
on compensation or post-result retuning before any trusted mode or scientific
result exists. The unified checker now fails if this handoff or any of its
decision anchors disappears. This closes an autonomous result-integration gap
only: it creates no mode or evidence, leaves `overall_eligible` unresolved and
keeps publication status `NOT_READY`.

The next 2026-08-29 integration audit adds
`COVERAGE_OCCURRENCE_RESULT_RECONCILIATION.md` as a fail-closed boundary for the
two completed occurrence-calibration jobs. Their scheduler completion records
do not expose the five mandatory family decisions, so no scientific outcome is
inferred. The handoff requires both controller-visible compact payloads in one
atomic audit, verifies `overall_eligible` as their family conjunction, and
predeclares mutually exclusive positive and jointly negative publication
updates. It forbids a repeated computation, compensating interpretation or
activation of score-aware reweighting without two admitted negative gates and a
literal reviewed mode. This closes an autonomous result-integration gap while
publication status remains `NOT_READY`.

The 2026-08-30 raw-member integration continuation adds a fail-closed local
server-adapter boundary and focused inventory tests. The adapter accepts only
the frozen CPU envelope, sole source parameter and `summary_only`, binds dispatch
to all identities in one `admission=GO` payload, and refuses mode substitution
or inventory mutation. It contains no project-data loader, controller change or
invented production mode. The focused 25-test suite passes, but no independent
review or trusted executor registration exists yet; therefore this is engineering
evidence only, creates no scientific result, and publication status remains
`NOT_READY`.

The following semantic-parity audit closes an upstream admission gap: the
review validator now checks the exact contiguous folds, non-circular purge,
training-population feature scaling, unique analog count and deterministic
case-index ordering under an exact distance tie. Focused negative fixtures
reject both purge drift and reversed analog ordering before `admission=GO`.
This remains local engineering evidence: it does not register a trusted mode,
open any evaluation data or change publication status from `NOT_READY`.
