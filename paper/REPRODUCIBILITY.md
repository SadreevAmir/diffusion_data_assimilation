# Reproducibility handoff for the frozen mechanism result

External primary evidence state: RECONCILED_NEGATIVE

## Reconciled independent primary handoff

The permitted CPU recovery
`external_2024_calendar_global_bias_confirm48_primary_retry2` reuses source
`external_2024_calendar_global_bias_raw48_primary_v1` without resampling. It
applies only the reviewed deterministic inverse for truth normalization; the
superseded `retry1` metrics are scientifically invalid and excluded. It
completed 48 cases with ten members each. The compact metadata records primary
contract digest `f2225da7a05cab53b14604e45bed840a0ec559aed20856ae8ef2dd72d915b8f8`
and confirmatory-gate digest
`57e8dd1859c4ac9a144be68904450926a6098b08a3b58a35e3d7dcc6a5bb9185`.
All 48 raw hashes, 48 truth hashes and 480 member records were verified before
array loading; all 48 candidate arrays were hash-sealed before truth was opened.
The candidate manifest digest is
`55221b8cd165e6adcbb07be403f5e84ea5eb9a0a708ab31e79947ad07f8a5f67`.
All metrics are finite, no network or raw-array retrieval was used, and the
boundary masks equal raw. The authoritative no-compensation decision is
`overall_eligible=false`: proper scores, absolute rank reliability,
truth-relative boundary calibration and spatial/physical preservation fail;
only operational validity passes. Fair CRPS worsens from
`0.05541808434196047` raw to `0.061833300537408264` candidate, and normalized
mean rank worsens from `0.23610946912844127` raw to approximately `0.225`
candidate. These compact anchors support rejection only.

## Amended primary gate provenance

The primary evaluation must report the exact primary and gate digests from
`AMENDED_PRIMARY_EVALUATION_CONTRACT.md` plus trusted controller deploy/audit
attestation. Earlier compact gates remain frozen development evidence and
cannot establish final success. The return must include the absolute
randomized-rank decision, truth-referenced high-SIC diagnostics at
`q={0,.15,.90,.95,.99}`, and `>=.999`/exact-one masses labelled only as encoding
diagnostics. Missing identity, attestation or decision fields fails closed.

This handoff covers the 40-case development-period mechanism analysis and the
completed fixed-contract joint calibration audit. Neither requires raw
ensembles in the publication worktree.

## Rank-coherent runner review handoff

The exact admission checklist is
`paper/RANK_COHERENT_RUNNER_REVIEW_CHECKLIST.md`. It requires a recorded
publication commit and runner digest, oracle parity, negative fixtures and a
synthetic server dry run. Every checkbox is mandatory; an unchecked, failed or
waived item is `NO_GO` and cannot support an experiment proposal.
The same checklist now freezes the single controller-visible admission record:
literal reviewed mode, reviewed publication commit, runner/contract/synthetic
SHA-256 digests, exact test command and sentinel, a `PASS` decision-bearing
directory validation, and an empty deviations list. Missing or placeholder
fields cannot authorize a proposal; the mode must be copied literally and the
digests must not be turned into runtime parameters.

Rank-coherent admission is one combined operation rather than two sequential
checks. `validate_rank_coherent_admission.py` accepts the exact admission JSON
and decision-bearing compact directory, validates the literal mode and frozen
record schema, applies full directory parity with `decision_bearing=True`, and
returns SHA-256 identities for both exact inputs. Its directory digest
length-frames the four frozen filenames and bytes; final byte rechecks reject a
record or compact-output substitution during admission. The focused integration
fixtures mutate each input after its initial check and require fail-closed
rejection. These emitted identities are the only safe inputs to a later atomic
publication reconciliation; separate successful checks do not authorize a
proposal.

The frozen next mechanism is specified by `NEXT_RANK_COHERENT_CONTRACT.md`.
Before trusted integration, its dependency-free executable review oracle is:

```bash
python3 paper/rank_coherent_reference.py
```

The oracle reads no data and cannot launch an experiment. It fails closed on
fold/purge drift, any change to the exact `valid`, 40-case, ten-member,
stride-five development envelope, extra runtime parameters, non-finite or zero-variance
training-only forecast features, incomplete ten-neighbor, ten-date or ten-member
inputs, out-of-range normalized truth ranks, non-finite ordering quantities,
empty, ragged, non-scalar, non-finite or shape-incompatible complete fields,
a changed alpha set, an alpha outside that set, escaped physical bounds, and bounded
projection mean error above `1e-10` at every pixel. It also rejects a compact
gate that omits any mandatory family, uses non-boolean family flags, or reports
an `overall_eligible` value different from their conjunction. The structural
compact validator additionally requires exactly five ordered fold records with
the frozen holdout and purged training indices, exact alpha/boolean consistency
for `no_positive_feasible_alpha`, ten rank-target counts each equal to 40, and
exact projection diagnostics. It rejects missing or extra fields, non-finite or
negative values, fractions outside `[0,1]`, and maximum mean error above `1e-10`.
Negative fixtures exercise incomplete folds, a contradictory zero-alpha flag,
rank-target imbalance, projection-invariant failure, an inconsistent aggregate
delta, and paired means that disagree
with the aggregates, a member-spatial pass flag that violates its frozen
tolerance, and a family flag that disagrees with its criterion conjunction.
The numeric validator requires the three proper-score anchors, exact metric-set
coverage by paired uncertainty, finite arithmetic reconciliation, ordered date
and four-case-block intervals, and exact member-spatial-to-gate linkage.
Synthetic identity checks ensure that
complete anomaly fields, rather than independently shuffled pixel values, are
selected after training-only standardization and deterministic forecast-distance
ties, paired by the frozen date and member order statistics, and carried through
the full spatial candidate construction. Passing this check is a
runner-review prerequisite, not scientific evidence or an implemented mode.
The unified publication audit invokes the same command with the current Python
interpreter and requires its exact success sentinel, so syntax-only acceptance
cannot hide a failing executable invariant.

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

## Frozen ZOIB-EMOS/ECC-Q handoff

The previously frozen design has been executed by the reviewed server runner: a
date-balanced ZOIB-EMOS marginal model with
exactly 13 fitted coefficients, five purged contiguous cross-fitting folds and
deterministic ECC-Q reconstruction. Its sole runtime parameter was
`source_experiment=joint_full_condition_validation_2022`; no ensemble artifact
was retrieved into this worktree. All 40 cases and folds completed, all
reported metrics were finite and ECC-Q strict rank-order violations were zero.

Reconciliation anchors are fair CRPS `0.0584905850` raw and `0.0585569940`
candidate; ordinary CRPS `0.0621082810` and `0.0649912008`; ensemble-mean RMSE
`0.1903366044` and `0.2007737230`; exact-zero mass absolute error `0.396858616`
and `0.008275438`; exact-one mass absolute error `0.009041882` and
`0.0000687234`; randomized-rank discrepancy per observation `0.0790135133`
and `0.0098823033`; inner-order attainable error `0.1759533520` and
`0.2539472381`; established-ice Brier `0.0569726083` and `0.0642804809`; mean
IIEE `0.0796004071` and `0.0871897938`; edge disagreement `0.0351291198` and
`0.0401654091`. Only operational validity passes at family level and
`overall_eligible=false`. The fixed ridge, predictors, optimizer and links must
not be tuned after this result. These compact anchors document a rejected
parametric boundary-aware baseline and must not be used to reconstruct
case-level pairs.

## Completed topology-preserving transport handoff

`paper/NEXT_BASELINE_CONTRACT.md` specifies the pre-result contract for the
completed topology-preserving stratified transport. It freezes purged folds, six
mechanistically spaced strengths, training-only selection, exact event-mask,
boundary-atom and pixel-mean invariants, the complete joint gate and uncertainty
seeds. The reviewed runner accepted only
`source_experiment=joint_full_condition_validation_2022`; no raw ensemble was
retrieved. The compact gate reports `overall_eligible=false`: boundary and
spatial/physical families pass with every memberwise mask exact and maximum
mean error at most `5e-13`, but all three finite-ensemble reliability criteria,
both fair-CRPS criteria and all-fold training feasibility fail. All 40 cases and
reported metrics are complete and finite. The scale set, strata, folds, seed and
thresholds must not be changed after this result. The compact reconciliation
anchors are identical raw/candidate fair CRPS (`0.0584905850`/
`0.0584905850`) and spread-skill (`0.7240662110`/`0.7240662110`); they document
an inactive held-out correction when training feasibility fails, not a rounded
improvement.

## Frozen latent-temperature recovery handoff

The original `latent_temperature_1p30_sampling_valid` wrapper failed after its
audited internal sampler had completed all 40 cases and ten members per case.
The reviewed CPU recovery
`latent_temperature_1p30_sampling_retry1` validates the recorded launch,
external case schema, all 400 sample hashes and finiteness before copying the
unchanged samples within the server result root. It performs no GPU sampling
and creates no new scientific candidate. Only the dependent compact payload
from `latent_temperature_1p30_gate_retry1`, whose source is exactly the recovery
id, may support a positive or negative latent-temperature claim. Completion or
aggregate sampling metadata are insufficient. The separate locked-MC-dropout
sampling and gate chain remains a mechanistically distinct comparison, not a
retry or tuning branch.

The compact payload `latent_temperature_1p30_gate_retry1` is now reconciled to
the recovery source. All 40 cases and 400 members pass checkpoint, scale
`1.30`, seed, base/scaled-hash, finiteness and no-substitution checks. It reports
`overall_eligible=false`: raw/candidate fair CRPS is
`0.0584905850`/`0.0631177443`, paired delta `0.0046271592`, date CI
`[0.0014355657, 0.0075956683]`, and sensitivity-only block CI
`[-0.0005197500, 0.0082640843]`; ordinary CRPS is
`0.0621082810`/`0.0684128432`. Reliability passes, while proper-score, boundary
and spatial/physical families fail.

## Locked-MC-dropout wrapper recovery handoff

The locked-MC-dropout internal worker completed the frozen 40-case, ten-member
sampling envelope. Its outer wrapper failed only because validation expected
cases embedded in metadata while the worker wrote `cases_file=cases.json`.
The trusted server-CPU recovery validated the recorded launch, case schema,
sample hashes and finiteness without GPU recomputation. The dependent compact
gate identifies `locked_mc_dropout_p010_final_ema_ensemble` and records
`overall_eligible=false`. This is now a negative scientific mechanism result,
not an active recovery handoff. The frozen probability, fourteen-layer set,
elementwise locked masks, seed schedule and unchanged gate must not be altered;
the result is also the audited prerequisite for the completed iid calendar
global-bias fallback.

## Completed purged analog-residual handoff

The reviewed runner used only
`source_experiment=joint_full_condition_validation_2022` and the contract in
`paper/NEXT_METHOD_CONTRACT.md`: five contiguous purged holdouts, six
forecast-only features, training-only population standardization, ten complete
nearest residual fields and fixed clipping. All 40 cases have ten distinct
analogs and finite metrics. The gate rejects the candidate. Reconciliation
anchors are fair CRPS `0.0584905850` raw and `0.0643049265` candidate; ordinary
CRPS `0.0621082810` and `0.0674642030`; mean RMSE `0.1903366044` and
`0.1800823325`; mean IIEE `0.0796004071` and `0.1077199457`; extent absolute
error `0.0461536485` and `0.0814045891`; established-ice Brier
`0.0569726083` and `0.0796927236`; lower/upper clipping masses
`0.2435959763`/`0.1027051936`; and maximum mean displacement `0.7530213545`.
Only reliability and operational validity pass. These values document a
rejected mechanism and must not be used to reconstruct paired cases.

The guidance-mixture result is complete and rejected; its operational checks
pass, but proper-score, reliability and boundary criteria fail, and every
spatial/physical criterion fails. It is retained as negative mechanism evidence
and is not retuned.

## Completed coherent-member-offset handoff

The reviewed runner used only
`source_experiment=joint_full_condition_validation_2022` and the frozen purged
coherent-member contract. Five contiguous holdouts with a three-case purge,
the fixed amplitude set, permutation-equivariant member score and bounded-
simplex projection were not changed after execution. All 40 cases are finite.
Training-only selection chooses amplitude `0.0`; raw and candidate fair CRPS
are both `0.0584905850`, randomized ranks and coverage are unchanged, and the
maximum mean-invariance error is zero. Boundary, mean-field, member-spatial and
operational families pass, while proper-score and finite-ensemble-reliability
families fail; `overall_eligible=false`. These compact anchors document a
rejected mechanism and do not authorize reconstruction of case-level pairs.

The subsequent slack-limited ablation used the same sole source and purged
folds, but admitted only a common amplitude allowed by every member's existing
distance to `[0,1]`, with no clipping or projection. Its compact result reports
all 40 cases finite, `blocked_pixel_fraction=1.0`, selected and effective
amplitudes `0.0`, zero new exact boundary values, and exact equality to raw for
every paired metric and interval. The authoritative gate has
`overall_eligible=false`: boundary, mean-field, member-spatial and operational
families pass, while proper scores and finite-ensemble reliability fail. These
anchors are sufficient to audit the mechanism decision without raw ensembles.

## Archived clean-checkpoint deep-ensemble executable handoff

`paper/NEXT_GENERATIVE_METHOD_CONTRACT.md` freezes the subsequent
clean-checkpoint deep ensemble. The reviewed trusted sampling and gate modes
implement this contract, but the route is not selectable as a fast fallback:
measured throughput makes the frozen three-seed training construction a
months-long route. This section preserves a reproducibility contract; it is not
a scientific result, an active dependency or authorization to launch either
stage.
Exactly three clean training seeds, `1701`,
`1702`, and `1703`, use identical non-seed configuration. Every checkpoint
contributes latent seeds `2401`, `2402`, and `2403`; the checkpoint selected by
the frozen case-index rotation also contributes `2404`. The extra-member totals
over forty cases are `14/13/13`, while every case retains ten members.

The reviewed sampling runner interface has only
`source_experiment=joint_full_condition_validation_2022`, and return policy is
`summary_only`; the dependent gate accepts only the sampling experiment as its
source. As a local contract check, execute
`python3 paper/deep_ensemble_reference.py` and require the message
`checkpoint extras 14/13/13; metadata admission fails closed`. The executable
oracle now validates the exact compact pre-score manifest: immutable shared
non-seed configuration hash, three distinct selected-checkpoint hashes, normal
and unique checkpoint selection, all ordered member identities, common-noise
hash equality within each case/latent seed, finite outputs and zero
substitution. Review must additionally verify the full unchanged gate. The
reference establishes construction and metadata-admission contract conformance
only; it is not a scientific result.

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

## Completed iid calendar global-bias mixture handoff

The reviewed server-CPU run `crossfit_iid_calendar_global_bias_mixture_valid`
completed all 40 cases with ten members per case. Its only runtime input was
`source_experiment=joint_full_condition_validation_2022`; the fixed 0.5 expert
probability, 30-day residual kernel, purged contiguous folds and seed schedule
were not exposed for tuning. The compact gate identifies
`crossfit_iid_calendar_global_bias_mixture_v1` and reports
`overall_eligible=false`: finite-ensemble reliability, boundary behaviour,
spatial/physical preservation and operational validity pass, while proper scores
fail. Fair CRPS is `0.0582878868` versus `0.0584905850` raw, with paired date CI
`[-0.00100456, 0.000700186]`; ordinary CRPS is `0.0623775052` versus
`0.0621082810` raw. These compact values are cited directly and were not
reconstructed from raw ensembles. No post-result change to the frozen mixture
contract is admissible.

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

Both figure generators are covered by the required publication regression
suite. The tests exercise deterministic rendering from compact fixtures and
fail-closed rejection of negative metrics, a jointly zero plotting scale, a
zero raw denominator, candidate substitution, duplicate full-region rows and a
non-Boolean gate decision. The long-form consumer is independently tested
against a wrong 160-row envelope and duplicate method/date identities. This
prevents malformed but finite compact input from producing a misleading or
undefined checked-in artifact.

### Execution boundary for documented commands

The following matrix is normative. A command listed as `SERVER_ONLY` is not a
clean-checkout oracle: its mandatory compact input is produced or retained by
trusted server analysis and must satisfy the exact schema below before the
command may run. No raw ensemble is an admissible substitute. Commands absent
from this matrix must not be inferred to accept external compact inputs.

| Command id | Execution class | Exact producer experiment | Compact artifact | Exact mandatory compact-input schema | Verifiable manifest identity |
|---|---|---|---|---|---|
| `case_level_artifacts_long_form` | `SERVER_ONLY` | `joint_existing_ensemble_calibration_audit_valid` | `per_case_metrics.csv` | `per_case_metrics.csv`: exactly 160 rows; columns `target_date`, `fold`, `method` and the full proper-score, rank, boundary and spatial diagnostic family; unique ISO `target_date`/`method` pairs; exactly 40 dates and both exact raw/global-spread method labels on every date | `metadata.json`: `experiment_id == joint_existing_ensemble_calibration_audit_valid`; artifact manifest binds `per_case_metrics.csv` by SHA-256 |
| `calibration_summary_figure` | `SERVER_ONLY` | `joint_crossfit_spread_calibration_valid` | `aggregate_case_mean_metrics.json` | `aggregate_case_mean_metrics.json`: one-row JSON list; `num_cases == 40`; finite raw and corrected values for ordinary CRPS, fair CRPS, spread-skill ratio and all four interval diagnostics | `metadata.json`: `experiment_id == joint_crossfit_spread_calibration_valid`; artifact manifest binds `aggregate_case_mean_metrics.json` by SHA-256 |
| `joint_gate_figure` | `SERVER_ONLY` | `joint_existing_ensemble_mean_preserving_projected_spread_valid` | `aggregate_case_mean_metrics.json` | `aggregate_case_mean_metrics.json`: exact reviewed candidate identifier; exactly two full-region method rows; 40 finite cases per method; finite gate metrics and Boolean `overall_eligible == false` | `metadata.json`: `experiment_id == joint_existing_ensemble_mean_preserving_projected_spread_valid`; artifact manifest binds `aggregate_case_mean_metrics.json` by SHA-256 |

The data-free `python3 paper/rank_coherent_reference.py` command and the
publication regression/integrity commands below are `LOCAL_ORACLE` operations:
they require only versioned files in a clean checkout. This explicit separation
prevents a server generator from being mistaken for an autonomously reproducible
local check.

Each row must resolve all four linked identities before execution: exact producer
experiment, exact compact filename, exact payload schema and the producer's
`metadata.json` artifact-manifest SHA-256 binding for that filename. A file with
the right basename or schema but a different producer or absent hash binding is
inadmissible. The compact CSV and aggregate JSON remain server-side; venue
metadata, author statements and data-release decisions remain external inputs.

Every `SERVER_ONLY` consumer requires the trusted handoff to materialize the
relevant binding as `compact_manifest.json`. The consumer validates that sidecar
inside its own process before parsing the payload, creating output directories
or writing output files. The standalone equivalent is:

Consumer outputs are published through same-directory temporary files followed
by `os.replace`. For the two-output case-level consumer, both files are fully
written and `fsync`-ed before either destination is replaced. The executable
fault fixture forces the second temporary-file write to fail and requires both
pre-existing destinations to remain byte-identical, with no temporary file left
behind. This prepare-stage guarantee does not claim a filesystem-wide atomic
transaction across the two final `replace` calls.

```bash
python3 paper/validate_server_only_manifest.py \
  COMPACT_ARTIFACT compact_manifest.json EXPECTED_PRODUCER_EXPERIMENT
```

The sidecar is strict JSON with exactly `schema_version`, `experiment_id` and
`artifacts`. `schema_version` is `server_only_compact_manifest_v1`;
`experiment_id` must equal the matrix producer; and `artifacts` must contain
exactly one entry mapping the requested artifact basename to its lowercase
64-character SHA-256 digest. The validator hashes the artifact bytes before any
schema-specific consumer runs. A missing entry, extra artifact, producer
substitution, malformed digest or digest mismatch fails closed. The mandatory
regression suite includes fixtures for the valid contract, missing hash entry,
digest mismatch and producer substitution. End-to-end CLI fixtures additionally
exercise all three consumers with bad manifests and bad payloads and require
that no partial SVG or JSON remains.

The publication package itself has a local fail-closed integrity audit:

`NEXT_CONFORMAL_BASELINE_CONTRACT.md` freezes the outstanding conformal
comparison as a purged split-conformal date-level sea-ice-area interval. The
integrity audit requires its nonconformity score, finite-sample quantile,
numeric coverage/width decision and explicit non-executable status. This is a
pre-result contract only: it neither changes the `MISSING` evidence status nor
authorizes an unimplemented trusted mode.

The conformal admission boundary is executable and fail closed. An independent
review must supply one controller-visible JSON record with the exact literal
mode, publication commit, runner hash, contract hash, synthetic-result hash,
test command and sentinel, `decision_bearing_validation=PASS`, and no deviations.
`validate_conformal_area_admission.py` rejects missing or extra keys,
placeholders, malformed identities, a substituted contract digest, any non-PASS
decision or any deviation. This validator does not itself create review
evidence or authorize a mode.

`NEXT_PROBABILISTIC_DA_COMPARISON_CONTRACT.md` freezes the other outstanding
minimum-tier family as a ten-member LETKF comparison. The integrity audit
requires hash-identical observation inputs, the fixed localization/inflation
set, leakage-safe fold selection and the numerical fair-CRPS, absolute-rank and
3D-Var-relative RMSE decision. It is also a pre-result contract only: it leaves
the evidence status `MISSING` and does not authorize an unimplemented mode.

`PAPER_DRAFT.md` also freezes the joint four-outcome interpretation of these two
contracts. Every valid positive/negative combination closes each exact evidence
row on its own terms, while invalid execution leaves that row `MISSING`.
Crucially, baseline-row closure is invariantly separate from learned-joint
calibration eligibility; neither baseline outcome can compensate for a failed
no-compensation gate family. The unified audit requires these interpretation
anchors so future result insertion cannot silently promote comparison
completeness into calibration success.

The cross-document claim audit also preserves the evidence-strength distinction:
conformal and probabilistic DA are the only `MISSING` result rows, whereas the
deterministic comparison is `PRESENT_DEVELOPMENT_ONLY`. It remains a blocker for
the main comparison table without being misreported as a third absent family.

The exact minimum-tier compact identities below are part of the reproducibility
handoff. They must change atomically with the manuscript, claim ledger and
readiness record; `NONE` forbids a decision-bearing presentation.

| Route | Evidence status | Compact evidence record SHA-256 | Allowed presentation |
|---|---|---|---|
| `conformal` | `MISSING` | `NONE` | `PRE_RESULT_ONLY` |
| `probabilistic_da` | `MISSING` | `NONE` | `PRE_RESULT_ONLY` |
| `independent_deterministic` | `PRESENT_DEVELOPMENT_ONLY` | `NONE` | `DEVELOPMENT_ONLY` |

| `eligible_calibration` | `MISSING_ELIGIBLE_RESULT` | `NONE` | `BLOCKED` |

The positive eligible-calibration row uses a separate four-surface compact
identity guard. A result becomes decision-bearing only if manuscript, claim
ledger, readiness and this handoff carry the same SHA-256 record.

Generated text artifacts use a prepare-first, same-directory publication
helper. Every temporary file is flushed and `fsync`-ed before publication. For
multi-file outputs, each existing target is then moved to a same-directory
backup immediately before replacement. If any final filesystem operation raises
an exception, the helper walks the attempted targets in reverse, restores every
pre-existing target and removes targets that did not exist on entry. Executable
fault injection covers both an exception between the two final replacements and
a mixed existing/new-target pair; it also requires removal of all `.tmp` and
`.bak` files. This is deliberately an exception-rollback guarantee, not a
journaled transaction or a claim of recovery after abrupt process or operating-
system termination.

```bash
python3 -m unittest -v \
  paper.test_conformal_area_runner_prototype \
  paper.test_validate_conformal_area_admission \
  paper.test_probabilistic_da_contract_oracle \
  paper.test_probabilistic_da_adapter_parity \
  paper.test_validate_probabilistic_da_admission \
  paper.test_reconcile_probabilistic_da_admission \
  paper.test_rank_coherent_runner_prototype \
  paper.test_rank_coherent_adapter_parity \
  paper.test_validate_rank_coherent_admission \
  paper.test_validate_rank_coherent_manifest \
  paper.test_publication_immutable_identities \
  paper.test_publication_empirical_traceability \
  paper.test_minimum_tier_comparison_audit \
  paper.test_publication_figure_generators \
  paper.test_publication_compact_payload_schemas \
  paper.test_publication_reference_traceability \
  paper.test_publication_limitation_traceability \
  paper.test_publication_claim_status_consistency \
  paper.test_score_aware_reconciliation_consistency \
  paper.test_rank_coherent_result_reconciliation \
  paper.test_rank_coherent_publication_renderer \
  paper.test_validate_server_only_manifest \
  paper.test_server_only_consumer_cli \
  paper.test_atomic_publish \
  paper.test_raw_member_reweighting_reference \
  paper.test_validate_raw_member_reweighting_review \
  paper.test_score_aware_raw_reweighting_reference \
  paper.test_validate_score_aware_raw_reweighting_admission \
  paper.test_validate_score_aware_compact_outputs \
  paper.test_validate_coverage_occurrence_admission
python3 paper/check_publication_artifacts.py
```

The rank-coherent path has a separate executable downstream reconciliation
boundary in `RANK_COHERENT_RESULT_RECONCILIATION.md`. It invokes combined
admission itself and accepts only the identities returned for the exact compact
directory, derives the positive or negative branch from the five literal gate
families, and publishes the manuscript, claim ledger, reproducibility record,
readiness audit and reconciliation record through one rollback-protected
operation. Missing or unequal canonical markers fail before any write.
The publisher independently repeats combined admission and marker derivation;
it does not trust a caller-supplied marker identity.

Before any future literal score-aware mode can become decision-bearing, its
publication update must also follow
`SCORE_AWARE_RESULT_RECONCILIATION.md`. The pre-result state is normative until
combined admission succeeds; afterward the manuscript, claim ledger, readiness
audit, reproducibility record and reconciliation file must be updated atomically
from the same byte-exact compact directory. Their identical canonical marker
also binds the SHA-256 identity of the exact admission JSON and requires the
complete frozen 40-case, 10-member envelope; agreement on incomplete counts is
rejected rather than treated as successful reconciliation.

The combined admission command emits one sorted JSON object containing
`reviewed_mode`, `admission_record_sha256` and `compact_directory_sha256`.
Both digests are outputs of the same operation that performs semantic and
compact-payload admission; the command rereads both inputs before returning and
fails if either changed. The reconciliation marker must copy these emitted
identities, not recompute or manually infer them in a later step.

Before any future literal score-aware mode can become decision-bearing, its
reviewed runner and exact admission JSON must pass semantic parity with the
frozen oracle:

```bash
python3 paper/validate_score_aware_raw_reweighting_admission.py \
  ADMISSION.json TRUSTED_RUNNER.py COMPACT_DIRECTORY
```

The v2 validator binds the runner, frozen contract, independent reference and
the exact four-file compact directory by SHA-256, rejects any deviation,
compares all frozen constants and callable API,
exercises tie/extreme-risk systematic-selection fixtures, and compares the
ridge fit and held-out predictions numerically.  Missing `numpy` is fail-closed
for decision-bearing admission rather than a skipped parity claim.  Passing the
local reference itself is only a validator self-test: admission still requires
a controller-visible literal mode and the separately reviewed trusted runner.

For diagnostic isolation, the directory parity component can also be run alone:

```bash
python3 paper/validate_score_aware_compact_outputs.py COMPACT_DIRECTORY
```

This check recomputes every risk-derived weight, selection, multiplicity,
unique-member count and ESS; reconciles aggregate copy/mask counts and paired-
score point estimates; and requires the operational and overall gate decisions
to agree with recorded invariants and all five mandatory families. It rejects
extra files and does not read raw ensembles.
The decision-bearing combined command additionally uses length-framed filenames
and exact payload bytes to compute one `compact_directory_sha256`, validates
semantic and cross-file parity in the same process, and recomputes the digest at
the end.  Integration fixtures mutate a semantically valid file both before and
after semantic admission; both substitutions fail closed.

The controller-facing downstream invocation is:

```bash
python3 paper/validate_probabilistic_da_admission.py COMPACT_DIRECTORY > ADMISSION.json
python3 paper/reconcile_probabilistic_da_admission.py \
  ADMISSION.json EXPECTED_COMPACT_DIRECTORY_SHA256
```

The expected digest is the controller-retained identity of the bundle being
reconciled. The consumer rejects an outcome without that exact identity and
never reconstructs a decision from metrics.

The first command explicitly runs every publication regression suite; adding a
new required suite therefore also requires updating this documented list. The
second command requires the manuscript, claim ledger, research plan, reproducibility handoff
and readiness audit. It also requires `FROZEN_EVALUATION_HANDOFF.md`, which
fixes the external preflight inputs, no-retuning rule, no-compensation decision
contract and compact return package without authorizing the locked run. The
audit resolves every manuscript figure within `paper/`; parses
linked SVG files as XML; checks contiguous numbered references; and rejects a
ready status with nonempty scientific blockers or a not-ready status without
explicit blockers. It also requires the final fixed diagnostic's fair-CRPS,
established-ice Brier and near-upper-bound anchors, together with the frozen
no-post-hoc-tuning decision, in the manuscript, claim ledger, reproducibility
handoff and readiness audit. This catches a partially updated evidence chain;
it validates package integrity, not scientific eligibility.

## Completed rank-first coherent transport handoff

`joint_rank_coherent_rank_first_valid` completed 40/40 cases with ten members
per case. The compact gate records `overall_eligible=false`: operational
validity passes, while proper-score, finite-ensemble-reliability, boundary and
spatial/physical families fail. Candidate/raw fair CRPS is
`0.0638396195`/`0.0584905850`; the paired candidate-minus-raw delta is
`0.0053490344`, with date-bootstrap interval
`[0.0026379728, 0.0085435566]`. Date-balanced rank total variation improves
from `0.3566938377` to `0.1969742919`, but every absolute rank-histogram
adequacy flag remains false. The frozen alpha set, folds, purge and thresholds
must not be altered after this result.
