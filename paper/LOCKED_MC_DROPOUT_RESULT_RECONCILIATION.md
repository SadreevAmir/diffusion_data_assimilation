# Locked-MC-dropout compact-result reconciliation

Status: pre-result, fail closed. This checklist records no scientific outcome.

## Admissible evidence unit

Reconcile only the compact summary produced by
`locked_mc_dropout_p010_gate_valid` from
`locked_mc_dropout_p010_sampling_valid`. Sampling completion, progress metadata,
or any other gate id is not a scientific substitute. Raw ensembles remain on
the server.

## Wrapper-only recovery boundary

The audited internal sampler completed the exact 40-by-10 output set, but the
outer wrapper failed because its metadata validator expected embedded cases
while the worker recorded `cases_file=cases.json`. This is an operational
schema failure, not a scientific result, and it does not authorize a GPU retry.
The internal samples must remain server-side and unchanged.

Admit a recovered sampling artifact only if a trusted server-CPU finalizer
validates the exact failed job, publication commit, recorded launch, external
case schema, all sample hashes and finiteness before copying within the server
result root. The dependent gate must name that recovered artifact as its exact
source. Until both recovery provenance and the decision-bearing compact gate
are present, record neither a positive nor a negative dropout claim. Worker
completion alone is insufficient evidence, and fields must not be merged from
the failed wrapper, recovery summary and gate payload.

Before admitting either a positive or negative claim, require all of the
following in the same compact payload:

- exactly 40 completed cases and ten finite members per case;
- candidate id `locked_mc_dropout_p010_final_ema_ensemble`,
  `gate.overall_eligible`, `no_compensation_across_families=true`, and an
  explicit decision for every mandatory proper-score, finite-ensemble
  reliability, boundary, spatial/physical and operational family;
- raw and candidate `analysis_fair_crps`, their relative change, paired-date
  interval and the separately labelled non-overlapping four-case-block
  sensitivity interval;
- raw and candidate `analysis_crps` with its frozen tolerance decision;
- the final EMA checkpoint identity, unchanged unit-latent seed schedule and
  successful cross-check of its initial-noise hashes against the completed
  latent-temperature reference;
- exact dropout probability `0.1`, exactly fourteen admitted training-time
  dropout layers, eval mode outside those layers, elementwise masks locked
  across every DOPRI5 RHS call, and the fixed seed rule
  `271828000 + 140*case + 14*member + layer`;
- all 5,600 frozen-mask checks passing, with zero refreshed, substituted,
  missing or non-finite masks or members and unchanged conditioning and solver
  metadata.

Any missing field, non-finite value, identifier mismatch, incomplete case,
failed initial-noise or mask check, or disagreement between a family decision
and its reported metric fails reconciliation. Do not infer a missing decision
from an aggregate mean or merge fields from separate payloads.

## Atomic publication update

After the payload passes the checks above, update all five files in one change:

1. `PAPER_DRAFT.md`: report the frozen mechanism, raw/candidate proper scores,
   paired-date interval, every mandatory-family decision and no-compensation
   conclusion.
2. `CLAIM_LEDGER.md`: add one claim row whose status follows the complete gate,
   cites the compact payload and retains the one-checkpoint development-period
   limitation.
3. `REPRODUCIBILITY.md`: record the exact sampling and gate ids, checkpoint,
   latent-hash cross-check, layer admission, seed rule, 5,600 mask checks,
   case/member completeness and compact reconciliation anchors.
4. `RESEARCH_PLAN.md`: close the mechanism with its frozen decision and name
   the next predeclared route, if any, without tuning this construction.
5. `PUBLICATION_READINESS.md`: update the scientific blocker honestly while
   retaining every independent-evaluation, baseline, authorship and release
   blocker that this result does not resolve.

Run `python3 paper/check_publication_artifacts.py` after the atomic update. A
passing local integrity audit checks consistency only; it never overrides the
trusted gate.

## Decision branches

Positive admission requires all of these simultaneously:

- `gate.overall_eligible=true` for candidate
  `locked_mc_dropout_p010_final_ema_ensemble`;
- candidate `analysis_fair_crps` is at least 3% lower than raw and the upper
  endpoint of the paired-date 95% interval for candidate minus raw is below
  zero;
- `analysis_crps` satisfies its frozen tolerance;
- every mandatory family and every operational provenance, initial-noise and
  frozen-mask check passes.

Only then may the manuscript call the construction an eligible
development-period calibration. This does not establish independent
generalization.

If any requirement fails, record the method as a negative mechanism result,
name every failed family and its decision-bearing metrics, and do not change
the dropout probability, layer set, mask granularity, refresh rule, seed,
checkpoint, solver or member schedule. The next route may be admitted only
through its already frozen reviewed contract; this result cannot authorize
post-hoc tuning.
