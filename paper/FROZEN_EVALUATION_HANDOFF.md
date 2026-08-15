# Frozen independent evaluation handoff

Status: BLOCKED_PENDING_EXTERNAL_AUTHORIZATION

This document fixes the preflight boundary for the confirmatory run. It is not
an authorization to open the locked evaluation data, train a checkpoint or
launch an experiment. No independent result is present in this worktree.

## Required external inputs

The trusted controller must provide all of the following before admission:

1. an explicit authorization naming the clean checkpoint and the frozen
   independent evaluation;
2. a reviewed runner and an implemented mode for that exact evaluation;
3. immutable checkpoint identity, dataset-manifest digest, code revision and
   environment manifest;
4. the deterministic comparator identity and its aligned case manifest;
5. the frozen random seeds and confirmation that every case is evaluated once.

Absence or mismatch of any item is a hard preflight failure. The worktree must
not infer an identifier, substitute the legacy checkpoint, repair a manifest or
fall back to development data.

## Frozen scientific contract

- Method selection is closed. No calibration family, coefficient, threshold,
  fold, seed, ensemble size, stopping rule or metric definition may change in
  response to confirmatory results.
- The selected candidate, if an eligible candidate becomes available through a
  separately reviewed contract, must be compared with the raw learned-joint
  ensemble and the exact deterministic comparator on the same cases.
- The no-compensation families remain those in `RESEARCH_PLAN.md`: proper
  scores, finite-ensemble reliability, boundary behaviour, spatial/physical
  preservation and operational validity. Every mandatory family must pass.
- Date-level cases are the uncertainty unit. Pixels are never independent
  replicates. Any temporal sensitivity must preserve chronological adjacency
  and be labelled predeclared or post-hoc truthfully.
- A failed family rejects the broad calibrated-ensemble claim. Aggregate score
  improvement, a favourable subgroup or a deterministic-mean gain cannot
  compensate for that failure.

## Required compact return package

Retrieval defaults to `summary_only`. The trusted result must contain:

- immutable provenance: checkpoint, code, environment and dataset-manifest
  digests, runner/mode identifier, seeds, case count and ensemble size;
- aggregate raw, candidate and deterministic-comparator metrics with explicit
  units and regions;
- paired date-level effect summaries and predeclared uncertainty intervals for
  every decision-bearing proper-score and spatial/physical metric;
- finite-ensemble rank, attainable-coverage and boundary-mass diagnostics;
- the complete per-family Boolean gate, `no_compensation_across_families` and
  `overall_eligible`;
- counts for attempted, completed, failed and non-finite cases, plus explicit
  convergence and rank-order checks where applicable.

No raw ensemble, case table or unrestricted model artifact is requested by
default. A compact figure source may be added only by exact filename after a
paper claim demonstrates that aggregate summaries are insufficient.

## Stop/go and reconciliation

The run is admissible only after the five external-input conditions above are
attested by the trusted controller. After return, local manuscript integration
must stop on any provenance mismatch, incomplete case count, non-finite metric,
missing family decision or missing comparator alignment. `overall_eligible=true`
supports selection only when every recorded mandatory family is true; otherwise
the outcome is a negative confirmatory result and claims must be narrowed.

Before changing `Publication status`, update `PAPER_DRAFT.md`,
`CLAIM_LEDGER.md`, `REPRODUCIBILITY.md`, `RESEARCH_PLAN.md` and
`PUBLICATION_READINESS.md` from the same compact result, then run
`python3 paper/check_publication_artifacts.py`. Author identity, venue metadata,
release licensing and the external authorization itself remain genuinely
external inputs.
