# Frozen next-method contract: purged analog-residual ensemble dressing

## Scientific role and falsifiable hypothesis

All completed postprocessors transform or reconstruct the existing members.
The topology-preserving result shows that safe within-regime amplitude changes
do not expose enough useful dispersion. This method instead tests whether the
missing uncertainty is represented by spatially coherent forecast-error fields
from other development dates.

The candidate keeps the raw learned-joint ensemble mean as its deterministic
forecast and replaces member anomalies by ten complete historical residual
fields. The hypothesis is that leakage-safe analog residuals improve fair CRPS
by at least 3% and finite-ensemble reliability while the common boundary,
spatial/physical and operational families remain admissible. Failure rejects
the empirical residual-library mechanism; it must not trigger a new analog
metric, residual scale or library size on these dates.

## Frozen folds and residual library

- Use the same 40 ordered cases and five contiguous eight-case holdouts.
- For each holdout, remove the three chronologically nearest cases on each side
  from training, without wrapping.
- On every retained training case form one full spatial residual field as
  verifying field minus raw ensemble mean. No held-out truth enters the library,
  feature normalization or neighbor selection.
- Represent each case by exactly six raw-mean features: weighted ice area,
  weighted ice extent at threshold `0.15`, weighted spatial mean, weighted
  spatial standard deviation, and weighted variogram errors at lags 1 and 4.
  Standardize each feature with the retained training mean and population
  standard deviation; reject a fold if any standard deviation is zero.
- Select the ten training dates with smallest squared Euclidean feature
  distance. Break ties by earlier ordered case index. The library size, features,
  metric and tie rule are fixed and are not hyperparameters.

## Candidate construction

For held-out raw mean field `m` and selected residual fields `r_j`, construct
member `j` as `clip(m + r_j, 0, 1)`. Use each selected residual exactly once and
retain its full grid geometry; do not rescale, rotate, localize, recenter or
rank-shuffle residuals. Record clipping mass at both bounds and the candidate
ensemble-mean displacement from `m`. The displacement is permitted but is
evaluated by the unchanged deterministic, boundary and spatial/physical gates.

## Frozen decision

Apply the full existing no-compensation gate on all 40 held-out cases. A pass
requires `overall_eligible=true`, every mandatory family true, all metrics
finite and all folds operational. In particular, fair CRPS must improve by at
least 3% with its paired date interval excluding zero; ordinary CRPS cannot
worsen beyond 1%; all finite-ensemble reliability criteria must pass; and no
boundary, spatial/physical or deterministic failure may be compensated by a
proper-score gain. Report the existing paired date bootstrap and fixed
four-case-block temporal sensitivity.

Interpret failures as follows:

- proper-score or reliability failure rejects useful analog-residual diversity;
- boundary failure attributes rejection to clipping or event-frequency damage;
- spatial/physical failure shows that historical residual coherence does not
  transfer safely between the selected states;
- zero feature variance, fewer than ten eligible training dates, or any
  non-finite output is an operational failure, not evidence for the hypothesis.

## Execution and artifact contract

All fitting, construction and analysis run server-side using the existing source
experiment. A future reviewed runner must accept only
`source_experiment=joint_full_condition_validation_2022`; it must expose no
feature, fold, purge, neighbor-count, distance, clipping, scale, seed, threshold
or path parameter. Retrieval is `summary_only`. The compact return must contain
the complete gate, raw/candidate aggregate metrics, paired uncertainty,
per-fold selected case identifiers, feature-normalization validity, clipping
masses and operational counts. No raw member, truth or residual field is
requested.

Design frozen after reconciliation of the rejected topology-preserving result;
no experiment has been launched.
