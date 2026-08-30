# Independent review: observation-likelihood scenario reweighting

Status: `NOT_READY_RELIABILITY_DEVIATION`

Reviewed artifact:
`NEXT_OBSERVATION_LIKELIHOOD_REWEIGHTING_CONTRACT.md`.
This review is outcome-agnostic, does not authorize an executor mode and does
not inspect project arrays or outcome metrics.

## Result

The member-specific component log likelihood, stable softmax and unequal-weight
fair-CRPS expression are mathematically identified.  In particular, at
`w_m=1/M`, the pairwise correction reduces to
`1/(2*M*(M-1)) * sum_m sum_n |x_m-x_n|`, as required by the existing fair
finite-ensemble score.

The reliability definition does not yet establish the claimed parity with the
unchanged finite-ensemble randomized-rank gate.  For continuous truth with no
exact member tie, the frozen expression

```text
u = W_lt + U*W_eq
```

has `W_eq=0`, so it removes randomization entirely.  At equal weights it emits
the discrete values `0, 0.1, ..., 1` rather than explicitly reproducing the
existing randomized rank-cell mapping.  Therefore the sentence claiming that
the raw baseline is evaluated by an identical unchanged gate is not proven by
the contract and cannot be accepted as runner parity.

## Decision-bearing deviation

```text
decision_bearing_validation=FAIL
deviations=["RELIABILITY_EQUAL_WEIGHT_PARITY_UNSPECIFIED"]
```

Admission remains `NO_GO`.  Before trusted-runner review, the contract must
define one weighted rank-cell transform that:

1. reduces algebraically and fixture-by-fixture to the current randomized-rank
   implementation when all ten weights equal `0.1`;
2. specifies below-minimum, between-member, exact-tie and above-maximum cases;
3. binds the unchanged seed and indexing rule without adding a tuning
   parameter;
4. is checked against the existing equal-weight implementation on synthetic
   fixtures, including duplicated members and boundary truths; and
5. states whether the existing absolute-uniformity and coverage thresholds are
   valid on the resulting scale.  If not, the method must be described as a new
   reliability diagnostic and cannot inherit the old gate thresholds.

No runner, mode, compact schema or proposal should be registered from the
reviewed revision until this deviation is closed with
`decision_bearing_validation=PASS` and `deviations=[]`.

## Publication consequence

This finding does not reject likelihood reweighting as a mechanism.  It rejects
only the current decision rule: a positive or negative result under a changed
reliability scale would not support the intended no-compensation claim.  The
proper-score formula and copy-only boundary/spatial invariants remain suitable
for the next revision.
