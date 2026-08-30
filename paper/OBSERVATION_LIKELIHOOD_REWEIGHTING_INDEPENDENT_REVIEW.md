# Independent review: observation-likelihood scenario reweighting

Status: `RELIABILITY_SPECIFICATION_PASS_PRE_ADMISSION`

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

The revised reliability definition now establishes parity with the unchanged
finite-ensemble randomized-rank gate.  Direct inspection of the authoritative
local implementation found that the existing gate randomizes only exact ties:

```text
less + rng.integers(0, ties + 1)
```

The new coordinate is `M*W_lt` without ties and
`M*(W_lt+(J/K_eq)*W_eq)` with the same inclusive integer tie draw.  With equal
weights this simplifies exactly to `less+J`.  Fractional unequal-weight ranks
deposit one mass-preserving soft count across adjacent existing bins; no member
is resampled and no new random source is introduced.

## Decision-bearing deviation

```text
decision_bearing_validation=PASS
deviations=[]
```

The focused oracle fixtures establish all requested specification properties:

1. exact one-hot parity at all eleven equal-weight ranks;
2. inclusive tie parity for duplicated members and truths at both boundaries;
3. explicit below-minimum, between-member and above-maximum behavior;
4. one-count conservation for fractional unequal-weight coordinates; and
5. fail-closed handling of invalid weights, draws, members and truths.

This closes the mathematical deviation only.  Admission remains `NO_GO`
because no independent trusted-runner identity, compact schema, publication
commit or literal controller-visible mode is bound.  No experiment proposal is
authorized by this review.

## Publication consequence

Likelihood reweighting remains a pre-result mechanism.  The revision makes its
rank decision comparable to the raw baseline without claiming that local
specification parity is scientific evidence.  Publication use still requires
atomic independent admission and a trusted compact result.
