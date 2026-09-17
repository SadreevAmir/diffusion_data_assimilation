# Independent review request: calibrated probabilistic sea-ice dynamics

You are an independent scientific reviewer. Read this document and every file in
the attached bundle before answering. Do not accept the current diagnosis or
proposed architecture by default. Separate mathematical facts, empirical facts,
working hypotheses, and claims that rely on oracle future information.

## Required output

Return one self-contained review with these sections:

1. **Verdict:** `GO`, `HOLD`, or `NO-GO` for a face-aware spatial refinement
   cascade without a separate occurrence-mask model.
2. **Mathematical audit:** state precisely whether the proposed refinement law
   is capable of representing the complete mixed-support distribution.
3. **Main failure risk:** identify the single most likely reason it would fail
   on the real archive despite passing the synthetic test.
4. **Minimal decisive experiment:** specify one bounded experiment, exact arms,
   frozen inputs, metrics, and falsification gates. Avoid a large redesign.
5. **Oracle audit:** list every place where future truth enters and which claims
   remain valid after removing it.
6. **Recommended next action:** implementation-level instructions sufficient for
   another researcher to run the experiment.

Do not propose opening test-2023. Do not recommend a broad hyperparameter sweep.
Do not treat visually smooth samples or a flat pooled rank histogram as adequate
evidence. Prefer paired proper-score comparisons and explicit mixed-support
reliability.

## Scientific objective

Build a publication-quality calibrated ensemble model for joint Arctic sea-ice
concentration (SIC) dynamics at leads `+3/+6/+9` days. The model must preserve
mean forecast skill while producing coherent member-level scenarios, calibrated
spread/ranks, correct zero/cap atoms, and realistic spatial texture.

## Current strongest end-to-end baseline

`base32 EMA8192`, 2.12M parameters, joint `+3/+6/+9`, evaluated on 24
seasonally distributed holdout-2020 dates with 16 members:

| Metric | +3 | +6 | +9 |
|---|---:|---:|---:|
| Fair CRPS | 0.01704 | 0.02204 | 0.02342 |
| Ensemble-mean RMSE | 0.07268 | 0.08714 | 0.09012 |
| Spread/RMSE | 0.770 | 0.683 | 0.707 |
| Rank-TV | 0.01333 | 0.02485 | 0.02477 |

The baseline is skillful but underdispersed and has incorrect stable-pack
texture. In regions with truth SIC >= 0.95, neighbor-L1 is:

| Support | +3 | +6 | +9 |
|---|---:|---:|---:|
| Truth | 0.00412 | 0.00426 | 0.00436 |
| Generated occurrence | 0.02815 | 0.03049 | 0.03000 |
| Oracle truth occurrence | 0.02623 | 0.02566 | 0.02559 |

Thus occurrence errors materially affect absolute forecast skill, but replacing
generated occurrence with truth does not remove the approximately sixfold
stable-pack texture error.

## Established negative results

- Longer training after EMA8192 contracts member variance and worsens proper
  scores, even when individual maps become visually smoother.
- The old independently trained multiresolution cascade produced
  block/checkerboard artifacts and poorly organized fine residuals.
- More Euler steps, fixed colored sources, source temperature changes,
  affine/spread post-calibration, and signed-increment auxiliary objectives did
  not fix the underlying spatial law.
- A matched cap-tail dequantization A/B (`SIC-M55`, 1024 updates per arm) changed
  iid cap-tail noise to a Gaussian-copula tail while preserving marginals. It
  was rejected: candidate/control mean fair CRPS was
  `0.02088712/0.02089055`, the bootstrap interval crossed zero, spread/RMSE was
  identical (`.763/.676/.698`), and stable-pack roughness improved only
  `1.48%/0.70%/1.25%`.

The current diagnosis is finite learned transport: terminal local regression
leaves incoherent fine-scale phase while earlier reverse-flow segments contract
coherent coarse member variability. This is plausible but not proven uniquely.

## Later coarse/root experiments and their claim boundary

A compact physical `5x4` root generator was trained for 1024 updates. These
experiments use **oracle future support `truth_Z`** and are therefore mechanism
diagnostics, not end-to-end forecasts.

- Raw full physical root on 24 retrospective train cases:
  fair CRPS `0.0321303`, RMSE `0.0917406`, spread/RMSE `1.76297`, rank-TV
  `0.17759`. It is overdispersed under oracle support.
- A frequency-adjustment A/B improved paired active CRPS by approximately
  `-6.97e-05`, with CI95 `[-1.162e-04, -2.222e-05]`, but failed rank and
  spread gates and was rejected.
- A 256-update eight-member proper-score fine-tune was also rejected: paired
  active-CRPS CI95 `[-2.837e-04, 9.685e-05]` crossed zero and rank/spread gates
  failed.
- A retrospective contraction diagnostic found spread/RMSE closest to one near
  scale `0.575`; active CRPS was minimized near `0.675`.
- A train out-of-fold scale calibrator was accepted only as a mechanism:
  full-root fair CRPS `0.0248479 -> 0.0232232`, spread/RMSE
  `1.74808 -> 1.19312`, rank-TV `0.15177 -> 0.13958`, unchanged ensemble mean,
  and paired active-CRPS delta `-0.0032725` with CI95
  `[-0.0055703, -0.0007242]`.

Critical limitation: the calibrator is OOF within the train archive, but the
root model itself saw all train years, and all these results condition on oracle
`truth_Z`. They do not establish calibration after replacing future truth with
a generated support process.

## Mathematical status of a resolution cascade

For deterministic restrictions `X_l = D_l Y`, a lossless probability
factorization exists:

```text
P(dX_0,...,dX_L | C)
  = P_0(dX_0 | C) product_l K_l(dX_l | X_{l-1}, C),

with K_l{D_{l-1} X_l = X_{l-1}} = 1.
```

This is exact only if each `K_l` is a joint whole-field/whole-trajectory kernel,
not a product of independent local child allocators. Generated-parent values may
fall outside the truth-parent training support; the regular-conditional-law
theorem does not identify behavior there. Exact conservation is a decoder
invariant, not proof of a correct forecast law.

For one parent budget `S`, child totals `x_i`, and capacities `c_i`, the earlier
finite-logit capped-simplex decoder covered only the interior. It did not exactly
represent zero/cap boundary faces. A mathematically complete hard cascade needs
an explicit face law inside each refinement kernel.

## New face-aware refinement algorithm

The new algorithm does **not** generate an occurrence mask as a separate model.
For a fixed autoregressive child order and remaining budget `R`, child `i` has
the exact feasible interval

```text
lower_i = max(0, R - sum(capacities of remaining children))
upper_i = min(c_i, R).
```

The refinement kernel chooses one of three branches:

- exact lower face;
- continuous interior fraction in `(0,1)`;
- exact upper face.

The final child receives the exact remainder. Consequently every sample obeys
the parent budget and child capacities algebraically. Zero/interior/cap status
is internal to the refinement kernel rather than produced by an external mask
stage. Any fixed ordering remains distributionally complete as an autoregressive
factorization, although ordering bias and practical whole-field parameterization
remain empirical risks.

### Synthetic admission result

The bundled script tested both a single refinement and a recursive
`1x1 -> 2x2 -> 4x4` irregular-water hierarchy:

- 6,000 permutation round-trips;
- maximum one-level reconstruction error `1.11e-15`;
- maximum two-level native reconstruction error `8.88e-16`;
- maximum root budget error `1.33e-15`;
- no meaningful lower/upper-bound violation;
- a tiny learned conditional kernel matched held-out synthetic distributions:
  child-mean MAE `0.00890`, zero/cap frequency MAE `0.00962`, covariance relative
  error `0.0708`, and energy distance `0.00228`;
- all predeclared synthetic admission gates passed.

This proves only algebraic representability and toy learnability. It does not
test spatially joint native-grid learning, generated-parent robustness, or
sea-ice forecast calibration.

## Central architecture question

Should the next model preserve a separate occurrence-mask generator, or should
open/interior/cap faces be generated inside joint resolution-refinement kernels?

The evidence does **not** establish that a separate mask stage is necessary.
Oracle-mask experiments were diagnostic interventions, not architecture
proofs. The proposed face-aware kernel can represent exact support atoms without
a separate mask model. However, a local/blockwise implementation would still be
insufficient unless a shared spatial backbone generates the entire field and all
leads jointly.

An alternative is one generator in an exactly invertible masked
multiresolution basis. That avoids generated-parent shift but gives weaker hard
mixed-support guarantees unless the observation law is handled explicitly.

## Proposed bounded experiment to audit or replace

The current candidate is a **train-only two-level screen**, not a full model:

- Resolution: `80x64 -> 160x128 -> native` or the largest pair that fits the
  existing bounded budget; do not start at `20x16`.
- One shared whole-field backbone produces all `+3/+6/+9` refinement variables.
- Arm A: interior-only capped-simplex refinement, equivalent to the naive
  continuous allocation law.
- Arm B: face-aware lower/interior/upper refinement with identical backbone,
  conditioning, optimizer, schedules, dates, seeds, and compute.
- No separate occurrence model in Arm B.
- Truth-parent conditional pretraining is allowed, but final scoring must use a
  generated-parent rollout. Do not pair an arbitrary generated parent with an
  unchanged fine truth target and call it exact conditional training.
- First run CPU/native-data codec tests and a tiny memorization screen. Only
  then run one bounded GPU pilot.

Candidate evidence must include:

- exact conservation and zero/cap reliability;
- fair CRPS and ensemble-mean RMSE;
- spread/RMSE and stratified randomized ranks;
- occurrence Brier/reliability, cap reliability, and positive-interior scores;
- edge/area trajectory scores;
- stable-pack neighbor-L1 and spatial variogram;
- coarse/detail spread decomposition;
- fixed all-member panels, not only ensemble means.

The reviewer should decide whether this is the right minimal experiment or
replace it with a more decisive design.

## Data and evaluation restrictions

- Static land/ocean geometry is a legitimate observed input.
- Future truth occurrence/support is oracle information and cannot be an
  operational input.
- Development-2021 and test-2023 must remain closed during algorithm admission.
- Test-2023 is reserved for the final selected candidate.
- Do not infer physical conservation across `+3/+6/+9`; ice can grow, melt, and
  cross the domain boundary. Conservation here is only between representations
  of the same member and horizon.
- SIT semantics are strongly but not completely established as floe thickness;
  `V=A*H` is therefore a working product proxy, not an unquestionable physical
  volume claim.

## Files in the bundle

- `CLAUDE_HANDOFF.md` — this independent-review request.
- `sic_grain_root_cause_audit_ru.md` — detailed latest grain/root-cause audit.
- `data_driven_calibration_design.md` — mathematical and empirical design audit.
- `audit_face_aware_cascade_toy.py` — executable synthetic algorithm admission.
- `face_aware_cascade_toy_result.json` — exact numerical toy result.
- `face_aware_cascade_toy_comparison.png` — held-out synthetic comparison.

