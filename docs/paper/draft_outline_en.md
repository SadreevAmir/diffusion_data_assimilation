# When can a conditional generative model be trusted as a posterior? Identifiability, bandwidth, and out-of-fold recalibration

*Working draft outline (English, for the paper). Research notes and evidence: `docs/research/claude_general_ml_plan_ru.md`, `claude_theory_notes_ru.md`, `claude_autoresearch_analysis_ru.md`. Status 2026-09-19.*

## Abstract (claims)
1. A conditional flow-matching/diffusion model trained on one realisation per condition is a nonparametric conditional-density estimator with an implicit bandwidth over the condition. It reproduces p(y|x) only when the training conditions are dense at the scale over which p(y|x) varies. When the condition identifies the training realisation, the empirical minimiser of any strictly proper objective is a point mass; the model's spread is a bandwidth artefact that shrinks with training.
2. The collapse degrades the marginal scale but not the dependence structure (copula) of the samples.
3. Two remedies with guarantees: (i) noise on the condition acts as an explicit bandwidth — calibration is obtained iff the noise exceeds the nearest-neighbour distance of the training conditions, at the price of information; (ii) out-of-fold recalibration: OOF marginals per observable cell + the model's own copula (ECC/GCA-type), which is auto-calibrated w.r.t. the cell σ-algebra and leaves the joint structure to the generator.
4. Real case: sea-ice data assimilation with a last-year background (6 independent years) sits in the collapse regime; GenCast-type forecasting does not. With the OOF layer the ensemble is calibrated on an out-of-sample year (CRPS −18 %, spread/skill .99, flat ranks, atoms correct); a generator trained without the identifying key is calibrated by construction but loses skill.

## 1. Introduction
- Diffusion/FM ensembles are increasingly read as posteriors (weather, inverse problems, downscaling). GenCast is calibrated; many others are under-dispersed. Why?
- Contribution list = abstract claims; emphasise diagnosis + protocol + guarantees, reuse of classical post-processing blocks.

## 2. Setup and the smoother view
- CFM objective; population limit (Lipman 2023). Empirical minimiser with one realisation per condition (Prop. 1 = P1).
- Kernel-smoother model of a finite-capacity network; Prop. 2 (P2): covariance of the induced law = within-neighbourhood covariance + spread of neighbours' means + O(1/N_h); collapse when N_h → 1.
- Prop. 3 (P3): condition noise resampled per step ⇒ Nadaraya–Watson with bandwidth σ; calibration iff σ ≳ d_NN; cost = information.
- Prop. 4 (P4): copula of a mixture of real realisations converges to the true copula (location-scale families).

## 3. Controlled experiments (known truth)
- Fig. 1: spread/σ_true vs training steps for N ∈ {512, 2048, 8192}, k ∈ {1, 16} (synth_realizations_per_condition).
- Fig. 2: noisy partial observation as condition (synth v3): heat-map of spread/σ over (N, σ_obs) with the NN-distance contour; RMSE vs optimal posterior RMSE.
- Fig. 3: copula error vs marginal collapse (synth_copula_check).
- Fig. 4 (planned): 2-D turbulence with controllable realisations per condition; teacher–student images.

## 4. Out-of-fold recalibration (method)
- Latent with atoms; kriged conditional mean from track innovations (per-date nugget by LOO); OOF residual quantiles per cell (distance to observation × mean); copula: mixture of standardised generator pattern and a long-scale stationary field (exchangeability, marginal preservation — lemmas). Relation to EMOS/ECC/GCA/Schaake.
- Guarantees (P5): consistency and conformal validity of cell marginals; auto-calibration w.r.t. σ(cell); sharpness bound.
- Ablations: gaussian copula only (v9) / generator copula (v11) / standardised (v12, non-exchangeable — rank check) / mixture (v13) / long-scale field (v15); no-generator (background + layer); regime cells; nugget variants; cap tail; per-date scale (oracle bound 0.2 %).

## 5. Real case: sea-ice assimilation (OSSE, real SRAL geometry)
- Data, protocols (native / siconc-only), models (pure CFM: collapse curve; CFG-balanced: constant spread; regularised retrain A: calibrated but under-skilled; protocol-trained C).
- Evidence of no neighbours: NN background distance / anomaly scale = .47 (median); interannual mode 73 % of variance with 6 realisations.
- Layer results on dev-2022 (OOS): CRPS .0415 → .0340 (38/40), RMSE .156 → .137, s/r .99, pooled rank-TV .015 (null .034), zero-atom Brier .069 → .038, ES −16 %, block CRPS −21…−34 %, texture as generator. Stress test train-2019: −32 % mean CRPS, winter regime limitation (seasonal conditional bias) — honest section.
- Tracks-only generators (arms D/E): calibration-by-construction vs skill (pending).
- Multivariate scores; test-2023 frozen run (pending).
- Comparison with 3D-Var protocol (deterministic numbers; probabilistic line).

## 6. Limitations and honest boundaries
One OOS year; cap atom not discriminated; >40 px stratum; choice of cells; copula class is a model; selection optimism among layer versions (~4 % CRPS); OSSE with exact truth on tracks vs real SRAL errors (SIC-dependent, coherent, MIZ mostly unclassified).

## 7. Reproducibility
Scripts in `docs/research/` and `docs/research/general_ml/`; server branch `claude/clean-retrain-v1`.

## Figure/table inventory (existing)
rank histograms (raw / v9 / v13 / v15), atom reliability, layer-by-distance both protocols, collapse curves, member panels v13, 3D-Var 200-day time series, track geometry straight vs real, obs-error model, synthetic curves (JSON; plots to make).
