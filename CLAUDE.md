# Claude Code repository instructions

Work as an independent scientific collaborator on the calibrated probabilistic
sea-ice dynamics project. Your first task is defined in `CLAUDE_HANDOFF.md`.
Read that file completely before editing or running experiments.

## Required first actions

1. Run `git status --short --branch` and preserve all pre-existing changes.
2. Verify commit `3f3fac1` exists in the current history.
3. Read, in full:
   - `CLAUDE_HANDOFF.md`
   - `docs/research/sic_grain_root_cause_audit_ru.md`
   - `docs/research/data_driven_calibration_design.md`
   - `docs/research/audit_face_aware_cascade_toy.py`
   - `.codex_artifacts/face_aware_cascade_toy_v1/result.json`
4. Re-run the synthetic toy before trusting its reported result.
5. Write the independent review requested by the handoff before implementing a
   real-data candidate.

## Git collaboration

- This is a dirty shared checkout. Existing modifications and untracked files
  belong to the user or another agent.
- Never run destructive cleanup, reset, checkout-overwrite, or bulk staging.
- Stage only explicit paths you created or intentionally modified.
- Make a separate commit for the review before any implementation commit.
- Do not amend or rewrite existing commits.

## Scientific constraints

- Static land/ocean geometry is observed information. Future truth occurrence is
  oracle information and is forbidden in an operational forecast.
- Oracle experiments are allowed only as clearly labelled diagnostics.
- Keep development-2021 and test-2023 sealed during admission.
- Do not equate exact spatial aggregation with conservation across forecast
  horizons.
- Do not accept visual smoothness, train loss, or pooled ranks alone as evidence.
- A valid candidate must be assessed with paired proper scores, stratified atom
  reliability, spread, edge/area trajectories, spatial texture, and all-member
  panels.
- Avoid broad sweeps. Prefer one causal A/B with frozen seeds and predeclared
  gates.

## Infrastructure constraints

- Do not alter VPN, Cisco, Tailscale, Amnezia, host routes, dispatchers, or
  unrelated services.
- If server access is unavailable, do not improvise network repairs.
- No GPU run before CPU admission. Any authorized later run is limited to one GPU
  with UUID `GPU-40ff1cbb-07cc-992e-23cb-8fe181972b76`, after the established
  five-minute idle check, with ClearML tracking.
- Use at most eight CPU threads for numerical jobs.

## Communication

Write research conclusions and review documents in Russian. Be explicit about
claim boundaries, rejected hypotheses, and information leakage. If a required
choice would materially change the scientific design, stop and ask the user.
