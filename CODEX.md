# Codex Operational Notes

## Hard Boundary: Where Code Runs

All training, evaluation, long-running sampling, and process management commands must be run inside the container on the server.

The local laptop/workstation where code is edited is not the execution environment for experiments. It is only for editing, committing, pushing, and lightweight static checks. Do not start model training from the notebook/laptop copy of the repository.

Before giving or running experiment commands, assume the user is inside the server container unless explicitly told otherwise. If the command is meant for the server, say that it must be executed inside the container.

## Repository Sync On The Server Container

The server container may use GitHub as `origin`. The development machine may use GitLab as `origin`. Always verify the remote before assuming where `origin` points:

```bash
git remote -v
```

For the current `conditioned-model` branch, the server container should be able to pull from GitHub:

```bash
cd /home
git fetch origin conditioned-model
git reset --hard origin/conditioned-model
git rev-parse --short HEAD
```

For the diffusion balanced background-mask run, the expected HEAD is at least:

```text
cd83f2e
```

Verify the exact experiment config before launching:

```bash
grep -n '"training_objective"' config/methods/concat_conditioning_diffusion_balanced_2f.json
grep -n '"model_config"' config/experiments/train_m2m_concat_conditioning_diffusion_balanced_2f.json
```

Expected output must include:

```text
"training_objective": "diffusion"
"model_config": "../methods/concat_conditioning_diffusion_balanced_2f.json"
```

Do not launch the old residual-named config for this experiment.

## Launch Training Inside The Server Container

All launches should be detached so the user can disconnect from SSH safely.

Current pure diffusion + balanced conditioning + explicit background mask run:

```bash
cd /home
mkdir -p logs
nohup python3 -m assim_lib.main \
  --config config/experiments/train_m2m_concat_conditioning_diffusion_balanced_2f.json \
  > logs/train_diffusion_balanced_background_mask.log 2>&1 &
```

Immediately check that the process started:

```bash
ps -eo pid,ppid,stat,args | grep '[p]ython[0-9.]* -m assim_lib\.main'
tail -f logs/train_diffusion_balanced_background_mask.log
```

The log should show:

```text
objective=diffusion
in_channels=13
task=concat_conditioning_diffusion_balanced_2f_1y
```

## Stop Training Safely

Never kill broad Python patterns before inspecting them. First list candidate processes:

```bash
ps -eo pid,ppid,stat,args | grep '[p]ython[0-9.]* -m assim_lib\.main'
```

To stop only the detached training command launched above:

```bash
pids=$(ps -eo pid,args | awk '/[p]ython[0-9.]* -m assim_lib\.main/ {print $1}')
echo "$pids"
if [ -n "$pids" ]; then
  kill -TERM $pids
  sleep 10
  kill -KILL $pids 2>/dev/null || true
fi
```

If multiple training processes are running, identify the exact config path first:

```bash
ps -eo pid,ppid,stat,args | grep '[a]ssim_lib.main'
```

Then kill only the PID(s) for the intended run:

```bash
kill -TERM PID
sleep 10
kill -KILL PID 2>/dev/null || true
```

To target the current diffusion balanced run specifically:

```bash
pids=$(ps -eo pid,args | awk '/train_m2m_concat_conditioning_diffusion_balanced_2f\.json/ && !/awk/ {print $1}')
echo "$pids"
if [ -n "$pids" ]; then
  kill -TERM $pids
  sleep 10
  kill -KILL $pids 2>/dev/null || true
fi
```

Confirm it stopped:

```bash
ps -eo pid,ppid,stat,args | grep '[a]ssim_lib.main' || true
```

## Engineering Requirements

Write code for this repository as production scientific ML code, not notebook glue.

Prefer:

- Vectorized tensor operations over Python loops.
- Batched data/model operations over per-sample calls.
- GPU-friendly PyTorch operations that avoid CPU-GPU synchronization inside hot paths.
- Parallel data loading and prefetching when it matches the dataset and memory budget.
- Reusing existing cached grids, masks, metadata, datasets, and samplers instead of rebuilding them in inner loops.
- Clear shape checks at module boundaries where a wrong channel count would silently corrupt training.
- Focused tests for conditioning modes, masks, sampler arguments, and config/channel consistency.

Avoid:

- Per-pixel Python loops in training, sampling, metrics, or data transforms.
- Calling `.item()`, `.cpu()`, NumPy conversion, plotting, or file I/O inside hot training/sampling loops unless strictly needed.
- Recomputing static tensors such as coordinate grids every solver step.
- Launching long-running jobs from the local development machine.
- Starting a new training run before confirming the config path, `training_objective`, `in_channels`, branch, and commit hash.

## Current Intended Run

Use this experiment for the current background-mask pure diffusion run:

```bash
config/experiments/train_m2m_concat_conditioning_diffusion_balanced_2f.json
```

It points to:

```bash
config/methods/concat_conditioning_diffusion_balanced_2f.json
```

The method config must use:

```json
"training_objective": "diffusion",
"in_channels": 13
```

