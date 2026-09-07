# Structured Gaussian pilot: memory-recovery contract

Status: frozen for independent Astra review; not authorized for execution.

## Failure addressed

The consumed technical retry ended with a bounded CUDA out-of-memory failure
during training.  This contract permits one implementation-only correction.  It
does not authorize another experiment identifier, a launch, full training, or a
change to the scientific hypothesis.

## Single implementation change

Enable non-reentrant activation checkpointing for the existing UNet training
forward, with RNG state preservation enabled.  Checkpoint exactly the same UNet
blocks that the installed model implementation marks as checkpointable.  The
strict-FP32 paired evaluator remains unchanged and runs without activation
checkpointing.

No microbatching, batch-size reduction, spatial cropping, autocast, gradient
scaling, CPU parameter offload, optimizer-state sharding, loss smoothing,
sample rejection, seed selection, or gate weakening is permitted.

## Immutable scientific and optimization invariants

- The target law, linear Gaussian path, conditioning tensors, model
  architecture, reconstructed-velocity loss, data order, timestep draws and all
  seeds are byte-identical inputs to the original pilot.
- Effective batch size and per-example loss weights are unchanged.
- There is exactly one backward reduction, optimizer step, LR-scheduler step and
  EMA update for every original batch, in the original order.
- The run must finish exactly 2128 optimizer steps and retain the original
  LR/EMA schedule and recovery-artifact semantics.
- Activation recomputation must preserve RNG state.  A preflight must fail
  closed if checkpoint recomputation changes any stochastic-layer output or
  gradient relative to an uncheckpointed forward on the same fixed mini-case.
- The existing 8-case, 8-member hash-bound paired strict-FP32 evaluation and all
  quantitative and visual hard vetoes remain unchanged.

## Required admission evidence

Before a new executor contract can be proposed, independent review must receive
one compact CPU/GPU preflight record containing:

1. model/config identity hashes and the exact checkpointed module list;
2. equal forward outputs and parameter gradients for checkpointed and
   uncheckpointed execution on one fixed mini-case, with explicit max-absolute
   and max-relative differences and finite-value checks;
3. equal counts and ordering of backward, optimizer, scheduler and EMA calls;
4. measured peak allocated and reserved CUDA memory for both paths, with the
   checkpointed peak strictly lower;
5. confirmation that evaluation code and the frozen panel/protocol hashes are
   unchanged.

Admission fails if any semantic invariant is absent, if memory does not fall, or
if equivalence exceeds the independently reviewed floating-point tolerance.  A
failed admission closes this correction; it must not be hidden by tuning the
checkpoint partition, seeds, precision, or gates.

