# CPU reference

These are independently executable analytical baselines, not GPU training code.
Run from the package root:

```bash
python -m unittest discover -s reference -p 'test_*.py' -v
```

Dependencies: NumPy and SciPy. Exact versions tested are recorded in `checks/environment.json`; do not upgrade a working research environment just to match them.

Inputs to score functions have shape `(members, variables)`; masks and area weights must be applied consistently by a repository adapter. Missing values are rejected, not silently removed. Unbiased estimates are not clipped. `u_score` names a formula and does not certify independence. The checked implementation is not a formal verification of the mathematical proofs.

The full experiment drivers and model trainers must be implemented in the user's repository according to `EXPERIMENT_SPECIFICATIONS.md`. No server data, weights, or new ML results are included.
