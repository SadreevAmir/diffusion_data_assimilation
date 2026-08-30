#!/usr/bin/env python3
"""Fail-closed semantic parity check for a future casewise-selector runner."""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

try:
    from . import casewise_safety_selector_reference as reference
except ImportError:
    import casewise_safety_selector_reference as reference


def _load(path):
    spec = importlib.util.spec_from_file_location("casewise_selector_candidate", path)
    if spec is None or spec.loader is None:
        raise ValueError("runner is not an importable Python module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _close(actual, expected, label):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{label} schema diverges from reference")
        for key in expected:
            _close(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, (list, tuple)) or hasattr(expected, "shape"):
        if len(actual) != len(expected):
            raise ValueError(f"{label} length diverges from reference")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _close(left, right, f"{label}[{index}]")
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{label} diverges from reference")
    elif actual != expected:
        raise ValueError(f"{label} diverges from reference")


def validate_semantic_parity(runner: Path) -> None:
    module = _load(runner)
    constants = ("CASES", "HOLDOUT_SIZE", "PURGE", "PREDICTORS", "RIDGE",
                 "SVD_RELATIVE_CUTOFF", "ACTION_MARGIN", "ACTIONS")
    for name in constants:
        if getattr(module, name, None) != getattr(reference, name):
            raise ValueError(f"runner constant {name} diverges from reference")
    for name in ("build_purged_folds", "scalar_loss", "fit_ridge", "predict_loss", "select_action"):
        if not callable(getattr(module, name, None)):
            raise ValueError(f"runner lacks callable {name}")
    identifiers = tuple(f"case-{index:02d}" for index in range(40))
    reordered = (identifiers[1], identifiers[0], *identifiers[2:])
    for index, values in enumerate((identifiers, reordered)):
        _close(module.build_purged_folds(values), reference.build_purged_folds(values), f"folds[{index}]")
    limits = {"rank_limit": 0.1, "inner_limit": 0.2, "boundary_limit": 0.03}
    diagnostics = {"fair_crps": 0.08, "rank_abs_error": 0.14,
                   "inner_coverage_error": 0.19, "boundary_error": 0.05,
                   "iiee_ratio": 1.03, "edge_ratio": 1.01, "variogram_ratio": 1.07}
    _close(module.scalar_loss(diagnostics, limits), reference.scalar_loss(diagnostics, limits), "loss")
    for index, pair in enumerate(((0.002, 0.0), (0.0019, 0.0), (0.0021, 0.0))):
        _close(module.select_action(*pair), reference.select_action(*pair), f"action[{index}]")
    if reference.np is None:
        raise RuntimeError("numpy is required for decision-bearing ridge parity")
    np = reference.np
    design = np.asarray([[((i + 2) * (j + 3) % 29) / 7.0 + i * 0.01 for j in range(6)] for i in range(24)])
    targets = np.asarray([0.2 + 0.03 * i + (i % 4) * 0.007 for i in range(24)])
    descriptor = np.asarray([0.1, 0.4, 0.3, 0.2, 0.05, 0.01])
    expected = reference.fit_ridge(design, targets)
    actual = module.fit_ridge(design, targets)
    _close(actual, expected, "ridge")
    _close(module.predict_loss(actual, descriptor), reference.predict_loss(expected, descriptor), "prediction")


def main():
    parser = argparse.ArgumentParser(description="Validate frozen casewise-selector semantics.")
    parser.add_argument("runner", type=Path)
    args = parser.parse_args()
    validate_semantic_parity(args.runner)
    print("GO")


if __name__ == "__main__":
    main()
