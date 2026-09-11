"""Train-only low-rank Gaussian reference law for standardized coarse residuals.

The diagonal control is exactly ``N(0, I)``.  Correlated candidates preserve
every one-dimensional marginal variance and only change dependence between
fixed active coarse coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import torch


def tensor_sha256(value: torch.Tensor) -> str:
    canonical = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(canonical.shape)).encode())
    digest.update(str(canonical.dtype).encode())
    digest.update(canonical.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ActiveCoarseLayout:
    """Stable channel-major vectorization of active coarse ocean cells."""

    active_cells: torch.Tensor
    ocean_fraction: torch.Tensor
    channels: int = 6

    @classmethod
    def build(
        cls, active: torch.Tensor, ocean_fraction: torch.Tensor, *, channels: int = 6
    ) -> "ActiveCoarseLayout":
        if active.shape != ocean_fraction.shape or active.ndim != 4 or active.shape[:2] != (1, 1):
            raise ValueError("reference layout requires [1,1,H,W] active and fraction tensors")
        if not torch.isfinite(ocean_fraction).all() or torch.any((ocean_fraction < 0) | (ocean_fraction > 1)):
            raise ValueError("ocean fractions must be finite in [0,1]")
        cells = active[0, 0].bool().cpu()
        fractions = ocean_fraction[0, 0].cpu()[cells]
        if cells.sum() == 0 or torch.any(fractions <= 0):
            raise ValueError("active layout must contain strictly positive ocean fractions")
        return cls(cells, fractions, channels)

    @property
    def dimension(self) -> int:
        return self.channels * int(self.active_cells.sum())

    @property
    def coordinate_fraction(self) -> torch.Tensor:
        return self.ocean_fraction.repeat(self.channels)

    def vectorize(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 4 or field.shape[1] != self.channels or tuple(field.shape[-2:]) != tuple(self.active_cells.shape):
            raise ValueError("field is incompatible with the fixed active layout")
        selected = field[..., self.active_cells.to(field.device)]
        result = selected.reshape(field.shape[0], -1)
        if not torch.isfinite(result).all():
            raise FloatingPointError("active standardized residual contains NaN/Inf")
        return result

    def restore(self, flat: torch.Tensor) -> torch.Tensor:
        if flat.ndim != 2 or flat.shape[1] != self.dimension:
            raise ValueError("flat field is incompatible with the fixed active layout")
        result = torch.zeros(
            flat.shape[0], self.channels, *self.active_cells.shape,
            dtype=flat.dtype, device=flat.device,
        )
        result[..., self.active_cells.to(flat.device)] = flat.reshape(
            flat.shape[0], self.channels, -1
        )
        return result


@dataclass(frozen=True)
class LowRankUnitDiagonalGaussian:
    """K = diag(D) + U U^T with diag(K)=1 and D strictly positive."""

    alpha: float
    rank: int
    low_rank_basis: torch.Tensor
    diagonal_normalizer: torch.Tensor

    @property
    def dimension(self) -> int:
        return int(self.diagonal_normalizer.numel())

    @property
    def is_identity(self) -> bool:
        return self.rank == 0 and self.alpha == 1.0

    def factors(self, *, dtype=torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.diagonal_normalizer.to(dtype=dtype)
        if self.is_identity:
            return torch.ones_like(h), torch.empty(h.numel(), 0, dtype=dtype)
        basis = self.low_rank_basis.to(dtype=dtype)
        diagonal = self.alpha * h.square()
        low_rank = math.sqrt(1.0 - self.alpha) * h[:, None] * basis
        return diagonal, low_rank

    def sample_flat(
        self,
        count: int,
        *,
        generator: torch.Generator,
        latent_generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if count <= 0:
            raise ValueError("sample count must be positive")
        xi = torch.randn((count, self.dimension), generator=generator, dtype=dtype)
        if self.is_identity:
            # Exact legacy branch: no latent generator and no additional draw.
            return xi
        if latent_generator is None:
            raise ValueError("correlated reference requires an independent latent generator")
        z = torch.randn((count, self.rank), generator=latent_generator, dtype=dtype)
        h = self.diagonal_normalizer.to(dtype=dtype)
        basis = self.low_rank_basis.to(dtype=dtype)
        return h * (
            math.sqrt(self.alpha) * xi
            + math.sqrt(1.0 - self.alpha) * (z @ basis.T)
        )

    def gaussian_score(self, values: torch.Tensor) -> torch.Tensor:
        """Return (w^T K^-1 w + log det K)/(2d) in FP64."""
        quadratic, logdet = self.gaussian_score_components(values)
        return (quadratic + logdet) / (2.0 * self.dimension)

    def gaussian_score_components(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-sample quadratic terms and the shared FP64 log determinant."""
        x = values.double()
        if x.ndim != 2 or x.shape[1] != self.dimension or not torch.isfinite(x).all():
            raise ValueError("score values must be finite [N,d]")
        if self.is_identity:
            return x.square().sum(dim=1), torch.zeros((), dtype=torch.float64)
        diagonal, low_rank = self.factors(dtype=torch.float64)
        inverse_diagonal = diagonal.reciprocal()
        small = torch.eye(self.rank, dtype=torch.float64) + low_rank.T @ (
            inverse_diagonal[:, None] * low_rank
        )
        chol = torch.linalg.cholesky(small)
        projected = (x * inverse_diagonal) @ low_rank
        correction = torch.cholesky_solve(projected.T, chol).T
        quadratic = (x.square() * inverse_diagonal).sum(dim=1) - (
            projected * correction
        ).sum(dim=1)
        logdet = torch.log(diagonal).sum() + 2.0 * torch.log(torch.diag(chol)).sum()
        return quadratic, logdet


def identity_reference(dimension: int) -> LowRankUnitDiagonalGaussian:
    if dimension <= 0:
        raise ValueError("identity dimension must be positive")
    return LowRankUnitDiagonalGaussian(
        alpha=1.0,
        rank=0,
        low_rank_basis=torch.empty(dimension, 0),
        diagonal_normalizer=torch.ones(dimension),
    )


def fit_reference(
    standardized_residuals: torch.Tensor,
    coordinate_fraction: torch.Tensor,
    *,
    rank: int,
    alpha: float,
    seed: int,
    oversampling: int = 8,
) -> LowRankUnitDiagonalGaussian:
    """Fit a weighted full-second-moment randomized low-rank reference."""
    basis = fit_weighted_second_moment_basis(
        standardized_residuals,
        coordinate_fraction,
        max_rank=rank,
        seed=seed,
        oversampling=oversampling,
    )
    return reference_from_basis(basis, rank=rank, alpha=alpha)


def fit_weighted_second_moment_basis(
    standardized_residuals: torch.Tensor,
    coordinate_fraction: torch.Tensor,
    *,
    max_rank: int,
    seed: int,
    oversampling: int = 8,
) -> torch.Tensor:
    """Fit the maximum-rank basis once so rank/shrinkage scoring can reuse it."""
    x = standardized_residuals.float().cpu()
    fraction = coordinate_fraction.float().cpu()
    if x.ndim != 2 or fraction.shape != (x.shape[1],) or not torch.isfinite(x).all():
        raise ValueError("fit arrays have incompatible shapes or non-finite values")
    if torch.any(fraction <= 0) or max_rank <= 0 or max_rank >= min(x.shape):
        raise ValueError("fractions and rank are incompatible with the fit panel")
    q = min(max_rank + oversampling, min(x.shape))
    weighted = x * torch.sqrt(fraction)[None]
    generator = torch.Generator().manual_seed(seed)
    omega = torch.randn((x.shape[1], q), generator=generator) / math.sqrt(q)
    sketch = weighted @ omega
    q_left = torch.linalg.qr(sketch, mode="reduced").Q
    small = q_left.T @ weighted
    _, singular, vh = torch.linalg.svd(small, full_matrices=False)
    eigenvalues = singular[:max_rank].square() / x.shape[0]
    weighted_vectors = vh[:max_rank].T
    basis = weighted_vectors * torch.sqrt(eigenvalues)[None]
    basis = basis / torch.sqrt(fraction)[:, None]
    if not torch.isfinite(basis).all():
        raise FloatingPointError("reference basis fit produced NaN/Inf")
    return basis.contiguous()


def fit_weighted_second_moment_basis_exact(
    standardized_residuals: torch.Tensor,
    coordinate_fraction: torch.Tensor,
    *,
    max_rank: int,
) -> torch.Tensor:
    """Exact FP64 leading basis through the sample-space Gram eigensystem."""
    x = standardized_residuals.double().cpu()
    fraction = coordinate_fraction.double().cpu()
    if x.ndim != 2 or fraction.shape != (x.shape[1],) or not torch.isfinite(x).all():
        raise ValueError("fit arrays have incompatible shapes or non-finite values")
    if torch.any(fraction <= 0) or max_rank <= 0 or max_rank >= min(x.shape):
        raise ValueError("fractions and rank are incompatible with the exact fit panel")
    weighted = x * torch.sqrt(fraction)[None]
    if weighted.shape[1] <= weighted.shape[0]:
        # Small synthetic or reduced-coordinate oracle: use the feature-space moment.
        moment = weighted.T @ weighted / x.shape[0]
        eigenvalues, weighted_vectors = torch.linalg.eigh(moment)
        order = torch.argsort(eigenvalues, descending=True)[:max_rank]
        eigenvalues = eigenvalues[order]
        weighted_vectors = weighted_vectors[:, order]
        basis = weighted_vectors * torch.sqrt(eigenvalues)[None]
    else:
        # Production path: exact non-zero spectrum through the smaller sample Gram matrix.
        gram = weighted @ weighted.T / x.shape[0]
        eigenvalues, left = torch.linalg.eigh(gram)
        order = torch.argsort(eigenvalues, descending=True)[:max_rank]
        eigenvalues = eigenvalues[order]
        left = left[:, order]
        weighted_vectors = weighted.T @ left / torch.sqrt(x.shape[0] * eigenvalues)[None]
        basis = weighted_vectors * torch.sqrt(eigenvalues)[None]
    if torch.any(eigenvalues <= torch.finfo(torch.float64).eps):
        raise ValueError("exact reference basis contains a numerically zero requested mode")
    basis = basis / torch.sqrt(fraction)[:, None]
    if not torch.isfinite(basis).all():
        raise FloatingPointError("exact reference basis fit produced NaN/Inf")
    return basis.contiguous()


def reference_from_basis(
    maximum_basis: torch.Tensor, *, rank: int, alpha: float
) -> LowRankUnitDiagonalGaussian:
    if maximum_basis.ndim != 2 or rank <= 0 or rank > maximum_basis.shape[1]:
        raise ValueError("requested rank is unavailable in the fitted basis")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    basis = maximum_basis[:, :rank].contiguous()
    g_diagonal = alpha + (1.0 - alpha) * basis.square().sum(dim=1)
    h = torch.rsqrt(g_diagonal)
    if not torch.isfinite(basis).all() or not torch.isfinite(h).all():
        raise FloatingPointError("reference fit produced NaN/Inf")
    return LowRankUnitDiagonalGaussian(alpha, rank, basis, h.contiguous())


def select_reference_leave_one_year_out(
    values_by_date_and_slice: torch.Tensor,
    coordinate_fraction: torch.Tensor,
    *,
    ranks: tuple[int, ...] = (8, 16, 32, 64),
    alphas: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75),
    seed: int = 97311,
    basis_method: str = "randomized",
) -> dict[str, Any]:
    """Six-fold blocked selection; all 24 slices of a date stay together."""
    x = values_by_date_and_slice
    if x.ndim != 3 or x.shape[:2] != (24, 24):
        raise ValueError("selection requires exactly 24 dates x 24 slices")
    specs = [(1.0, 0)] + [(alpha, rank) for alpha in alphas for rank in ranks]
    scores: dict[tuple[float, int], list[float]] = {spec: [] for spec in specs}
    component_scores: dict[tuple[float, int], dict[str, list[float]]] = {}
    for year_index in range(6):
        heldout_dates = slice(4 * year_index, 4 * (year_index + 1))
        keep = torch.ones(24, dtype=torch.bool)
        keep[heldout_dates] = False
        train = x[keep].reshape(-1, x.shape[-1])
        if basis_method == "exact_fp64_gram":
            maximum_basis = fit_weighted_second_moment_basis_exact(
                train, coordinate_fraction, max_rank=max(ranks)
            )
        elif basis_method == "randomized":
            maximum_basis = fit_weighted_second_moment_basis(
                train, coordinate_fraction, max_rank=max(ranks), seed=seed + year_index,
            )
        else:
            raise ValueError("unknown reference basis method")
        heldout = x[heldout_dates].reshape(-1, x.shape[-1])
        for alpha, rank in specs:
            reference = (
                identity_reference(x.shape[-1]) if rank == 0 else
                reference_from_basis(maximum_basis, rank=rank, alpha=alpha)
            )
            quadratic, logdet = reference.gaussian_score_components(heldout)
            quadratic_term = quadratic / (2.0 * reference.dimension)
            logdet_term = logdet / (2.0 * reference.dimension)
            per_slice = quadratic_term + logdet_term
            scores[(alpha, rank)].append(float(per_slice.reshape(4, 24).mean(dim=1).mean()))
            component_scores.setdefault((alpha, rank), {"quadratic": [], "logdet": []})
            component_scores[(alpha, rank)]["quadratic"].append(
                float(quadratic_term.reshape(4, 24).mean(dim=1).mean())
            )
            component_scores[(alpha, rank)]["logdet"].append(float(logdet_term))
    rows: list[dict[str, Any]] = []
    for alpha, rank in specs:
        fold_scores = scores[(alpha, rank)]
        score_tensor = torch.tensor(fold_scores, dtype=torch.float64)
        rows.append({
            "alpha": alpha, "rank": rank, "fold_scores": fold_scores,
            "fold_quadratic_terms": component_scores[(alpha, rank)]["quadratic"],
            "fold_logdet_terms": component_scores[(alpha, rank)]["logdet"],
            "mean_score": float(score_tensor.mean()),
            "fold_standard_error": float(score_tensor.std(unbiased=True) / math.sqrt(6)),
        })
    best = min(rows, key=lambda row: row["mean_score"])
    threshold = best["mean_score"] + best["fold_standard_error"]
    eligible = [row for row in rows if row["mean_score"] <= threshold]
    selected = sorted(eligible, key=lambda row: (-row["alpha"], row["rank"]))[0]
    identity_scores = scores[(1.0, 0)]
    for row in rows:
        row["paired_fold_differences_to_identity"] = [
            candidate - control for candidate, control in zip(row["fold_scores"], identity_scores)
        ]
    return {
        "basis_method": basis_method,
        "candidates": rows,
        "best": best,
        "one_se_threshold": threshold,
        "selected": selected,
    }
