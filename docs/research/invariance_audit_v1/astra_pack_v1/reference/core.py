"""Independent NumPy/SciPy reference implementations, not repository adapters.

All score functions return smaller-is-better values. Arrays have shape
(members, variables). `u_score` is just a U-statistic formula: calling it
does NOT establish the conditional independence needed for fair evaluation.
No function silently drops missing values, repairs arbitrary covariance
matrices, or clips negative unbiased estimates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.special import logsumexp, ndtr

FloatArray = NDArray[np.float64]
Distance = Literal["marginal", "energy"]
DEFAULT_SCALES = (0.25, 0.5, 1.0, 2.0)


def samples(x: ArrayLike) -> FloatArray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or min(a.shape) < 1 or not np.isfinite(a).all():
        raise ValueError("Expected a finite, nonempty (members, variables) array.")
    return a


def _truth(y: ArrayLike, dim: int) -> FloatArray:
    a = np.asarray(y, dtype=np.float64).reshape(-1)
    if a.shape != (dim,) or not np.isfinite(a).all():
        raise ValueError("Truth must have exactly one finite value per variable.")
    return a[None, :]


def _weights(w: ArrayLike, n: int) -> FloatArray:
    a = np.asarray(w, dtype=np.float64)
    if a.shape != (n,) or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("Weights must be finite and nonnegative.")
    total = a.sum()
    if total <= 0:
        raise ValueError("Weights must have a strictly positive sum.")
    return a / total


def distances(a: ArrayLike, b: ArrayLike, kind: Distance = "marginal") -> FloatArray:
    x, z = samples(a), samples(b)
    if x.shape[1] != z.shape[1]:
        raise ValueError("Variable dimensions do not agree.")
    if kind == "marginal":
        # Average CRPS across coordinates, not a strictly proper joint score.
        return cdist(x, z, metric="cityblock") / x.shape[1]
    if kind == "energy":
        # Euclidean norm, scaled by sqrt(d), NOT squared Euclidean distance.
        return cdist(x, z, metric="euclidean") / np.sqrt(x.shape[1])
    raise ValueError(f"Unknown distance: {kind}")


def empirical_score(x: ArrayLike, y: ArrayLike, kind: Distance = "marginal") -> float:
    a = samples(x)
    return float(distances(a, _truth(y, a.shape[1]), kind).mean()
                 - 0.5 * distances(a, a, kind).mean())


def weighted_score(x: ArrayLike, weights: ArrayLike, y: ArrayLike,
                   kind: Distance = "marginal") -> float:
    a = samples(x)
    w = _weights(weights, len(a))
    first = distances(a, _truth(y, a.shape[1]), kind).ravel()
    return float(w @ first - 0.5 * w @ distances(a, a, kind) @ w)


def u_score(x: ArrayLike, y: ArrayLike, kind: Distance = "marginal") -> float:
    """Unbiased underlying-distribution score ONLY for conditionally iid draws."""
    a = samples(x)
    n = len(a)
    if n < 2:
        raise ValueError("The U formula requires at least two members.")
    pair = distances(a, a, kind)
    offdiag = pair.sum() - np.trace(pair)
    return float(distances(a, _truth(y, a.shape[1]), kind).mean()
                 - 0.5 * offdiag / (n * (n - 1)))


def cross_score(a: ArrayLike, b: ArrayLike, y: ArrayLike,
                kind: Distance = "marginal") -> float:
    """Score of a uniformly selected member of one complete forecast procedure.

    A and B must be independent executions with the same external inputs,
    same member count and same declared algorithm. Dependence within each
    bank is allowed. The code cannot verify this probabilistic assumption.
    """
    x, z = samples(a), samples(b)
    if x.shape != z.shape:
        raise ValueError("Use equal shapes for two executions of the same Q_M.")
    truth = _truth(y, x.shape[1])
    return float(0.5 * (distances(x, truth, kind).mean()
                        + distances(z, truth, kind).mean())
                 - 0.5 * distances(x, z, kind).mean())


def rbf(a: ArrayLike, b: ArrayLike,
        scales: Sequence[float] = DEFAULT_SCALES) -> FloatArray:
    x, z = samples(a), samples(b)
    s = np.asarray(scales, dtype=np.float64)
    if x.shape[1] != z.shape[1] or s.ndim != 1 or len(s) == 0:
        raise ValueError("Invalid dimensions or kernel scales.")
    if not np.isfinite(s).all() or (s <= 0).any():
        raise ValueError("All kernel scales must be finite and positive.")
    sq = cdist(x, z, metric="sqeuclidean")
    return sum(np.exp(-sq / (2 * scale**2)) for scale in s) / len(s)


def kernel_empirical_score(x: ArrayLike, y: ArrayLike,
                           scales: Sequence[float] = DEFAULT_SCALES) -> float:
    a = samples(x)
    return float(1.0 + rbf(a, a, scales).mean()
                 - 2 * rbf(a, _truth(y, a.shape[1]), scales).mean())


def kernel_u_score(x: ArrayLike, y: ArrayLike,
                   scales: Sequence[float] = DEFAULT_SCALES) -> float:
    a = samples(x)
    n = len(a)
    if n < 2:
        raise ValueError("At least two members required.")
    k = rbf(a, a, scales)
    return float(1.0 + (k.sum() - np.trace(k)) / (n * (n - 1))
                 - 2 * rbf(a, _truth(y, a.shape[1]), scales).mean())


def kernel_cross_score(a: ArrayLike, b: ArrayLike, y: ArrayLike,
                       scales: Sequence[float] = DEFAULT_SCALES) -> float:
    x, z = samples(a), samples(b)
    if x.shape != z.shape:
        raise ValueError("Independent banks must use identical shapes.")
    truth = _truth(y, x.shape[1])
    return float(1 + rbf(x, z, scales).mean()
                 - rbf(x, truth, scales).mean() - rbf(z, truth, scales).mean())


def mmd2_unbiased(a: ArrayLike, b: ArrayLike,
                  scales: Sequence[float] = DEFAULT_SCALES) -> float:
    x, z = samples(a), samples(b)
    n, m = len(x), len(z)
    if min(n, m) < 2:
        raise ValueError("Both samples need at least two iid observations.")
    xx, zz = rbf(x, x, scales), rbf(z, z, scales)
    return float((xx.sum() - np.trace(xx)) / (n * (n - 1))
                 + (zz.sum() - np.trace(zz)) / (m * (m - 1))
                 - 2 * rbf(x, z, scales).mean())


def v_triplets(x: ArrayLike, x_same_key: ArrayLike, z_new_key: ArrayLike,
               scales: Sequence[float] = DEFAULT_SCALES) -> FloatArray:
    """One unbiased estimate H_i of V per row; no truth labels are used."""
    a, ap, b = samples(x), samples(x_same_key), samples(z_new_key)
    if not (a.shape == ap.shape == b.shape):
        raise ValueError("Triplet arrays must have identical shapes.")
    s = np.asarray(scales, dtype=np.float64)
    if s.ndim != 1 or len(s) == 0 or (s <= 0).any() or not np.isfinite(s).all():
        raise ValueError("Invalid scales.")
    ds = ((a-ap)**2).sum(axis=1)
    dn = ((a-b)**2).sum(axis=1)
    return sum(np.exp(-ds/(2*t*t))-np.exp(-dn/(2*t*t)) for t in s) / len(s)


def hoeffding_radius(independent_units: int, delta: float = 0.05) -> float:
    if independent_units < 1 or not 0 < delta < 1:
        raise ValueError("Need n>=1 and 0<delta<1.")
    return float(np.sqrt(2 * np.log(2/delta) / independent_units))


def fresh_keys(keys: ArrayLike, rho: float, rng: np.random.Generator) -> FloatArray:
    a = samples(keys)
    if not 0 <= rho <= 1:
        raise ValueError("rho must be in [0,1].")
    return rho * a + np.sqrt(1-rho*rho) * rng.standard_normal(a.shape)


def key_weights(training_keys: ArrayLike, test_key: ArrayLike, rho: float) -> FloatArray:
    """Exact empirical weights WITHIN ONE repeated content category.

    rho=1 is deliberately rejected: for a new continuous key the empirical
    conditional measure is undefined, not a nearest-neighbour distribution.
    """
    a = samples(training_keys)
    k = _truth(test_key, a.shape[1]).ravel()
    if not 0 <= rho < 1:
        raise ValueError("Exact fresh-key density requires 0<=rho<1.")
    logw = -((k - rho*a)**2).sum(axis=1) / (2*(1-rho*rho))
    return np.exp(logw - logsumexp(logw))


def effective_sample_size(weights: ArrayLike) -> float:
    a = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = _weights(a, len(a))
    return float(1 / (w @ w))


def exact_cfm_field(z: ArrayLike, t: float, atoms: ArrayLike,
                    weights: ArrayLike) -> FloatArray:
    a, y = samples(z), samples(atoms)
    if a.shape[1] != y.shape[1] or not 0 <= t < 1:
        raise ValueError("Dimensions must match; exact atomic field requires t<1.")
    w = _weights(weights, len(y))
    logw = np.full_like(w, -np.inf)
    positive = w > 0
    logw[positive] = np.log(w[positive])
    logits = logw[None, :] - cdist(a, t*y, "sqeuclidean") / (2*(1-t)**2)
    resp = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    return (resp @ y - a) / (1-t)


def exact_path_samples(n: int, t: float, atoms: ArrayLike, weights: ArrayLike,
                       rng: np.random.Generator) -> FloatArray:
    y = samples(atoms)
    w = _weights(weights, len(y))
    if n < 1 or not 0 <= t <= 1:
        raise ValueError("Invalid n or t.")
    idx = rng.choice(len(y), size=n, p=w)
    return (1-t)*rng.standard_normal((n, y.shape[1])) + t*y[idx]


def _covariance(cov: ArrayLike, dim: int) -> FloatArray:
    s = np.asarray(cov, dtype=np.float64)
    if s.shape != (dim, dim) or not np.isfinite(s).all():
        raise ValueError("Invalid covariance shape or values.")
    if not np.allclose(s, s.T, rtol=1e-10, atol=1e-10):
        raise ValueError("Covariance must be symmetric.")
    s = (s+s.T)/2
    if np.linalg.eigvalsh(s).min() < -1e-9:
        raise ValueError("Covariance is not positive semidefinite.")
    return s


def gaussian_noise(cov: ArrayLike, n: int, rng: np.random.Generator) -> FloatArray:
    s0 = np.asarray(cov, dtype=np.float64)
    s = _covariance(s0, s0.shape[0])
    if n < 1:
        raise ValueError("Need at least one sample.")
    values, vectors = np.linalg.eigh(s)
    # Only roundoff-level negative eigenvalues passed validation above.
    root = vectors * np.sqrt(np.maximum(values, 0))[None, :]
    return rng.standard_normal((n, len(s))) @ root.T


def gaussian_rbf_expectation(mu_a: ArrayLike, cov_a: ArrayLike,
                             mu_b: ArrayLike, cov_b: ArrayLike,
                             scales: Sequence[float] = DEFAULT_SCALES) -> float:
    u = np.asarray(mu_a, dtype=np.float64).reshape(-1)
    v = np.asarray(mu_b, dtype=np.float64).reshape(-1)
    if u.shape != v.shape or not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError("Invalid means.")
    d = len(u)
    covariance = _covariance(cov_a, d) + _covariance(cov_b, d)
    difference = u-v
    vals = []
    for scale in scales:
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Kernel scales must be positive.")
        mat = np.eye(d) + covariance / scale**2
        sign, logdet = np.linalg.slogdet(mat)
        if sign <= 0:
            raise ArithmeticError("I+cov/scale^2 must be positive definite.")
        exponent = -.5 * difference @ np.linalg.solve(scale**2 * mat, difference)
        vals.append(np.exp(-.5*logdet + exponent))
    if not vals:
        raise ValueError("At least one scale required.")
    return float(np.mean(vals))


@dataclass
class GaussianMixture:
    weights: FloatArray
    means: FloatArray
    covariances: FloatArray

    def __post_init__(self) -> None:
        self.means = samples(self.means)
        self.weights = _weights(self.weights, len(self.means))
        covs = np.asarray(self.covariances, dtype=np.float64)
        k, d = self.means.shape
        if covs.shape != (k, d, d):
            raise ValueError("Covariances must have shape (components,d,d).")
        self.covariances = np.stack([_covariance(c, d) for c in covs])

    def moments(self) -> tuple[FloatArray, FloatArray]:
        mu = self.weights @ self.means
        offsets = self.means-mu
        cov = np.einsum("k,kij->ij", self.weights, self.covariances)
        cov += np.einsum("k,ki,kj->ij", self.weights, offsets, offsets)
        return mu, cov

    def sample(self, n: int, rng: np.random.Generator) -> FloatArray:
        if n < 1:
            raise ValueError("Need n>=1.")
        labels = rng.choice(len(self.weights), size=n, p=self.weights)
        ans = np.empty((n, self.means.shape[1]), dtype=np.float64)
        for j in range(len(self.weights)):
            rows = np.flatnonzero(labels == j)
            if len(rows):
                ans[rows] = self.means[j] + gaussian_noise(self.covariances[j], len(rows), rng)
        return ans

    def marginal_cdf(self, z: ArrayLike, coordinate: int) -> FloatArray:
        if not 0 <= coordinate < self.means.shape[1]:
            raise ValueError("Invalid coordinate.")
        x = np.asarray(z, dtype=np.float64)
        ans = np.zeros_like(x)
        for w, mu, cov in zip(self.weights, self.means, self.covariances):
            var = cov[coordinate, coordinate]
            if var == 0:
                ans += w * (x >= mu[coordinate])
            else:
                ans += w * ndtr((x-mu[coordinate])/np.sqrt(var))
        return ans

    def kernel_self_expectation(self, scales: Sequence[float] = DEFAULT_SCALES) -> float:
        total = 0.0
        for i, wi in enumerate(self.weights):
            for j, wj in enumerate(self.weights):
                total += wi*wj*gaussian_rbf_expectation(
                    self.means[i], self.covariances[i], self.means[j], self.covariances[j], scales)
        return float(total)

    def kernel_to_samples(self, x: ArrayLike,
                          scales: Sequence[float] = DEFAULT_SCALES) -> FloatArray:
        a = samples(x)
        if a.shape[1] != self.means.shape[1]:
            raise ValueError("Dimension mismatch.")
        scale_array = np.asarray(scales, dtype=np.float64)
        if (scale_array.ndim != 1 or len(scale_array) == 0
                or not np.isfinite(scale_array).all() or (scale_array <= 0).any()):
            raise ValueError("Kernel scales must be a nonempty finite positive sequence.")
        out = np.zeros(len(a))
        for scale in scale_array:
            for w, mu, cov in zip(self.weights, self.means, self.covariances):
                mat = scale**2*np.eye(a.shape[1]) + cov
                sign, logdet = np.linalg.slogdet(mat / scale**2)
                if sign <= 0:
                    raise ArithmeticError("Invalid kernel covariance.")
                diff = a-mu
                quad = np.einsum("ni,in->n", diff, np.linalg.solve(mat, diff.T))
                out += w*np.exp(-.5*(logdet+quad)) / len(scales)
        return out

    def mmd2_from_samples(self, x: ArrayLike,
                          scales: Sequence[float] = DEFAULT_SCALES) -> float:
        a = samples(x)
        n = len(a)
        if n < 2:
            raise ValueError("At least two iid forecast samples required.")
        aa = rbf(a, a, scales)
        return float((aa.sum()-np.trace(aa))/(n*(n-1))
                     - 2*self.kernel_to_samples(a, scales).mean()
                     + self.kernel_self_expectation(scales))


def category_teacher(c: int) -> GaussianMixture:
    """Exact bimodal 2-D oracle for category c in {0,...,15}."""
    if not isinstance(c, (int, np.integer)) or not 0 <= c < 16:
        raise ValueError("Category must be an integer in [0,15].")
    angle = 2*np.pi*c/16
    center = 2*np.array([np.cos(angle), np.sin(angle)])
    offset = .7*np.array([np.cos(2*angle), np.sin(2*angle)])
    pi = .25 + .5*(c % 4)/3
    rotation = np.array([[np.cos(angle/2), -np.sin(angle/2)],
                         [np.sin(angle/2), np.cos(angle/2)]])
    s1, s2 = .18+.12*(c % 4)/3, .12+.08*(c//4)/3
    covariance = rotation @ np.diag([s1*s1, s2*s2]) @ rotation.T
    return GaussianMixture(np.array([pi, 1-pi]),
                           np.stack([center+offset, center-offset]),
                           np.stack([covariance, covariance]))


def conditional_gaussian(mean: ArrayLike, covariance: ArrayLike,
                         h: ArrayLike, observations: ArrayLike,
                         noise_covariance: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    mu = np.asarray(mean, dtype=np.float64).reshape(-1)
    cov = _covariance(covariance, len(mu))
    hh = np.asarray(h, dtype=np.float64)
    obs = np.asarray(observations, dtype=np.float64).reshape(-1)
    if hh.ndim != 2 or hh.shape != (len(obs), len(mu)):
        raise ValueError("H must have shape (observations,state).")
    if not np.isfinite(mu).all() or not np.isfinite(hh).all() or not np.isfinite(obs).all():
        raise ValueError("Non-finite inputs.")
    noise = _covariance(noise_covariance, len(obs))
    innovation_cov = hh @ cov @ hh.T + noise
    factor = cho_factor(innovation_cov, lower=True, check_finite=True)
    gain = cho_solve(factor, hh @ cov, check_finite=True).T
    post_mean = mu + gain @ (obs-hh@mu)
    post_cov = cov - gain @ hh @ cov
    post_cov = _covariance((post_cov+post_cov.T)/2, len(mu))
    return post_mean, post_cov, gain


def matheron_samples(mean: ArrayLike, covariance: ArrayLike, h: ArrayLike,
                     observations: ArrayLike, noise_covariance: ArrayLike,
                     n: int, rng: np.random.Generator) -> FloatArray:
    mu = np.asarray(mean, dtype=np.float64).reshape(-1)
    cov = _covariance(covariance, len(mu))
    hh = np.asarray(h, dtype=np.float64)
    obs = np.asarray(observations, dtype=np.float64).reshape(-1)
    noise = _covariance(noise_covariance, len(obs))
    _, _, gain = conditional_gaussian(mu, cov, hh, obs, noise)
    prior = mu + gaussian_noise(cov, n, rng)
    eta = gaussian_noise(noise, n, rng)
    return prior + (obs[None, :] - prior @ hh.T - eta) @ gain.T
