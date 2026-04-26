from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

torch.set_float32_matmul_precision("high")

DEFAULT_SEED = 7
DEFAULT_NUM_EPOCHS = 60
DEFAULT_NUM_STEPS = 128
DEFAULT_LR = 1e-3
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_BATCH_SIZE_MPS = 4096
DEFAULT_BATCH_SIZE_CPU = 1024
DEFAULT_NUM_WORKERS_MPS = 4
DEFAULT_NUM_WORKERS_CPU = 2
DEFAULT_POSTERIOR_NOISE_STD = 0.15
DEFAULT_POSTERIOR_NUM_CASES = 16
DEFAULT_POSTERIOR_NUM_SAMPLES = 256
DEFAULT_OBSERVED_LOSS_WEIGHT = 5.0
FIXED_Y_MASK = np.array([0.0, 1.0, 0.0], dtype=np.float32)


@dataclass(frozen=True)
class Lorenz63Config:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    dt: float = 0.01
    burn_in_steps: int = 1000
    thin: int = 10


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_states(cls, states: np.ndarray) -> "Standardizer":
        mean = states.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = states.std(axis=0, dtype=np.float64).astype(np.float32)
        std = np.clip(std, 1e-6, None)
        return cls(mean=mean, std=std)

    def normalize_numpy(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def denormalize_numpy(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean

    def normalize_torch(self, values: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=values.device, dtype=values.dtype)
        std = torch.as_tensor(self.std, device=values.device, dtype=values.dtype)
        return (values - mean) / std

    def denormalize_torch(self, values: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=values.device, dtype=values.dtype)
        std = torch.as_tensor(self.std, device=values.device, dtype=values.dtype)
        return values * std + mean


@dataclass(frozen=True)
class GaussianPrior:
    mean: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray
    log_norm_const: float

    @classmethod
    def from_normalized_states(cls, states: np.ndarray, jitter: float = 1e-4) -> "GaussianPrior":
        mean = states.mean(axis=0, dtype=np.float64)
        covariance = np.cov(states, rowvar=False).astype(np.float64)
        covariance = covariance + jitter * np.eye(covariance.shape[0], dtype=np.float64)
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ValueError("covariance matrix must be positive definite")
        precision = np.linalg.inv(covariance)
        dim = covariance.shape[0]
        log_norm_const = -0.5 * (dim * math.log(2.0 * math.pi) + logdet)
        return cls(
            mean=mean.astype(np.float32),
            covariance=covariance.astype(np.float32),
            precision=precision.astype(np.float32),
            log_norm_const=float(log_norm_const),
        )

    def log_prob_numpy(self, samples: np.ndarray) -> np.ndarray:
        centered = samples.astype(np.float64) - self.mean.astype(np.float64)
        quad = np.einsum("...i,ij,...j->...", centered, self.precision.astype(np.float64), centered)
        return (self.log_norm_const - 0.5 * quad).astype(np.float64)


class StateDataset(Dataset[torch.Tensor]):
    def __init__(self, states: np.ndarray):
        self.states = torch.from_numpy(np.asarray(states, dtype=np.float32))

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.states[idx]


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even")
        half_dim = embedding_dim // 2
        frequencies = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                half_dim,
                dtype=torch.float32,
            )
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1)
        angles = t * self.frequencies.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class FlowMatchingMLP(nn.Module):
    def __init__(
        self,
        state_dim: int = 3,
        conditional: bool = False,
        hidden_dim: int = 256,
        hidden_layers: int = 4,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.conditional = conditional
        cond_dim = state_dim * 2 if conditional else 0
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)

        input_dim = state_dim + time_embed_dim + cond_dim
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None = None,
        obs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [x, self.time_embedding(t)]
        if self.conditional:
            if mask is None or obs is None:
                raise ValueError("conditional model requires mask and obs")
            parts.extend([mask, obs])
        features = torch.cat(parts, dim=-1)
        return self.network(features)


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_training_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    warnings.warn(
        "MPS is unavailable in the active Python session. Falling back to CPU.",
        stacklevel=2,
    )
    return torch.device("cpu")


def describe_runtime(device: torch.device) -> dict[str, Any]:
    device_type = device.type
    batch_size = DEFAULT_BATCH_SIZE_MPS if device_type == "mps" else DEFAULT_BATCH_SIZE_CPU
    num_workers = DEFAULT_NUM_WORKERS_MPS if device_type == "mps" else DEFAULT_NUM_WORKERS_CPU
    return {
        "device": device_type,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device_type == "cuda",
        "persistent_workers": num_workers > 0,
    }


def load_split_states(data_dir: str | Path, split: str) -> np.ndarray:
    path = Path(data_dir) / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing split file: {path}")
    with np.load(path) as payload:
        states = payload["states"].astype(np.float32)
    return states


def make_state_loader(
    states: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device_type: str,
) -> DataLoader:
    return DataLoader(
        StateDataset(states),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device_type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def lorenz63_rhs(state: np.ndarray, sigma: float, rho: float, beta: float) -> np.ndarray:
    x = state[..., 0]
    y = state[..., 1]
    z = state[..., 2]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.stack([dx, dy, dz], axis=-1)


def rk4_step_numpy(state: np.ndarray, dt: float, sigma: float, rho: float, beta: float) -> np.ndarray:
    k1 = lorenz63_rhs(state, sigma=sigma, rho=rho, beta=beta)
    k2 = lorenz63_rhs(state + 0.5 * dt * k1, sigma=sigma, rho=rho, beta=beta)
    k3 = lorenz63_rhs(state + 0.5 * dt * k2, sigma=sigma, rho=rho, beta=beta)
    k4 = lorenz63_rhs(state + dt * k3, sigma=sigma, rho=rho, beta=beta)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def sample_sparse_observations_torch(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, state_dim = states.shape
    perm = torch.argsort(torch.rand(batch_size, state_dim, device=states.device), dim=1)
    ranks = torch.empty_like(perm)
    order = torch.arange(state_dim, device=states.device).expand(batch_size, -1)
    ranks.scatter_(1, perm, order)
    counts = torch.randint(1, 3, (batch_size,), device=states.device)
    mask = (ranks < counts.unsqueeze(1)).to(states.dtype)
    obs = states * mask
    return mask, obs


def sample_fixed_y_observations_torch(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.zeros_like(states)
    mask[:, 1] = 1.0
    obs = states * mask
    return mask, obs


def make_fixed_conditioning(states: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    batch_size, state_dim = states.shape
    scores = rng.random((batch_size, state_dim))
    perm = np.argsort(scores, axis=1)
    ranks = np.empty_like(perm)
    ranks[np.arange(batch_size)[:, None], perm] = np.arange(state_dim)[None, :]
    counts = rng.integers(1, 3, size=batch_size)
    mask = (ranks < counts[:, None]).astype(np.float32)
    obs = states.astype(np.float32) * mask
    return mask, obs


def make_fixed_y_conditioning(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=np.float32)
    mask = np.broadcast_to(FIXED_Y_MASK, states.shape).copy().astype(np.float32)
    obs = states * mask
    return mask, obs


def make_noisy_observation_case(
    truth_state_raw: np.ndarray,
    standardizer: Standardizer,
    seed: int,
    obs_noise_std: float = DEFAULT_POSTERIOR_NOISE_STD,
    mask: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    if obs_noise_std < 0.0:
        raise ValueError("obs_noise_std must be non-negative")
    truth_state_raw = np.asarray(truth_state_raw, dtype=np.float32)
    truth_state_norm = standardizer.normalize_numpy(truth_state_raw[None, :])[0].astype(np.float32)
    if mask is None:
        sampled_mask, _ = make_fixed_conditioning(truth_state_raw[None, :], seed=seed)
        mask = sampled_mask[0]
    else:
        mask = np.asarray(mask, dtype=np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=obs_noise_std, size=truth_state_norm.shape).astype(np.float32)
    obs_norm = truth_state_norm * mask + noise * mask
    obs_raw = np.zeros_like(truth_state_raw, dtype=np.float32)
    observed = mask > 0.5
    obs_raw[observed] = truth_state_raw[observed] + noise[observed] * standardizer.std[observed]
    return {
        "truth_raw": truth_state_raw,
        "truth_norm": truth_state_norm,
        "mask": mask.astype(np.float32),
        "obs_norm": obs_norm.astype(np.float32),
        "obs_raw": obs_raw.astype(np.float32),
        "obs_noise_std": float(obs_noise_std),
    }


def make_flow_matching_batch(x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
    x0 = torch.randn_like(x1)
    xt = (1.0 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * x1
    target_velocity = x1 - x0
    return xt, t, target_velocity


def _loss_on_batch(
    model: FlowMatchingMLP,
    states_raw: torch.Tensor,
    standardizer: Standardizer,
    conditional: bool,
    conditioning_mode: str = "random",
    observed_loss_weight: float = 1.0,
) -> torch.Tensor:
    if observed_loss_weight < 1.0:
        raise ValueError("observed_loss_weight must be at least 1.0")
    states = standardizer.normalize_torch(states_raw)
    xt, t, target_velocity = make_flow_matching_batch(states)
    mask = None
    if conditional:
        if conditioning_mode == "random":
            mask, obs = sample_sparse_observations_torch(states)
        elif conditioning_mode == "fixed_y":
            mask, obs = sample_fixed_y_observations_torch(states)
        else:
            raise ValueError(f"unknown conditioning_mode: {conditioning_mode}")
        pred_velocity = model(xt, t, mask=mask, obs=obs)
    else:
        pred_velocity = model(xt, t)
    sq_error = (pred_velocity - target_velocity) ** 2
    if conditional and observed_loss_weight > 1.0:
        weights = 1.0 + (observed_loss_weight - 1.0) * mask
        sq_error = sq_error * weights
    return torch.mean(sq_error)


def fit_flow_matching(
    model: FlowMatchingMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    standardizer: Standardizer,
    device: torch.device,
    conditional: bool,
    epochs: int = DEFAULT_NUM_EPOCHS,
    lr: float = DEFAULT_LR,
    weight_decay: float = 1e-4,
    conditioning_mode: str = "random",
    observed_loss_weight: float = 1.0,
) -> dict[str, list[float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(
            train_loader,
            desc=f"train epoch {epoch + 1}/{epochs}",
            leave=False,
        )
        for batch in train_bar:
            states_raw = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_on_batch(
                model,
                states_raw,
                standardizer,
                conditional=conditional,
                conditioning_mode=conditioning_mode,
                observed_loss_weight=observed_loss_weight,
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * states_raw.shape[0]
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_epoch_loss = train_loss / len(train_loader.dataset)
        history["train_loss"].append(train_epoch_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(
                val_loader,
                desc=f"val epoch {epoch + 1}/{epochs}",
                leave=False,
            )
            for batch in val_bar:
                states_raw = batch.to(device)
                loss = _loss_on_batch(
                    model,
                    states_raw,
                    standardizer,
                    conditional=conditional,
                    conditioning_mode=conditioning_mode,
                    observed_loss_weight=observed_loss_weight,
                )
                val_loss += loss.item() * states_raw.shape[0]
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        val_epoch_loss = val_loss / len(val_loader.dataset)
        history["val_loss"].append(val_epoch_loss)
        print(
            f"epoch {epoch + 1:03d} | "
            f"train_loss={train_epoch_loss:.6f} | "
            f"val_loss={val_epoch_loss:.6f}"
        )

    return history


def _sample_conditioned_posterior(
    model: FlowMatchingMLP,
    mask: np.ndarray,
    obs_norm: np.ndarray,
    standardizer: Standardizer,
    device: torch.device,
    num_samples: int,
    num_steps: int,
    conditional: bool,
    guidance_scale: float,
    seed: int,
    desc: str,
) -> dict[str, np.ndarray]:
    set_seed(seed)
    model.eval()
    dtype = torch.float32
    mask_t = torch.from_numpy(np.broadcast_to(mask, (num_samples, mask.shape[0])).copy()).to(device=device, dtype=dtype)
    obs_t = torch.from_numpy(np.broadcast_to(obs_norm, (num_samples, obs_norm.shape[0])).copy()).to(device=device, dtype=dtype)
    x_init = torch.randn((num_samples, mask.shape[0]), device=device, dtype=dtype)

    if conditional:
        velocity_fn = lambda x, t: model(x, t, mask=mask_t, obs=obs_t)
    else:
        velocity_fn = lambda x, t: _gradient_guided_velocity(
            model,
            x,
            t,
            mask=mask_t,
            obs=obs_t,
            guidance_scale=guidance_scale,
        )

    with torch.no_grad() if conditional else torch.enable_grad():
        samples_norm_t = _heun_integrate(
            velocity_fn=velocity_fn,
            x_init=x_init,
            num_steps=num_steps,
            desc=desc,
        )
    samples_norm = samples_norm_t.detach().cpu().numpy().astype(np.float32)
    samples_raw = standardizer.denormalize_numpy(samples_norm).astype(np.float32)
    return {"samples_norm": samples_norm, "samples_raw": samples_raw}


def _heun_integrate(
    velocity_fn,
    x_init: torch.Tensor,
    num_steps: int,
    desc: str,
) -> torch.Tensor:
    x = x_init
    times = torch.linspace(0.0, 1.0, num_steps + 1, device=x.device, dtype=x.dtype)
    step_bar = tqdm(range(num_steps), desc=desc, leave=False)
    for idx in step_bar:
        t0 = times[idx]
        t1 = times[idx + 1]
        dt = t1 - t0
        t0_batch = torch.full((x.shape[0],), t0.item(), device=x.device, dtype=x.dtype)
        k1 = velocity_fn(x, t0_batch)
        x_euler = x + dt * k1
        t1_batch = torch.full((x.shape[0],), t1.item(), device=x.device, dtype=x.dtype)
        k2 = velocity_fn(x_euler, t1_batch)
        x = x + 0.5 * dt * (k1 + k2)
    return x


def _gradient_guided_velocity(
    model: FlowMatchingMLP,
    x: torch.Tensor,
    t: torch.Tensor,
    mask: torch.Tensor,
    obs: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    x_req = x.detach().requires_grad_(True)
    with torch.enable_grad():
        velocity = model(x_req, t)
        terminal_estimate = x_req + (1.0 - t.unsqueeze(1)) * velocity
        masked_error = (terminal_estimate - obs) * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        loss = (masked_error.square().sum(dim=1) / denom).mean()
        guidance = torch.autograd.grad(loss, x_req)[0]
    return velocity.detach() - guidance_scale * guidance.detach()


def _collect_metrics(
    predictions: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    error = predictions - truth
    sq = np.square(error)
    rmse = float(np.sqrt(sq.mean()))
    observed_sq = float((sq * mask).sum())
    observed_count = float(mask.sum())
    unobserved_mask = 1.0 - mask
    unobserved_sq = float((sq * unobserved_mask).sum())
    unobserved_count = float(unobserved_mask.sum())
    return {
        "rmse": rmse,
        "observed_rmse": float(np.sqrt(observed_sq / max(observed_count, 1.0))),
        "unobserved_rmse": float(np.sqrt(unobserved_sq / max(unobserved_count, 1.0))),
    }


def _observation_log_likelihood_numpy(
    samples_norm: np.ndarray,
    mask: np.ndarray,
    obs_norm: np.ndarray,
    obs_noise_std: float,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float64)
    obs_norm = np.asarray(obs_norm, dtype=np.float64)
    samples_norm = np.asarray(samples_norm, dtype=np.float64)
    obs_dims = int(mask.sum())
    if obs_dims <= 0:
        return np.zeros(samples_norm.shape[0], dtype=np.float64)
    residual = (samples_norm - obs_norm) * mask[None, :]
    if obs_noise_std <= 0.0:
        exact = np.linalg.norm(residual, axis=1) <= 1e-6
        return np.where(exact, 0.0, -np.inf)
    quad = np.sum(np.square(residual), axis=1) / (obs_noise_std ** 2)
    log_norm = obs_dims * math.log(2.0 * math.pi * (obs_noise_std ** 2))
    return -0.5 * (quad + log_norm)


def sample_reference_posterior(
    prior_states: np.ndarray,
    standardizer: Standardizer,
    mask: np.ndarray,
    obs_norm: np.ndarray,
    obs_noise_std: float = DEFAULT_POSTERIOR_NOISE_STD,
    num_samples: int = DEFAULT_POSTERIOR_NUM_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, np.ndarray]:
    prior_states = np.asarray(prior_states, dtype=np.float32)
    prior_norm = standardizer.normalize_numpy(prior_states).astype(np.float32)
    obs_dims = int(np.asarray(mask, dtype=np.float32).sum())
    rng = np.random.default_rng(seed)
    if obs_noise_std < 0.0:
        raise ValueError("obs_noise_std must be non-negative")
    if obs_noise_std == 0.0 and obs_dims > 0:
        observed = np.asarray(mask, dtype=np.float32) > 0.5
        residual = prior_norm[:, observed] - np.asarray(obs_norm, dtype=np.float32)[observed][None, :]
        distance_sq = np.sum(np.square(residual), axis=1)
        top_k = min(num_samples, len(prior_states))
        top_indices = np.argpartition(distance_sq, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(distance_sq[top_indices])]
        if top_k < num_samples:
            indices = rng.choice(top_indices, size=num_samples, replace=True)
        else:
            indices = top_indices
        weights = np.zeros(len(prior_states), dtype=np.float32)
        weights[top_indices] = 1.0 / float(top_k)
        return {
            "samples_raw": prior_states[indices].astype(np.float32),
            "samples_norm": prior_norm[indices].astype(np.float32),
            "weights": weights,
        }
    log_weights = _observation_log_likelihood_numpy(
        samples_norm=prior_norm,
        mask=mask,
        obs_norm=obs_norm,
        obs_noise_std=obs_noise_std,
    )
    log_weights = log_weights - np.max(log_weights)
    weights = np.exp(log_weights)
    weights = weights / np.sum(weights)
    indices = rng.choice(len(prior_states), size=num_samples, replace=True, p=weights)
    return {
        "samples_raw": prior_states[indices].astype(np.float32),
        "samples_norm": prior_norm[indices].astype(np.float32),
        "weights": weights.astype(np.float32),
    }


def compute_wasserstein_distance(
    reference_samples: np.ndarray,
    approx_samples: np.ndarray,
) -> float:
    if reference_samples.shape != approx_samples.shape:
        raise ValueError("reference_samples and approx_samples must have the same shape")
    cost = cdist(reference_samples, approx_samples, metric="euclidean")
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].mean())


def summarize_posterior_samples(
    samples_raw: np.ndarray,
    samples_norm: np.ndarray,
    gaussian_prior: GaussianPrior,
    mask: np.ndarray,
    obs_norm: np.ndarray,
    obs_noise_std: float,
) -> dict[str, float]:
    log_prior = gaussian_prior.log_prob_numpy(samples_norm)
    log_likelihood = _observation_log_likelihood_numpy(
        samples_norm=samples_norm,
        mask=mask,
        obs_norm=obs_norm,
        obs_noise_std=obs_noise_std,
    )
    observed = int(np.asarray(mask, dtype=np.float32).sum())
    if observed > 0:
        observed_residual = (samples_norm - obs_norm) * mask[None, :]
        observed_rmse_norm = float(np.sqrt(np.sum(np.square(observed_residual)) / (samples_norm.shape[0] * observed)))
    else:
        observed_rmse_norm = 0.0
    expected_log_likelihood = float(np.mean(log_likelihood)) if obs_noise_std > 0.0 else float("nan")
    return {
        "expected_log_prior": float(np.mean(log_prior)),
        "expected_log_likelihood": expected_log_likelihood,
        "observed_rmse_norm": observed_rmse_norm,
        "sample_mean_norm": float(np.linalg.norm(np.mean(samples_raw, axis=0))),
    }


def _evaluate_batches(
    model: FlowMatchingMLP,
    states: np.ndarray,
    mask: np.ndarray,
    standardizer: Standardizer,
    device: torch.device,
    batch_size: int,
    num_steps: int,
    desc: str,
    conditional: bool,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    set_seed(seed)
    model.eval()
    predictions: list[np.ndarray] = []
    batch_bar = tqdm(range(0, len(states), batch_size), desc=desc)
    for start in batch_bar:
        stop = min(start + batch_size, len(states))
        truth_raw = torch.from_numpy(states[start:stop]).to(device=device, dtype=torch.float32)
        mask_batch = torch.from_numpy(mask[start:stop]).to(device=device, dtype=torch.float32)
        truth_norm = standardizer.normalize_torch(truth_raw)
        obs_norm = truth_norm * mask_batch
        x_init = torch.randn_like(truth_norm)

        if conditional:
            velocity_fn = lambda x, t: model(x, t, mask=mask_batch, obs=obs_norm)
        else:
            velocity_fn = lambda x, t: _gradient_guided_velocity(
                model,
                x,
                t,
                mask=mask_batch,
                obs=obs_norm,
                guidance_scale=guidance_scale,
            )

        with torch.no_grad() if conditional else torch.enable_grad():
            pred_norm = _heun_integrate(
                velocity_fn=velocity_fn,
                x_init=x_init,
                num_steps=num_steps,
                desc=f"{desc} steps {start}:{stop}",
            )
        pred_raw = standardizer.denormalize_torch(pred_norm).detach().cpu().numpy()
        predictions.append(pred_raw.astype(np.float32))

    prediction_array = np.concatenate(predictions, axis=0)
    metrics = _collect_metrics(prediction_array, states, mask)
    return {
        "metrics": metrics,
        "predictions": prediction_array,
    }


def evaluate_gradient_assimilation(
    model: FlowMatchingMLP,
    states: np.ndarray,
    mask: np.ndarray,
    standardizer: Standardizer,
    device: torch.device,
    batch_size: int,
    num_steps: int = DEFAULT_NUM_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    return _evaluate_batches(
        model=model,
        states=states,
        mask=mask,
        standardizer=standardizer,
        device=device,
        batch_size=batch_size,
        num_steps=num_steps,
        desc="gradient assimilation",
        conditional=False,
        guidance_scale=guidance_scale,
        seed=seed,
    )


def evaluate_concat_assimilation(
    model: FlowMatchingMLP,
    states: np.ndarray,
    mask: np.ndarray,
    standardizer: Standardizer,
    device: torch.device,
    batch_size: int,
    num_steps: int = DEFAULT_NUM_STEPS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    return _evaluate_batches(
        model=model,
        states=states,
        mask=mask,
        standardizer=standardizer,
        device=device,
        batch_size=batch_size,
        num_steps=num_steps,
        desc="concat assimilation",
        conditional=True,
        seed=seed,
    )


def evaluate_mean_fill_baseline(
    states: np.ndarray,
    mask: np.ndarray,
    standardizer: Standardizer,
) -> dict[str, Any]:
    mean = standardizer.mean.astype(np.float32)
    predictions = states * mask + (1.0 - mask) * mean[None, :]
    return {
        "metrics": _collect_metrics(predictions, states, mask),
        "predictions": predictions.astype(np.float32),
    }


def evaluate_posterior_restoration(
    model: FlowMatchingMLP,
    train_states: np.ndarray,
    test_states: np.ndarray,
    standardizer: Standardizer,
    device: torch.device,
    conditional: bool,
    obs_noise_std: float = DEFAULT_POSTERIOR_NOISE_STD,
    num_cases: int = DEFAULT_POSTERIOR_NUM_CASES,
    num_posterior_samples: int = DEFAULT_POSTERIOR_NUM_SAMPLES,
    num_steps: int = DEFAULT_NUM_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    reference_bank_size: int = 50_000,
    seed: int = DEFAULT_SEED,
    conditioning_mode: str = "random",
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    train_states = np.asarray(train_states, dtype=np.float32)
    test_states = np.asarray(test_states, dtype=np.float32)
    train_norm = standardizer.normalize_numpy(train_states).astype(np.float32)
    gaussian_prior = GaussianPrior.from_normalized_states(train_norm)

    if reference_bank_size < len(train_states):
        bank_indices = rng.choice(len(train_states), size=reference_bank_size, replace=False)
        reference_bank = train_states[bank_indices]
    else:
        reference_bank = train_states

    case_indices = rng.choice(len(test_states), size=min(num_cases, len(test_states)), replace=False)
    details: list[dict[str, Any]] = []
    progress = tqdm(case_indices, desc="posterior restoration")
    for case_id, state_idx in enumerate(progress):
        if conditioning_mode == "random":
            mask = None
        elif conditioning_mode == "fixed_y":
            mask = FIXED_Y_MASK
        else:
            raise ValueError(f"unknown conditioning_mode: {conditioning_mode}")
        case = make_noisy_observation_case(
            truth_state_raw=test_states[state_idx],
            standardizer=standardizer,
            seed=seed + case_id,
            obs_noise_std=obs_noise_std,
            mask=mask,
        )
        reference = sample_reference_posterior(
            prior_states=reference_bank,
            standardizer=standardizer,
            mask=case["mask"],  # type: ignore[arg-type]
            obs_norm=case["obs_norm"],  # type: ignore[arg-type]
            obs_noise_std=obs_noise_std,
            num_samples=num_posterior_samples,
            seed=seed + 10_000 + case_id,
        )
        approx = _sample_conditioned_posterior(
            model=model,
            mask=case["mask"],  # type: ignore[arg-type]
            obs_norm=case["obs_norm"],  # type: ignore[arg-type]
            standardizer=standardizer,
            device=device,
            num_samples=num_posterior_samples,
            num_steps=num_steps,
            conditional=conditional,
            guidance_scale=guidance_scale,
            seed=seed + 20_000 + case_id,
            desc=f"posterior case {case_id + 1}/{len(case_indices)}",
        )
        ref_summary = summarize_posterior_samples(
            samples_raw=reference["samples_raw"],
            samples_norm=reference["samples_norm"],
            gaussian_prior=gaussian_prior,
            mask=case["mask"],  # type: ignore[arg-type]
            obs_norm=case["obs_norm"],  # type: ignore[arg-type]
            obs_noise_std=obs_noise_std,
        )
        approx_summary = summarize_posterior_samples(
            samples_raw=approx["samples_raw"],
            samples_norm=approx["samples_norm"],
            gaussian_prior=gaussian_prior,
            mask=case["mask"],  # type: ignore[arg-type]
            obs_norm=case["obs_norm"],  # type: ignore[arg-type]
            obs_noise_std=obs_noise_std,
        )
        wasserstein = compute_wasserstein_distance(reference["samples_raw"], approx["samples_raw"])
        details.append(
            {
                "state_index": int(state_idx),
                "truth_raw": np.asarray(case["truth_raw"], dtype=np.float32),
                "mask": np.asarray(case["mask"], dtype=np.float32),
                "obs_raw": np.asarray(case["obs_raw"], dtype=np.float32),
                "reference_summary": ref_summary,
                "approx_summary": approx_summary,
                "wasserstein": float(wasserstein),
                "reference_samples_raw": reference["samples_raw"],
                "approx_samples_raw": approx["samples_raw"],
            }
        )

    summary = {
        "reference_expected_log_prior": float(np.mean([d["reference_summary"]["expected_log_prior"] for d in details])),
        "reference_expected_log_likelihood": float(np.mean([d["reference_summary"]["expected_log_likelihood"] for d in details])),
        "reference_observed_rmse_norm": float(np.mean([d["reference_summary"]["observed_rmse_norm"] for d in details])),
        "approx_expected_log_prior": float(np.mean([d["approx_summary"]["expected_log_prior"] for d in details])),
        "approx_expected_log_likelihood": float(np.mean([d["approx_summary"]["expected_log_likelihood"] for d in details])),
        "approx_observed_rmse_norm": float(np.mean([d["approx_summary"]["observed_rmse_norm"] for d in details])),
        "wasserstein": float(np.mean([d["wasserstein"] for d in details])),
    }
    return {
        "summary": summary,
        "details": details,
        "obs_noise_std": obs_noise_std,
        "num_cases": len(details),
        "num_posterior_samples": num_posterior_samples,
    }


def plot_loss_curves(history: dict[str, list[float]], title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_attractor_comparison(truth: np.ndarray, predictions: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 5))
    ax_true = fig.add_subplot(121, projection="3d")
    ax_pred = fig.add_subplot(122, projection="3d")

    sample = slice(None, min(len(truth), 5000))
    ax_true.scatter(truth[sample, 0], truth[sample, 1], truth[sample, 2], s=2, alpha=0.35)
    ax_true.set_title("Truth attractor")
    ax_true.set_xlabel("x")
    ax_true.set_ylabel("y")
    ax_true.set_zlabel("z")

    ax_pred.scatter(
        predictions[sample, 0],
        predictions[sample, 1],
        predictions[sample, 2],
        s=2,
        alpha=0.35,
        color="tab:orange",
    )
    ax_pred.set_title("Reconstructed attractor")
    ax_pred.set_xlabel("x")
    ax_pred.set_ylabel("y")
    ax_pred.set_zlabel("z")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_reconstruction_examples(
    truth: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    title: str,
    max_examples: int = 6,
) -> None:
    import matplotlib.pyplot as plt

    indices = np.linspace(0, len(truth) - 1, num=min(max_examples, len(truth)), dtype=int)
    fig, axes = plt.subplots(len(indices), 1, figsize=(9, 2.2 * len(indices)), squeeze=False)
    coords = np.arange(truth.shape[1])
    for row, idx in enumerate(indices):
        ax = axes[row, 0]
        ax.plot(coords, truth[idx], marker="o", label="truth")
        ax.plot(coords, predictions[idx], marker="s", label="prediction")
        observed = coords[mask[idx].astype(bool)]
        ax.scatter(observed, truth[idx, observed], color="black", s=80, label="observed")
        ax.set_xticks(coords)
        ax.set_xticklabels(["x", "y", "z"])
        ax.set_ylabel(f"sample {idx}")
        ax.grid(True, alpha=0.2)
        if row == 0:
            ax.legend(loc="best")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_posterior_case(
    reference_samples: np.ndarray,
    approx_samples: np.ndarray,
    truth_state: np.ndarray,
    mask: np.ndarray,
    obs_raw: np.ndarray,
    title: str,
    max_points: int = 500,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    labels = ["x", "y", "z"]
    observed_names = [labels[idx] for idx, value in enumerate(mask) if value > 0.5]
    observed_text = ", ".join(observed_names) if observed_names else "none"

    rng = np.random.default_rng(0)
    ref_plot = reference_samples
    approx_plot = approx_samples
    if len(ref_plot) > max_points:
        ref_plot = ref_plot[rng.choice(len(ref_plot), size=max_points, replace=False)]
    if len(approx_plot) > max_points:
        approx_plot = approx_plot[rng.choice(len(approx_plot), size=max_points, replace=False)]

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.5), squeeze=False)
    ref_color = "tab:blue"
    model_color = "tab:orange"
    truth_color = "black"
    obs_color = "tab:red"

    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            if row < col:
                ax.axis("off")
                continue

            if row == col:
                values = np.concatenate([reference_samples[:, col], approx_samples[:, col]])
                low, high = np.percentile(values, [0.5, 99.5])
                if np.isclose(low, high):
                    low, high = values.min(), values.max()
                bins = np.linspace(low, high, 35)
                ax.hist(
                    reference_samples[:, col],
                    bins=bins,
                    density=True,
                    alpha=0.35,
                    color=ref_color,
                    label="reference",
                )
                ax.hist(
                    approx_samples[:, col],
                    bins=bins,
                    density=True,
                    alpha=0.35,
                    color=model_color,
                    label="model",
                )
                ax.axvline(truth_state[col], color=truth_color, linestyle="-", linewidth=1.6)
                if mask[col] > 0.5:
                    ax.axvline(obs_raw[col], color=obs_color, linestyle="--", linewidth=1.8)
                ax.set_title(f"{labels[col]} marginal")
                ax.set_ylabel("density")
            else:
                ax.scatter(
                    ref_plot[:, col],
                    ref_plot[:, row],
                    s=12,
                    alpha=0.22,
                    color=ref_color,
                    edgecolors="none",
                )
                ax.scatter(
                    approx_plot[:, col],
                    approx_plot[:, row],
                    s=12,
                    alpha=0.28,
                    color=model_color,
                    edgecolors="none",
                )
                ax.scatter(
                    [truth_state[col]],
                    [truth_state[row]],
                    color=truth_color,
                    s=70,
                    marker="x",
                    linewidths=2.0,
                )
                if mask[col] > 0.5:
                    ax.axvline(obs_raw[col], color=obs_color, linestyle="--", linewidth=1.4, alpha=0.9)
                if mask[row] > 0.5:
                    ax.axhline(obs_raw[row], color=obs_color, linestyle="--", linewidth=1.4, alpha=0.9)
                ax.set_ylabel(labels[row])

            if row == 2:
                ax.set_xlabel(labels[col])
            ax.grid(True, alpha=0.2)

    legend_handles = [
        Patch(facecolor=ref_color, alpha=0.35, label="reference posterior"),
        Patch(facecolor=model_color, alpha=0.35, label="model posterior"),
        Line2D([0], [0], color=truth_color, marker="x", linestyle="None", markersize=8, label="truth"),
        Line2D([0], [0], color=obs_color, linestyle="--", linewidth=1.8, label="observed coordinate"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=True)
    fig.suptitle(f"{title}\nobserved: {observed_text}", y=0.98)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    plt.show()
