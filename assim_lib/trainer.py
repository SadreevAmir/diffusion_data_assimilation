from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import __version__ as diffusers_version
from diffusers.training_utils import EMAModel
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from tqdm.auto import tqdm

from .clearml_tracking import ClearMLTracker
from .config import TrainingConfig, jsonable
from .dashboard import make_multi_case_background_condition_assim_figure
from .flow_parameterization import reconstruct_velocity, velocity_model_state
from .runtime import make_normalized_xy_grid, preserve_persistent_worker_rng
from .sampler import Sampler
from .structured_joint_state import (
    StructuredDecodeSaturationError,
    decode_structured_joint_trajectory,
    encode_structured_joint_trajectory,
)
from .structured_trajectory_evaluation import (
    make_structured_trajectory_figure,
    sample_structured_batch,
    structured_trajectory_metrics,
)
from .transforms import channel_denormalize, make_conditioned_model_input

logger = get_logger(__name__)


_DASHBOARD_EVERY_N_EPOCHS = 5
_UNCONDITIONAL_DASHBOARD_EVERY_N_EPOCHS = 1
_DASHBOARD_NUM_CASES = 12
_DASHBOARD_HISTORY_EPOCHS = 1
_DASHBOARD_CASES_PER_PAGE = 12
_DASHBOARD_CHANNELS = (0,)
_DASHBOARD_DPI = 200
_DASHBOARD_PANEL_WIDTH = 3.6
_DASHBOARD_PANEL_HEIGHT = 2.5
_DASHBOARD_LARGE_PANEL_WIDTH = 5.5
_DASHBOARD_LARGE_PANEL_HEIGHT = 4.4
_UNCONDITIONAL_PANEL_WIDTH = 7.0
_UNCONDITIONAL_PANEL_HEIGHT = 5.6


def _debug(message: str) -> None:
    print(f"[assim_lib][trainer] {message}", flush=True)


def _latest_structured_sampling_valid(records: list[dict]) -> bool:
    return bool(records and records[-1].get("status") == "passed")


_STRUCTURED_RESUME_SCHEMA = "structured_joint_resume_v1"
_ACTIVATION_CHECKPOINTING_DIFFUSERS_VERSION = "0.36.0"


def _non_reentrant_checkpoint(module, *args):
    """Checkpoint one native diffusers block without changing its RNG semantics."""
    return torch_checkpoint(
        module.__call__,
        *args,
        use_reentrant=False,
        preserve_rng_state=True,
    )


def configure_activation_checkpointing(model, enabled: bool) -> tuple[str, ...]:
    """Enable the single audited activation-checkpointing implementation."""
    if not enabled:
        return ()
    if diffusers_version != _ACTIVATION_CHECKPOINTING_DIFFUSERS_VERSION:
        raise RuntimeError(
            "activation checkpointing requires the audited "
            f"diffusers=={_ACTIVATION_CHECKPOINTING_DIFFUSERS_VERSION}, "
            f"got {diffusers_version}"
        )
    enable = getattr(model, "enable_gradient_checkpointing", None)
    if not callable(enable) or not bool(getattr(model, "_supports_gradient_checkpointing", False)):
        raise RuntimeError("model does not support native diffusers activation checkpointing")
    expected_modules = tuple(
        name for name, module in model.named_modules() if hasattr(module, "gradient_checkpointing")
    )
    if not expected_modules:
        raise RuntimeError("model exposes no native checkpointable modules")
    enable(gradient_checkpointing_func=_non_reentrant_checkpoint)
    modules = tuple(
        name
        for name, module in model.named_modules()
        if bool(getattr(module, "gradient_checkpointing", False))
    )
    if not bool(getattr(model, "is_gradient_checkpointing", False)) or modules != expected_modules:
        raise RuntimeError(
            "activation checkpointing was requested but did not activate on every "
            f"native block: expected={expected_modules}, active={modules}"
        )
    return modules


def _canonical_sha256(payload) -> str:
    encoded = json.dumps(jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_payload_manifest(root: Path) -> dict[str, str]:
    manifest = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"checkpoint payload cannot contain symlinks: {path}")
        if path.is_file() and path.name != "resume.json":
            manifest[path.relative_to(root).as_posix()] = _file_sha256(path)
    if not manifest:
        raise ValueError("checkpoint payload is empty")
    return manifest


def _require_finite_loss(loss: torch.Tensor, provenance: str) -> None:
    if loss.numel() != 1 or not bool(torch.isfinite(loss).all()):
        raise FloatingPointError(f"non-finite scalar training loss; {provenance}")


def _require_finite_gradients(model, provenance: str) -> None:
    bad = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            bad.append(name)
            if len(bad) >= 8:
                break
    if bad:
        raise FloatingPointError(f"non-finite gradients before optimizer step in {bad}; {provenance}")


def _validate_existing_structured_run_files(
    output_dir: Path,
    config_payload: dict,
    contract_sha256: str,
) -> bool:
    """Validate existing provenance before the constructor writes anything."""
    config_path = output_dir / "config.json"
    metadata_path = output_dir / "metadata.json"
    existing = (
        config_path.exists() or config_path.is_symlink(),
        metadata_path.exists() or metadata_path.is_symlink(),
    )
    if not any(existing):
        return False
    if not all(existing):
        raise ValueError("structured run has only one of config.json/metadata.json")
    if any(path.is_symlink() or not path.is_file() for path in (config_path, metadata_path)):
        raise ValueError("structured run provenance files must be ordinary files")
    existing_config = json.loads(config_path.read_text(encoding="utf-8"))
    existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if existing_config != jsonable(config_payload):
        raise ValueError("existing structured run config identity differs")
    if existing_metadata.get("resume_contract_sha256") != contract_sha256:
        raise ValueError("existing structured run metadata identity differs")
    return True


class UNetTrainer:
    def __init__(
        self,
        config: TrainingConfig,
        model,
        optimizer,
        data_loader_train,
        data_loader_val,
        lr_scheduler,
        add_noise_func,
        experiment_config=None,
        model_config=None,
        data_config=None,
        dataset_provenance=None,
        dashboard_dataset=None,
    ):
        self.config = config
        self.run_name = config.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config.base_output_dir, self.run_name)
        self.add_noise = add_noise_func
        self.best_val_loss = float("inf")
        self.val_history = []
        self.dashboard_history = []
        self.clearml = None
        self.data_config = data_config or {}
        self.dashboard_dataset = dashboard_dataset
        self._resume_contract_payload = {
            "training_config": jsonable(asdict(config)),
            "data_config": jsonable(data_config or {}),
            "dataset_provenance": jsonable(dataset_provenance or {}),
            "experiment_config": jsonable(experiment_config or {}),
            "model_config": jsonable(model_config or {}),
            "code_sha256": {
                name: _file_sha256(Path(__file__).with_name(name))
                for name in (
                    "trainer.py",
                    "data.py",
                    "sampler.py",
                    "flow_parameterization.py",
                    "structured_joint_state.py",
                    "structured_trajectory_evaluation.py",
                )
            },
        }
        self._resume_contract_sha256 = _canonical_sha256(self._resume_contract_payload)
        self.fields = list(self.data_config.get("fields", [f"ch{i}" for i in range(config.out_channels)]))
        self.channel_means = list(self.data_config.get("means", [0.0] * config.out_channels))
        self.channel_stds = list(self.data_config.get("stds", [1.0] * config.out_channels))
        self._shared_viz_obs_mask: torch.Tensor | None = None
        self._shared_viz_obs_mask_attempted = False
        self.activation_checkpointed_modules = configure_activation_checkpointing(
            model, config.activation_checkpointing
        )

        config_payload = jsonable(asdict(config))
        run_metadata = {
            "fields": self.fields,
            "normalization_means": self.channel_means,
            "normalization_stds": self.channel_stds,
            "sample_metrics_values_space": "physical",
            "validation_weight_source": config.validation_weight_source,
            "checkpoint_selection_rule": (
                "minimum EMA validation flow loss"
                if config.validation_weight_source == "ema"
                else "minimum raw-weight validation flow loss"
            ),
            "ema_schedule": (
                "diffusers EMAModel: step=max(0,optimization_step-update_after_step-1); "
                "decay=max(min_decay,min(ema_decay,(1+step)/(10+step)))"
            ),
            "concentration_clipping": "[0, 1]" if "siconc" in self.fields else None,
            "data_config": jsonable(self.data_config),
            "dataset_provenance": jsonable(dataset_provenance or {}),
            "training_config": config_payload,
            "activation_checkpointed_modules": list(self.activation_checkpointed_modules),
            "experiment_config": jsonable(experiment_config) if experiment_config is not None else None,
            "model_config": jsonable(model_config) if model_config is not None else None,
            "resume_contract_sha256": self._resume_contract_sha256,
            "structured_diagnostic_seed_rule": (
                "validation_seed + 1000003*(epoch+1) + 10007*(case_index+1) "
                "+ 101*(member_index+1) + stream_index"
            ),
        }
        existing_structured_run = False
        if self._full_state_recovery_enabled():
            existing_structured_run = _validate_existing_structured_run_files(
                Path(self.output_dir), config_payload, self._resume_contract_sha256
            )

        logging_dir = os.path.join(self.output_dir, "logs")
        _debug(f"output_dir={self.output_dir}")
        _debug(f"creating Accelerator mixed_precision={config.mixed_precision} tracker={config.tracker}")
        accelerator_kwargs = {
            "mixed_precision": config.mixed_precision,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "project_config": ProjectConfiguration(project_dir=self.output_dir, logging_dir=logging_dir),
        }
        if config.tracker:
            accelerator_kwargs["log_with"] = config.tracker
        self.accelerator = Accelerator(**accelerator_kwargs)

        if self.accelerator.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
            if not existing_structured_run:
                _atomic_json(Path(self.output_dir) / "config.json", config_payload)
                _atomic_json(Path(self.output_dir) / "metadata.json", run_metadata)
            elif not self._full_state_recovery_enabled():
                raise RuntimeError("unreachable legacy run provenance branch")
            if config.clearml_enabled:
                _debug("initializing ClearML on main process")
                self.clearml = ClearMLTracker(
                    project_name=config.clearml_project_name,
                    task_name=config.clearml_task_name or self.run_name,
                    tags=config.clearml_tags,
                    output_uri=config.clearml_output_uri,
                    env_path=config.clearml_env_path,
                )
                self.clearml.connect("training_config", jsonable(asdict(config)))
                if experiment_config is not None:
                    self.clearml.connect("experiment_config", jsonable(experiment_config))
                if model_config is not None:
                    self.clearml.connect("model_config", jsonable(model_config))
                if data_config is not None:
                    self.clearml.connect("data_config", jsonable(data_config))
                self.clearml.connect("run_metadata", run_metadata)
            else:
                _debug("ClearML disabled by experiment config")
        else:
            _debug("non-main process: ClearML init skipped")

        if config.tracker:
            self.accelerator.init_trackers("concat_conditioning_training", config=jsonable(asdict(config)))

        if self.accelerator.device.type == "cuda" and hasattr(
            model, "enable_xformers_memory_efficient_attention"
        ):
            model.enable_xformers_memory_efficient_attention()

        (self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler) = (
            self.accelerator.prepare(model, optimizer, data_loader_train, data_loader_val, lr_scheduler)
        )
        if config.preserve_persistent_worker_rng:
            self.train_dataloader = preserve_persistent_worker_rng(self.train_dataloader)
            self.val_dataloader = preserve_persistent_worker_rng(self.val_dataloader)
        _debug(f"accelerator ready device={self.accelerator.device}")

        self.unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.ema_model = EMAModel(
            self.unwrapped_model.parameters(),
            decay=float(config.ema_decay),
            min_decay=float(config.ema_min_decay),
            update_after_step=int(config.ema_update_after_step),
            use_ema_warmup=bool(config.ema_use_warmup),
        )
        self.ema_model.to(self.accelerator.device)

        height, width = config.image_size
        self._grid = make_normalized_xy_grid(height, width).to(self.accelerator.device)

    def _sample_timesteps(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        device = self.accelerator.device
        if self.config.timestep_sampler == "beta":
            if generator is not None:
                raise ValueError("a dedicated generator is not supported for beta timestep sampling")
            alpha, beta = self.config.timestep_beta_params
            return torch.distributions.Beta(alpha, beta).sample((batch_size,)).to(device)
        if self.config.timestep_sampler == "stratified_uniform":
            jitter = torch.rand(batch_size, device=device, generator=generator)
            timesteps = (torch.arange(batch_size, device=device) + jitter) / float(batch_size)
            order = torch.randperm(batch_size, device=device, generator=generator)
            return timesteps[order]
        return torch.rand(batch_size, device=device)

    def _make_sampler(self, model) -> Sampler:
        return Sampler(
            model,
            structured_velocity_parameterization=(self.config.structured_velocity_parameterization),
        )

    def _structured_model_state(self, state: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if self.config.training_objective != "structured_joint_state_flow":
            return state
        return velocity_model_state(
            state,
            timesteps,
            self.config.structured_velocity_parameterization,
        )

    def _reconstruct_model_velocity(
        self,
        model_output: torch.Tensor,
        state: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.training_objective != "structured_joint_state_flow":
            return model_output
        return reconstruct_velocity(
            model_output,
            state,
            timesteps,
            self.config.structured_velocity_parameterization,
        )

    def _structured_diagnostic_seed(
        self,
        epoch: int,
        *,
        case_index: int = -1,
        member_index: int = -1,
        stream_index: int = 0,
    ) -> int:
        return int(
            (
                int(self.config.validation_seed)
                + 1_000_003 * (int(epoch) + 1)
                + 10_007 * (int(case_index) + 1)
                + 101 * (int(member_index) + 1)
                + int(stream_index)
            )
            % (2**63 - 1)
        )

    @contextmanager
    def _structured_diagnostic_rng(self, epoch: int, *, stream_index: int = 0):
        """Keep monitoring/plotting RNG entirely outside the training RNG stream."""
        if self.config.training_objective != "structured_joint_state_flow":
            yield
            return
        device = self.accelerator.device
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            seed = self._structured_diagnostic_seed(epoch, stream_index=stream_index)
            torch.manual_seed(seed)
            if cuda_devices:
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(seed)
            yield

    def _training_loop_start(self) -> tuple[int, int]:
        """Restore the structured run from its last atomic epoch boundary."""
        if not self._full_state_recovery_enabled():
            return 0, 0
        if self.accelerator.num_processes != 1:
            raise ValueError("structured recovery currently requires exactly one process/GPU")
        if self.config.resume_from_checkpoint != "auto":
            return 0, 0
        root = Path(self.output_dir) / self.config.recovery_checkpoint_name
        latest_path = root / "latest.json"
        if not latest_path.exists():
            if latest_path.is_symlink():
                raise ValueError("structured recovery pointer cannot be a dangling symlink")
            return 0, 0
        if latest_path.is_symlink() or not latest_path.is_file():
            raise ValueError("structured recovery pointer must be an ordinary file")
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        state_name = latest.get("state_dir")
        if (
            latest.get("schema_version") != _STRUCTURED_RESUME_SCHEMA
            or not isinstance(state_name, str)
            or not state_name.startswith("epoch_")
        ):
            raise ValueError("structured recovery pointer schema is invalid")
        state_dir = root / state_name
        state_path = state_dir / "resume.json"
        ema_path = state_dir / "ema_state.pth"
        accelerator_path = state_dir / "accelerator"
        if any(path.is_symlink() for path in (state_dir, state_path, ema_path, accelerator_path)):
            raise ValueError("structured recovery paths cannot be symlinks")
        if not state_dir.is_dir() or not state_path.is_file() or not ema_path.is_file():
            raise ValueError("structured recovery state is incomplete")
        if _file_sha256(state_path) != latest.get("resume_sha256"):
            raise ValueError("structured recovery metadata content differs from pointer")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        next_epoch = state.get("next_epoch")
        global_step = state.get("global_step")
        if (
            state.get("schema_version") != _STRUCTURED_RESUME_SCHEMA
            or state.get("contract_sha256") != self._resume_contract_sha256
            or type(next_epoch) is not int
            or not 1 <= next_epoch <= self.config.num_epochs
            or type(global_step) is not int
            or global_step != next_epoch * len(self.train_dataloader)
            or state.get("steps_per_epoch") != len(self.train_dataloader)
        ):
            raise ValueError("structured recovery state violates the run contract")
        if state.get("payload_manifest_sha256") != _canonical_sha256(_checkpoint_payload_manifest(state_dir)):
            raise ValueError("structured recovery payload content is corrupted or incomplete")
        history = state.get("validation_history")
        if (
            not isinstance(history, list)
            or len(history) != next_epoch
            or [row.get("epoch") for row in history] != list(range(next_epoch))
        ):
            raise ValueError("structured recovery validation history is inconsistent")
        self.accelerator.load_state(str(accelerator_path))
        try:
            ema_state = torch.load(ema_path, map_location="cpu", weights_only=True)
        except TypeError:
            ema_state = torch.load(ema_path, map_location="cpu")
        self.ema_model.load_state_dict(ema_state)
        self.ema_model.to(self.accelerator.device)
        self.val_history = history
        self.best_val_loss = float(state["best_val_loss"])
        _debug(f"resumed structured state at epoch={next_epoch} step={global_step}")
        return next_epoch, global_step

    def _save_structured_recovery(self, next_epoch: int, global_step: int) -> None:
        if not self._full_state_recovery_enabled():
            return
        if not self.accelerator.is_main_process or self.accelerator.num_processes != 1:
            return
        if global_step != next_epoch * len(self.train_dataloader):
            raise ValueError("structured recovery can only be saved at an exact epoch boundary")
        root = Path(self.output_dir) / self.config.recovery_checkpoint_name
        root.mkdir(parents=True, exist_ok=True)
        state_name = f"epoch_{next_epoch:04d}"
        target = root / state_name
        temporary = root / f".{state_name}.{os.getpid()}.tmp"
        if target.exists() or target.is_symlink():
            latest_path = root / "latest.json"
            latest = (
                json.loads(latest_path.read_text(encoding="utf-8"))
                if latest_path.is_file() and not latest_path.is_symlink()
                else {}
            )
            if latest.get("state_dir") == state_name:
                raise ValueError(f"committed structured recovery target already exists: {target}")
            if target.is_symlink() or not target.is_dir():
                raise ValueError(f"uncommitted structured recovery target has unsafe type: {target}")
            shutil.rmtree(target)
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()
        try:
            self.accelerator.save_state(str(temporary / "accelerator"), safe_serialization=False)
            torch.save(self.ema_model.state_dict(), temporary / "ema_state.pth")
            payload_manifest = _checkpoint_payload_manifest(temporary)
            state = {
                "schema_version": _STRUCTURED_RESUME_SCHEMA,
                "contract_sha256": self._resume_contract_sha256,
                "next_epoch": next_epoch,
                "global_step": global_step,
                "steps_per_epoch": len(self.train_dataloader),
                "best_val_loss": float(self.best_val_loss),
                "validation_history": self.val_history,
                "payload_manifest_sha256": _canonical_sha256(payload_manifest),
            }
            _atomic_json(temporary / "resume.json", state)
            os.replace(temporary, target)
            pointer = {
                "schema_version": _STRUCTURED_RESUME_SCHEMA,
                "state_dir": state_name,
                "resume_sha256": _file_sha256(target / "resume.json"),
            }
            _atomic_json(root / "latest.json", pointer)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return 0, 0

    def _full_state_recovery_enabled(self) -> bool:
        """Whether this trainer commits complete optimizer/scheduler/RNG state per epoch."""
        return self.config.training_objective == "structured_joint_state_flow"

    def _after_training_epoch(self, epoch: int, global_step: int) -> None:
        """Extension point called after all ordinary epoch artifacts are durable."""

    def _batch_to_device(self, batch: dict) -> dict[str, torch.Tensor]:
        moved = {
            key: batch[key].to(self.accelerator.device, dtype=torch.float32, non_blocking=True)
            for key in ("truth", "background", "obs_values", "obs_mask", "valid_mask", "water_mask")
        }
        if "structured_conditioning" in batch:
            moved["structured_conditioning"] = batch["structured_conditioning"].to(
                self.accelerator.device,
                dtype=torch.float32,
                non_blocking=True,
            )
        for key in ("structured_physical_truth", "structured_physical_background"):
            if key in batch:
                moved[key] = batch[key].to(
                    self.accelerator.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
        if "structured_flow_mask" in batch:
            moved["structured_flow_mask"] = batch["structured_flow_mask"].to(
                self.accelerator.device,
                dtype=torch.float32,
                non_blocking=True,
            )
        for key in ("structured_lag0_mask", "structured_lag0_physical_values"):
            if key in batch:
                moved[key] = batch[key].to(
                    self.accelerator.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
        return moved

    def _conditioned_inputs(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        background = batch["background"]
        if self.config.training_objective == "structured_joint_state_flow":
            background_mask = batch["valid_mask"][:, :1].expand_as(background)
            return background, background_mask, batch["obs_values"], batch["obs_mask"]
        background_mask = torch.ones_like(background)
        obs_values = batch["obs_values"]
        obs_mask = batch["obs_mask"]
        mode_probabilities = getattr(self.config, "conditioning_mode_probabilities", None)
        if mode_probabilities is not None:
            names = ("no_background", "no_track", "no_conditioning", "both")
            unexpected = set(mode_probabilities) - set(names)
            if unexpected:
                raise ValueError(f"Unknown conditioning modes: {sorted(unexpected)}")
            probabilities = [float(mode_probabilities.get(name, 0.0)) for name in names]
            if any(probability < 0.0 for probability in probabilities):
                raise ValueError("conditioning_mode_probabilities cannot contain negative values")
            if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("conditioning_mode_probabilities must sum to 1.0")
            if self.model.training:
                draw = torch.rand((background.shape[0], 1, 1, 1), device=background.device)
                no_background = draw < probabilities[0]
                no_track = (draw >= probabilities[0]) & (draw < probabilities[0] + probabilities[1])
                no_conditioning = (draw >= probabilities[0] + probabilities[1]) & (
                    draw < probabilities[0] + probabilities[1] + probabilities[2]
                )
                keep_background = (~(no_background | no_conditioning)).to(dtype=background.dtype)
                keep_track = (~(no_track | no_conditioning)).to(dtype=obs_values.dtype)
                background = background * keep_background
                background_mask = background_mask * keep_background
                obs_values = obs_values * keep_track
                obs_mask = obs_mask * keep_track.to(dtype=obs_mask.dtype)
            return background, background_mask, obs_values, obs_mask

        dropout_probability = float(getattr(self.config, "background_dropout_probability", 0.0))
        if not 0.0 <= dropout_probability <= 1.0:
            raise ValueError(
                f"background_dropout_probability must be within [0, 1], got {dropout_probability}"
            )
        if self.model.training and dropout_probability > 0.0:
            keep = (
                torch.rand((background.shape[0], 1, 1, 1), device=background.device) >= dropout_probability
            ).to(dtype=background.dtype)
            background = background * keep
            background_mask = background_mask * keep
        return background, background_mask, obs_values, obs_mask

    def _conditioned_background(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._conditioned_inputs(batch)[0]

    def _make_model_input(
        self,
        noisy_truth: torch.Tensor,
        batch: dict[str, torch.Tensor],
        background: torch.Tensor | None = None,
        background_mask: torch.Tensor | None = None,
        obs_values: torch.Tensor | None = None,
        obs_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = noisy_truth.shape[0]
        grid = self._grid.expand(batch_size, -1, -1, -1)
        if self.config.training_objective == "structured_joint_state_flow":
            condition = batch.get("structured_conditioning")
            if condition is None:
                raise KeyError("structured_joint_state_flow requires batch['structured_conditioning']")
            model_input = torch.cat((noisy_truth, grid, condition), dim=1)
            if model_input.shape[1] != self.config.in_channels:
                raise ValueError(
                    f"structured model input has {model_input.shape[1]} channels, "
                    f"expected {self.config.in_channels}"
                )
            return model_input
        if background is None or background_mask is None or obs_values is None or obs_mask is None:
            (
                conditioned_background,
                conditioned_background_mask,
                conditioned_obs_values,
                conditioned_obs_mask,
            ) = self._conditioned_inputs(batch)
            background = conditioned_background if background is None else background
            background_mask = conditioned_background_mask if background_mask is None else background_mask
            obs_values = conditioned_obs_values if obs_values is None else obs_values
            obs_mask = conditioned_obs_mask if obs_mask is None else obs_mask
        return make_conditioned_model_input(
            noisy_truth,
            grid,
            background,
            background_mask,
            obs_values,
            obs_mask,
            batch["water_mask"],
        )

    def _make_training_pair(
        self,
        truth: torch.Tensor,
        batch: dict[str, torch.Tensor],
        timesteps: torch.Tensor,
        residual_background: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.training_objective == "flow":
            return self.add_noise(truth, timesteps)
        if self.config.training_objective == "residual_flow":
            background = batch["background"] if residual_background is None else residual_background
            return self.add_noise(truth - background, timesteps)
        if self.config.training_objective == "bridge":
            t = timesteps.view(-1, *([1] * (truth.dim() - 1)))
            state = (1.0 - t) * truth + t * batch["background"]
            target_velocity = batch["background"] - truth
            return state, target_velocity
        if self.config.training_objective == "structured_joint_state_flow":
            physical_truth = batch.get("structured_physical_truth")
            if physical_truth is None:
                raise KeyError("structured_joint_state_flow requires batch['structured_physical_truth']")
            clean = encode_structured_joint_trajectory(
                physical_truth,
                batch["valid_mask"][:, :1],
                self.config.structured_state_stats,
                generator=generator,
            )
            flow_mask = batch.get("structured_flow_mask", batch["valid_mask"][:, :1])
            valid = flow_mask.expand_as(clean) > 0
            clean = torch.where(valid, clean, torch.zeros_like(clean))
            noise = torch.randn(
                clean.shape,
                device=clean.device,
                dtype=clean.dtype,
                generator=generator,
            )
            noise = torch.where(valid, noise, torch.zeros_like(noise))
            t = timesteps.view(-1, *([1] * (clean.dim() - 1)))
            return (1.0 - t) * clean + t * noise, noise - clean
        raise ValueError(
            f"Unknown training_objective={self.config.training_objective!r}; "
            "expected 'flow', 'residual_flow', 'bridge', or 'structured_joint_state_flow'"
        )

    def _sample_target(self) -> str:
        return "residual" if self.config.training_objective == "residual_flow" else "state"

    def _sample_solver_kwargs(self) -> dict[str, object]:
        method = str(getattr(self.config, "sample_method", "euler"))
        return {
            "method": method,
            "rtol": float(getattr(self.config, "sample_rtol", 1e-3)),
            "atol": float(getattr(self.config, "sample_atol", 1e-4)),
            "end_time": float(getattr(self.config, "sample_end_time", 0.001)),
            "memory_efficient_euler": method == "euler",
        }

    def _cfg_sampler_kwargs(self) -> dict[str, object]:
        return {
            "cfg_mode": str(getattr(self.config, "sample_cfg_mode", "none")),
            "cfg_background_scale": float(getattr(self.config, "sample_cfg_background_scale", 1.0)),
            "cfg_observation_scale": float(getattr(self.config, "sample_cfg_observation_scale", 1.0)),
        }

    def _physical_metric_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.config.training_objective == "structured_joint_state_flow":
            return decode_structured_joint_trajectory(
                tensor.to(dtype=torch.float32),
                self.config.structured_state_stats,
            )
        physical = channel_denormalize(tensor.to(dtype=torch.float32), self.channel_means, self.channel_stds)
        if "siconc" in self.fields:
            concentration_channel = self.fields.index("siconc")
            physical[:, concentration_channel].clamp_(0.0, 1.0)
        return physical

    @contextmanager
    def _sampling_model(self):
        unwrapped = self.accelerator.unwrap_model(self.model)
        if not self.config.sample_use_ema:
            yield unwrapped
            return

        self.ema_model.store(unwrapped.parameters())
        self.ema_model.copy_to(unwrapped.parameters())
        try:
            yield unwrapped
        finally:
            self.ema_model.restore(unwrapped.parameters())

    def _sampling_weight_label(self) -> str:
        if self.config.sample_use_ema:
            return f"EMA weights, decay={float(self.config.ema_decay):.6g}"
        return "raw training weights"

    @staticmethod
    def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.expand_as(pred)
        denom = mask.sum().clamp(min=1.0)
        return (F.mse_loss(pred, target, reduction="none") * mask).sum() / denom

    def _flow_matching_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.config.loss_domain == "valid":
            mask = batch.get("structured_flow_mask", batch["valid_mask"])
            if mask.shape[1] != pred.shape[1]:
                mask = mask[:, :1]
            return self._masked_mse(pred, target, mask)
        return F.mse_loss(pred, target)

    @staticmethod
    def _masked_smoothness_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.expand_as(pred)
        dy_mask = mask[..., 1:, :] * mask[..., :-1, :]
        dx_mask = mask[..., :, 1:] * mask[..., :, :-1]
        dy = (pred[..., 1:, :] - pred[..., :-1, :]).square() * dy_mask
        dx = (pred[..., :, 1:] - pred[..., :, :-1]).square() * dx_mask
        denom = (dy_mask.sum() + dx_mask.sum()).clamp(min=1.0)
        return (dy.sum() + dx.sum()) / denom

    def _sea_ice_concentration_smoothness_loss(
        self,
        pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.smoothness_loss_weight <= 0.0 or "siconc" not in self.fields:
            return pred.new_tensor(0.0)
        channel = self.fields.index("siconc")
        return self._masked_smoothness_loss(
            pred[:, channel : channel + 1],
            mask[:, channel : channel + 1],
        )

    @staticmethod
    def _masked_error_metrics(
        pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> tuple[float, float]:
        mask = mask.expand_as(pred)
        denom = mask.sum().clamp(min=1.0)
        diff = (pred - target) * mask
        mae = diff.abs().sum() / denom
        rmse = torch.sqrt((diff.square().sum() / denom).clamp(min=0.0))
        return float(mae.item()), float(rmse.item())

    @staticmethod
    def _masked_error_sums(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[float, float, float]:
        mask = mask.expand_as(pred)
        count = mask.sum()
        diff = (pred - target) * mask
        return (
            float(diff.abs().sum().item()),
            float(diff.square().sum().item()),
            float(count.item()),
        )

    def save_model_custom(self, name: str = "last_model.pth"):
        os.makedirs(self.output_dir, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model)
        torch.save(unwrapped.state_dict(), os.path.join(self.output_dir, name))
        torch.save(self.ema_model.state_dict(), os.path.join(self.output_dir, f"ema_{name}"))

    def _report_train_metrics(self, loss, loss_full, loss_obs, loss_smooth, step: int):
        if self.clearml is None:
            return
        self.clearml.report_scalar("loss_normalized/total", "train", loss.item(), step)
        self.clearml.report_scalar("loss_normalized/full", "train", loss_full.item(), step)
        self.clearml.report_scalar("loss_normalized/observed", "train", loss_obs.item(), step)
        self.clearml.report_scalar("loss_normalized/smoothness", "train", loss_smooth.item(), step)

    def _report_val_metrics(self, loss_full: float, loss_obs: float, loss: float, step: int):
        if self.clearml is None:
            return
        self.clearml.report_scalar("loss_normalized/total", "validation", loss, step)
        self.clearml.report_scalar("loss_normalized/full", "validation", loss_full, step)
        self.clearml.report_scalar("loss_normalized/observed", "validation", loss_obs, step)

    def _report_sample_validation_metrics(self, metrics: dict[str, float], step: int):
        if self.clearml is None or not metrics:
            return

        if self.config.training_objective == "structured_joint_state_flow":
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    self.clearml.report_scalar("structured_trajectory/metric", key, float(value), step)
            return

        def _report(title: str, series: str, key: str):
            value = metrics.get(key)
            if value is None:
                return
            value = float(value)
            if math.isfinite(value):
                self.clearml.report_scalar(title, series, value, step)

        for kind in ("mae", "rmse"):
            for field_name in self.fields:
                for region in ("full", "obs"):
                    suffix = f"{field_name}_{region}"
                    for aggregate in ("mean", "min", "max", "of_mean"):
                        _report(
                            f"metric_physical/analysis_{kind}",
                            f"{field_name}/{aggregate}_{region}",
                            f"analysis_{kind}_{aggregate}_{suffix}",
                        )
                    _report(
                        f"metric_physical/analysis_{kind}",
                        f"{field_name}/background_{region}",
                        f"background_{kind}_{suffix}",
                    )
                    _report(
                        f"metric_physical/background_{kind}",
                        f"{field_name}/{region}",
                        f"background_{kind}_{suffix}",
                    )
        for field_name in self.fields:
            for region in ("full", "obs"):
                suffix = f"{field_name}_{region}"
                for aggregate in ("mean", "min", "max", "of_mean"):
                    _report(
                        "metric_physical/analysis_rmse_skill",
                        f"{field_name}/{aggregate}_{region}",
                        f"analysis_rmse_skill_{aggregate}_{suffix}",
                    )
                _report("metric_physical/count", f"{field_name}/{region}", f"metric_count_{suffix}")

    @staticmethod
    def _meta_mean(value):
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            return float(value.to(dtype=torch.float32).mean().item())
        if isinstance(value, (list, tuple)):
            numeric = [float(item) for item in value if isinstance(item, (int, float))]
            if not numeric:
                return None
            return float(sum(numeric) / len(numeric))
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _report_condition_diagnostics(self, raw_batch: dict, step: int, series: str):
        if self.clearml is None or "meta" not in raw_batch:
            return
        meta = raw_batch["meta"]
        if not isinstance(meta, dict):
            return
        for key in ("obs_count", "observed_fraction", "sral_files_used", "empty_obs_days"):
            if key not in meta:
                continue
            value = self._meta_mean(meta[key])
            if value is not None:
                self.clearml.report_scalar(f"condition/{key}", series, value, step)

    def _upload_clearml_artifacts(self):
        if self.clearml is None:
            return
        output_dir = os.path.abspath(self.output_dir)
        self.clearml.upload_artifact("training_config", os.path.join(output_dir, "config.json"))
        self.clearml.upload_artifact("run_metadata", os.path.join(output_dir, "metadata.json"))
        self.clearml.upload_artifact("metrics", os.path.join(output_dir, "metrics.json"))
        if self.config.clearml_upload_checkpoints:
            for name in ("last_model.pth", "ema_last_model.pth", "best_model.pth", "ema_best_model.pth"):
                self.clearml.upload_artifact(name, os.path.join(output_dir, name))

    def compute_val_loss(self) -> tuple[float, float, float]:
        if self.config.validation_weight_source == "ema":
            with self._sampling_model():
                return self._compute_val_loss_on_current_weights()
        return self._compute_val_loss_on_current_weights()

    def _compute_val_loss_on_current_weights(self) -> tuple[float, float, float]:
        self.model.eval()
        total_full = 0.0
        n = len(self.val_dataloader)
        _debug(f"validation start batches={n}")
        validation_generator = None
        if self.config.training_objective == "structured_joint_state_flow":
            validation_generator = torch.Generator(device=self.accelerator.device)
            validation_generator.manual_seed(
                int(self.config.validation_seed) + int(self.accelerator.process_index)
            )

        with torch.no_grad():
            for raw_batch in self.val_dataloader:
                batch = self._batch_to_device(raw_batch)
                truth = batch["truth"]
                batch_size = truth.shape[0]
                timesteps = self._sample_timesteps(
                    batch_size,
                    generator=validation_generator,
                )
                background, background_mask, obs_values, obs_mask = self._conditioned_inputs(batch)
                model_state, v_real = self._make_training_pair(
                    truth,
                    batch,
                    timesteps,
                    residual_background=background,
                    generator=validation_generator,
                )

                model_input = self._make_model_input(
                    self._structured_model_state(model_state, timesteps),
                    batch,
                    background=background,
                    background_mask=background_mask,
                    obs_values=obs_values,
                    obs_mask=obs_mask,
                )
                model_output = self.model(model_input, timesteps * 1000, return_dict=False)[0]
                v_pred = self._reconstruct_model_velocity(model_output, model_state, timesteps)
                loss_full = self._flow_matching_loss(v_pred, v_real, batch)

                total_full += self.accelerator.gather_for_metrics(loss_full).mean().item()

        loss_full = total_full / max(n, 1)
        loss_obs = 0.0
        _debug(f"validation done loss_full={loss_full:.6f} loss_obs={loss_obs:.6f}")
        return loss_full, loss_obs, loss_full

    @staticmethod
    def _empty_metric_totals() -> dict[str, float]:
        return {
            "analysis_abs": 0.0,
            "analysis_sq": 0.0,
            "background_abs": 0.0,
            "background_sq": 0.0,
            "count": 0.0,
        }

    @staticmethod
    def _finalize_metric_totals(prefix: str, totals: dict[str, float]) -> dict[str, float]:
        count = totals["count"]
        if count <= 0.0:
            return {
                f"analysis_mae_{prefix}": float("nan"),
                f"analysis_rmse_{prefix}": float("nan"),
                f"background_mae_{prefix}": float("nan"),
                f"background_rmse_{prefix}": float("nan"),
                f"analysis_rmse_skill_{prefix}": float("nan"),
                f"metric_count_{prefix}": 0.0,
            }

        analysis_mae = totals["analysis_abs"] / count
        analysis_rmse = math.sqrt(max(totals["analysis_sq"] / count, 0.0))
        background_mae = totals["background_abs"] / count
        background_rmse = math.sqrt(max(totals["background_sq"] / count, 0.0))
        skill = float("nan") if background_rmse <= 0.0 else 1.0 - analysis_rmse / background_rmse
        return {
            f"analysis_mae_{prefix}": analysis_mae,
            f"analysis_rmse_{prefix}": analysis_rmse,
            f"background_mae_{prefix}": background_mae,
            f"background_rmse_{prefix}": background_rmse,
            f"analysis_rmse_skill_{prefix}": skill,
            f"metric_count_{prefix}": count,
        }

    def _validation_cases(self, max_cases: int) -> list[dict[str, torch.Tensor]]:
        cases = []
        if max_cases <= 0:
            return cases

        for raw_batch in self.val_dataloader:
            batch = self._batch_to_device(raw_batch)
            batch_size = batch["truth"].shape[0]
            for batch_idx in range(batch_size):
                cases.append({key: value[batch_idx : batch_idx + 1] for key, value in batch.items()})
                if len(cases) >= max_cases:
                    return cases
        return cases

    def _iter_validation_batches(self, max_cases: int):
        if max_cases <= 0:
            return
        remaining = max_cases
        for raw_batch in self.val_dataloader:
            if remaining <= 0:
                return
            batch = self._batch_to_device(raw_batch)
            batch_size = batch["truth"].shape[0]
            take = min(batch_size, remaining)
            if take < batch_size:
                batch = {key: value[:take] for key, value in batch.items()}
            remaining -= take
            yield batch

    def _metric_case_indices(self, max_cases: int, stride_days: int) -> list[int]:
        val_dataset = getattr(self.val_dataloader, "dataset", None)
        if val_dataset is None or max_cases <= 0:
            return []
        hours_per_day = int(getattr(val_dataset, "hours_per_day", 1))
        if hasattr(val_dataset, "strided_case_indices"):
            indices = val_dataset.strided_case_indices(max_cases, stride_days=stride_days)
            if indices:
                if hours_per_day > 1:
                    metric_hour = min(max(int(getattr(val_dataset, "hour_index", 0)), 0), hours_per_day - 1)
                    return [(idx // hours_per_day) * hours_per_day + metric_hour for idx in indices]
                return indices
        return list(range(min(max_cases, len(val_dataset))))

    def _iter_indexed_batches(self, indices: list[int], batch_size: int):
        val_dataset = getattr(self.val_dataloader, "dataset", None)
        if val_dataset is None or not indices:
            return
        from torch.utils.data import default_collate

        batch_size = max(int(batch_size), 1)
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            samples = [val_dataset[idx] for idx in chunk]
            raw_batch = default_collate(samples)
            yield self._batch_to_device(raw_batch)

    @torch.no_grad()
    def compute_sample_validation_metrics(self, epoch: int | None = None) -> dict[str, float]:
        if self.config.training_objective == "structured_joint_state_flow":
            return self._compute_structured_trajectory_metrics(epoch)
        max_cases = int(self.config.metric_num_cases)
        if max_cases <= 0:
            return {}

        num_ensemble = max(int(getattr(self.config, "metric_num_ensemble", 1)), 1)
        stride_days = max(int(getattr(self.config, "metric_stride_days", 30)), 1)
        metric_timesteps = int(self.config.metric_num_timesteps)
        if metric_timesteps <= 0:
            metric_timesteps = int(self.config.num_sample_timesteps)

        indices = self._metric_case_indices(max_cases, stride_days)
        if not indices:
            return {}

        _debug(
            f"sample validation metrics start cases={len(indices)} "
            f"ensemble={num_ensemble} timesteps={metric_timesteps} stride_days={stride_days}"
        )
        totals = {
            field_name: {
                region: {
                    "background": self._empty_metric_totals(),
                    "members": [self._empty_metric_totals() for _ in range(num_ensemble)],
                    "of_mean": self._empty_metric_totals(),
                }
                for region in ("full", "obs")
            }
            for field_name in self.fields
        }
        total_processed = 0
        save_ensemble = (
            bool(getattr(self.config, "metric_save_ensemble_samples", False)) and epoch is not None
        )
        saved_batches = []
        if save_ensemble:
            save_dtype_name = str(getattr(self.config, "metric_ensemble_save_dtype", "float16")).lower()
            save_dtypes = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }
            if save_dtype_name not in save_dtypes:
                raise ValueError(
                    "metric_ensemble_save_dtype must be one of "
                    f"{sorted(save_dtypes)}, got {save_dtype_name!r}"
                )
            save_dtype = save_dtypes[save_dtype_name]

        self.model.eval()
        with self._sampling_model() as sample_model:
            sampler = self._make_sampler(sample_model)
            for batch in self._iter_indexed_batches(indices, self.config.eval_batch_size):
                ensemble = []
                for _ in range(num_ensemble):
                    ensemble.append(
                        sampler.sample_conditioned(
                            background=batch["background"],
                            background_mask=torch.ones_like(batch["background"]),
                            obs_values=batch["obs_values"],
                            obs_mask=batch["obs_mask"],
                            water_mask=batch["water_mask"],
                            size=self.config.image_size,
                            num_timesteps=metric_timesteps,
                            device=self.accelerator.device,
                            start_mode=self.config.sample_start_mode,
                            start_noise_level=self.config.sample_start_noise_level,
                            enforce_observations=self.config.sample_enforce_observations,
                            valid_mask=batch["valid_mask"],
                            obs_guidance_scale=self.config.sample_obs_guidance_scale,
                            obs_guidance_eps=self.config.sample_obs_guidance_eps,
                            sample_target=self._sample_target(),
                            **self._cfg_sampler_kwargs(),
                            **self._sample_solver_kwargs(),
                        )
                    )
                ensemble_tensor = torch.stack(ensemble, dim=1)
                if save_ensemble:
                    saved_batches.append(
                        {
                            "samples": ensemble_tensor.detach().to(device="cpu", dtype=save_dtype),
                            "truth": batch["truth"].detach().to(device="cpu", dtype=save_dtype),
                            "background": batch["background"].detach().to(device="cpu", dtype=save_dtype),
                            "obs_values": batch["obs_values"].detach().to(device="cpu", dtype=save_dtype),
                            "obs_mask": batch["obs_mask"].detach().to(device="cpu", dtype=save_dtype),
                            "valid_mask": batch["valid_mask"].detach().to(device="cpu", dtype=save_dtype),
                            "water_mask": batch["water_mask"].detach().to(device="cpu", dtype=save_dtype),
                        }
                    )
                n_cases, n_members, n_channels, height, width = ensemble_tensor.shape
                ensemble_physical = self._physical_metric_tensor(
                    ensemble_tensor.reshape(n_cases * n_members, n_channels, height, width)
                ).reshape(n_cases, n_members, n_channels, height, width)
                ensemble_mean = ensemble_physical.mean(dim=1)
                truth_physical = self._physical_metric_tensor(batch["truth"])
                background_physical = self._physical_metric_tensor(batch["background"])
                for channel, field_name in enumerate(self.fields):
                    for region, mask in (("full", batch["valid_mask"]), ("obs", batch["obs_mask"])):
                        channel_mask = mask[:, channel : channel + 1]
                        truth_channel = truth_physical[:, channel : channel + 1]
                        background_abs, background_sq, background_count = self._masked_error_sums(
                            background_physical[:, channel : channel + 1], truth_channel, channel_mask
                        )
                        background_totals = totals[field_name][region]["background"]
                        background_totals["background_abs"] += background_abs
                        background_totals["background_sq"] += background_sq
                        background_totals["count"] += background_count

                        for member_index, analysis in enumerate(ensemble_physical.unbind(dim=1)):
                            analysis_abs, analysis_sq, count = self._masked_error_sums(
                                analysis[:, channel : channel + 1], truth_channel, channel_mask
                            )
                            member_totals = totals[field_name][region]["members"][member_index]
                            member_totals["analysis_abs"] += analysis_abs
                            member_totals["analysis_sq"] += analysis_sq
                            member_totals["count"] += min(count, background_count)

                        analysis_abs, analysis_sq, count = self._masked_error_sums(
                            ensemble_mean[:, channel : channel + 1], truth_channel, channel_mask
                        )
                        mean_totals = totals[field_name][region]["of_mean"]
                        mean_totals["analysis_abs"] += analysis_abs
                        mean_totals["analysis_sq"] += analysis_sq
                        mean_totals["count"] += min(count, background_count)
                total_processed += int(batch["truth"].shape[0])

        if total_processed == 0:
            return {}

        metrics = {
            "metric_num_cases": float(total_processed),
            "metric_num_timesteps": float(metric_timesteps),
            "metric_num_ensemble": float(num_ensemble),
            "metric_stride_days": float(stride_days),
        }
        for field_name in self.fields:
            for region in ("full", "obs"):
                suffix = f"{field_name}_{region}"
                background_totals = totals[field_name][region]["background"]

                def with_background(analysis_totals: dict[str, float]) -> dict[str, float]:
                    return {
                        "analysis_abs": analysis_totals["analysis_abs"],
                        "analysis_sq": analysis_totals["analysis_sq"],
                        "background_abs": background_totals["background_abs"],
                        "background_sq": background_totals["background_sq"],
                        "count": background_totals["count"],
                    }

                member_metrics = [
                    self._finalize_metric_totals(suffix, with_background(member_totals))
                    for member_totals in totals[field_name][region]["members"]
                ]
                mean_field_metrics = self._finalize_metric_totals(
                    suffix, with_background(totals[field_name][region]["of_mean"])
                )
                metrics[f"background_mae_{suffix}"] = mean_field_metrics[f"background_mae_{suffix}"]
                metrics[f"background_rmse_{suffix}"] = mean_field_metrics[f"background_rmse_{suffix}"]
                metrics[f"metric_count_{suffix}"] = mean_field_metrics[f"metric_count_{suffix}"]
                for key in ("analysis_mae", "analysis_rmse"):
                    values = [member[f"{key}_{suffix}"] for member in member_metrics]
                    metrics[f"{key}_mean_{suffix}"] = sum(values) / len(values)
                    metrics[f"{key}_min_{suffix}"] = min(values)
                    metrics[f"{key}_max_{suffix}"] = max(values)
                    metrics[f"{key}_of_mean_{suffix}"] = mean_field_metrics[f"{key}_{suffix}"]
                skill_values = [member[f"analysis_rmse_skill_{suffix}"] for member in member_metrics]
                metrics[f"analysis_rmse_skill_mean_{suffix}"] = sum(skill_values) / len(skill_values)
                metrics[f"analysis_rmse_skill_min_{suffix}"] = min(skill_values)
                metrics[f"analysis_rmse_skill_max_{suffix}"] = max(skill_values)
                metrics[f"analysis_rmse_skill_of_mean_{suffix}"] = mean_field_metrics[
                    f"analysis_rmse_skill_{suffix}"
                ]
        if saved_batches:
            samples_dir = os.path.join(self.output_dir, "samples")
            os.makedirs(samples_dir, exist_ok=True)
            artifact = {
                key: torch.cat([batch[key] for batch in saved_batches], dim=0)
                for key in (
                    "samples",
                    "truth",
                    "background",
                    "obs_values",
                    "obs_mask",
                    "valid_mask",
                    "water_mask",
                )
            }
            artifact.update(
                {
                    "case_indices": torch.as_tensor(indices[:total_processed], dtype=torch.long),
                    "epoch": int(epoch),
                    "num_timesteps": int(metric_timesteps),
                    "num_ensemble": int(num_ensemble),
                    "stride_days": int(stride_days),
                    "save_dtype": save_dtype_name,
                    "values_space": "normalized",
                    "metrics_values_space": "physical",
                    "concentration_clipping": "[0, 1]" if "siconc" in self.fields else None,
                }
            )
            artifact_path = os.path.join(samples_dir, f"epoch_{epoch:04d}_metric_ensemble.pt")
            torch.save(artifact, artifact_path)
            _debug(f"saved sample validation ensemble: {artifact_path}")
        primary_suffix = f"{self.fields[0]}_full"
        analysis_mean = metrics[f"analysis_rmse_mean_{primary_suffix}"]
        analysis_of_mean = metrics[f"analysis_rmse_of_mean_{primary_suffix}"]
        background_rmse = metrics[f"background_rmse_{primary_suffix}"]
        _debug(
            "sample validation metrics done "
            f"analysis_rmse_mean_{primary_suffix}={analysis_mean:.6f} "
            f"analysis_rmse_of_mean_{primary_suffix}={analysis_of_mean:.6f} "
            f"background_rmse_{primary_suffix}={background_rmse:.6f}"
        )
        return metrics

    @torch.no_grad()
    def _compute_structured_trajectory_metrics(self, epoch: int | None = None) -> dict:
        max_cases = int(self.config.metric_num_cases)
        num_ensemble = int(self.config.metric_num_ensemble)
        if max_cases <= 0:
            return {}
        if num_ensemble < 2:
            raise ValueError("structured trajectory metrics require at least two members")
        indices = self._metric_case_indices(max_cases, int(self.config.metric_stride_days))
        ensembles = []
        truths = []
        backgrounds = []
        valid_masks = []
        lag0_masks = []
        self.model.eval()
        with self._sampling_model() as sample_model:
            sampler = self._make_sampler(sample_model)
            effective_epoch = int(epoch) if epoch is not None else 0
            for case_index in indices:
                batch = next(self._iter_indexed_batches([case_index], 1))
                members = []
                for member_index in range(num_ensemble):
                    member_seed = self._structured_diagnostic_seed(
                        effective_epoch,
                        case_index=case_index,
                        member_index=member_index,
                        stream_index=11,
                    )
                    generator = torch.Generator(device=self.accelerator.device)
                    generator.manual_seed(member_seed)
                    initial_noise = torch.randn(
                        (1, self.config.out_channels, *self.config.image_size),
                        dtype=batch["background"].dtype,
                        device=self.accelerator.device,
                        generator=generator,
                    )
                    try:
                        members.append(
                            sample_structured_batch(
                                sampler,
                                batch,
                                stats=self.config.structured_state_stats,
                                size=self.config.image_size,
                                num_timesteps=self.config.metric_num_timesteps,
                                device=self.accelerator.device,
                                method=self.config.sample_method,
                                rtol=self.config.sample_rtol,
                                atol=self.config.sample_atol,
                                initial_noise=initial_noise,
                            )
                        )
                    except StructuredDecodeSaturationError as error:
                        raise StructuredDecodeSaturationError(
                            f"validation case={case_index} member={member_index}: {error}",
                            {
                                **error.diagnostics,
                                "case_index": int(case_index),
                                "member_index": int(member_index),
                                "member_seed": int(member_seed),
                            },
                        ) from error
                ensembles.append(torch.stack(members, dim=1))
                truths.append(batch["structured_physical_truth"])
                backgrounds.append(batch["structured_physical_background"])
                valid_masks.append(batch["valid_mask"])
                lag0_masks.append(batch["structured_lag0_mask"])
        if not ensembles:
            return {}
        ensemble = torch.cat(ensembles, dim=0)
        metrics = structured_trajectory_metrics(
            ensemble,
            torch.cat(truths, dim=0),
            torch.cat(backgrounds, dim=0),
            torch.cat(valid_masks, dim=0),
            lag0_mask=torch.cat(lag0_masks, dim=0),
            sic_cap=float(self.config.structured_state_stats["sic_cap"]),
            exclude_lag0_from_day0_scores=(
                bool(self.config.trajectory_lead_days) and self.config.trajectory_lead_days[0] == 0
            ),
        )
        metrics.update(
            {
                "metric_num_cases": float(ensemble.shape[0]),
                "metric_num_ensemble": float(num_ensemble),
                "metric_num_timesteps": float(self.config.metric_num_timesteps),
            }
        )
        if epoch is not None:
            samples_dir = os.path.join(self.output_dir, "samples")
            os.makedirs(samples_dir, exist_ok=True)
            torch.save(
                {
                    "samples": ensemble.detach().cpu(),
                    "truth": torch.cat(truths, dim=0).detach().cpu(),
                    "background": torch.cat(backgrounds, dim=0).detach().cpu(),
                    "valid_mask": torch.cat(valid_masks, dim=0).detach().cpu(),
                    "lag0_mask": torch.cat(lag0_masks, dim=0).detach().cpu(),
                    "values_space": "physical",
                    "trajectory_horizon_days": self.config.trajectory_horizon_days,
                    "metrics": metrics,
                },
                os.path.join(samples_dir, f"epoch_{epoch:04d}_structured_ensemble.pt"),
            )
        return metrics

    def save_samples(self, epoch: int):
        if self.config.training_objective == "structured_joint_state_flow":
            self._report_structured_trajectory_dashboard(epoch)
            return
        self.model.eval()
        raw_batch = next(iter(self.val_dataloader))
        batch = self._batch_to_device(raw_batch)
        one = {key: value[:1] for key, value in batch.items()}
        with self._sampling_model() as sample_model:
            sampler = self._make_sampler(sample_model)
            sample = sampler.sample_conditioned(
                background=one["background"],
                background_mask=torch.ones_like(one["background"]),
                obs_values=one["obs_values"],
                obs_mask=one["obs_mask"],
                water_mask=one["water_mask"],
                size=self.config.image_size,
                num_timesteps=self.config.num_sample_timesteps,
                device=self.accelerator.device,
                start_mode=self.config.sample_start_mode,
                start_noise_level=self.config.sample_start_noise_level,
                enforce_observations=self.config.sample_enforce_observations,
                valid_mask=one["valid_mask"],
                obs_guidance_scale=self.config.sample_obs_guidance_scale,
                obs_guidance_eps=self.config.sample_obs_guidance_eps,
                sample_target=self._sample_target(),
                **self._cfg_sampler_kwargs(),
                **self._sample_solver_kwargs(),
            )
        samples_dir = os.path.join(self.output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        torch.save(
            {
                "sample": sample.detach().cpu(),
                "truth": one["truth"].detach().cpu(),
                "background": one["background"].detach().cpu(),
                "obs_values": one["obs_values"].detach().cpu(),
                "obs_mask": one["obs_mask"].detach().cpu(),
                "valid_mask": one["valid_mask"].detach().cpu(),
                "water_mask": one["water_mask"].detach().cpu(),
            },
            os.path.join(samples_dir, f"epoch_{epoch:04d}.pt"),
        )

    def _ensure_shared_viz_obs_mask(self) -> torch.Tensor | None:
        if self._shared_viz_obs_mask is not None or self._shared_viz_obs_mask_attempted:
            return self._shared_viz_obs_mask
        self._shared_viz_obs_mask_attempted = True

        val_dataset = self.dashboard_dataset
        if val_dataset is None:
            val_dataset = getattr(self.val_dataloader, "dataset", None)
        if val_dataset is None:
            return None
        if hasattr(val_dataset, "strided_case_indices"):
            scan_indices = val_dataset.strided_case_indices(30, stride_days=5)
        else:
            scan_indices = list(range(min(30, len(val_dataset))))

        best_mask = None
        best_count = 0.0
        for idx in scan_indices:
            try:
                sample = val_dataset[idx]
            except Exception as exc:
                _debug(f"shared viz mask scan idx={idx} failed: {exc}")
                continue
            meta = sample.get("meta", {})
            if meta.get("mask_kind") != "sral_tracks":
                continue
            if int(meta.get("sral_files_used", 0)) <= 0:
                continue
            mask = sample["obs_mask"]
            if not torch.is_tensor(mask):
                mask = torch.as_tensor(mask)
            count = float(mask.sum().item())
            if count > best_count:
                best_count = count
                best_mask = mask.detach().clone()
                if best_count >= 200.0:
                    break

        if best_mask is None:
            _debug("shared viz obs_mask: no real SRAL case found in scan; using per-case masks")
        else:
            _debug(f"shared viz obs_mask: selected mask with count={best_count:.0f}")
        self._shared_viz_obs_mask = best_mask
        return self._shared_viz_obs_mask

    def _apply_shared_viz_mask(self, cases: list[dict[str, torch.Tensor]]) -> None:
        if not cases:
            return
        shared_mask = self._ensure_shared_viz_obs_mask()
        if shared_mask is None:
            return
        shared_mask = shared_mask.to(device=self.accelerator.device, dtype=torch.float32)
        for case in cases:
            mask = shared_mask.unsqueeze(0).clone()
            case["obs_mask"] = mask
            case["obs_values"] = case["truth"] * mask

    @torch.no_grad()
    def _dashboard_cases(self) -> list[dict[str, torch.Tensor]]:
        max_cases = max(_DASHBOARD_NUM_CASES, 0)
        if max_cases == 0:
            return []

        strided_dataset = self.dashboard_dataset
        if strided_dataset is None:
            strided_dataset = getattr(self.val_dataloader, "dataset", None)
        if strided_dataset is not None and hasattr(strided_dataset, "strided_case_indices"):
            strided_indices = strided_dataset.strided_case_indices(max_cases, stride_days=30)
            if strided_indices:
                from torch.utils.data import default_collate

                samples = [strided_dataset[index] for index in strided_indices]
                labels = [sample.get("meta", {}).get("target_date") for sample in samples]
                cases = self._dashboard_cases_from_raw_batch(default_collate(samples), labels=labels)
                self._apply_shared_viz_mask(cases)
                return cases

        cases = []
        for raw_batch in self.val_dataloader:
            cases.extend(self._dashboard_cases_from_raw_batch(raw_batch))
            if len(cases) >= max_cases:
                cases = cases[:max_cases]
                break
        self._apply_shared_viz_mask(cases)
        return cases

    def _dashboard_cases_from_raw_batch(self, raw_batch: dict, labels=None) -> list[dict[str, torch.Tensor]]:
        batch = self._batch_to_device(raw_batch)
        labels = list(labels or [])
        cases = []
        for batch_idx in range(batch["truth"].shape[0]):
            case = {key: value[batch_idx : batch_idx + 1] for key, value in batch.items()}
            if batch_idx < len(labels) and labels[batch_idx]:
                case["case_label"] = str(labels[batch_idx])
            cases.append(case)
        return cases

    @staticmethod
    def _dashboard_batch(cases: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        keys = ("truth", "background", "obs_values", "obs_mask", "valid_mask", "water_mask")
        return {key: torch.cat([case[key] for case in cases], dim=0) for key in keys}

    def _remember_dashboard_epoch(
        self,
        epoch: int,
        pages: list[dict],
        large_path: str | None = None,
        weights_label: str | None = None,
    ) -> None:
        max_history = max(_DASHBOARD_HISTORY_EPOCHS, 1)
        self.dashboard_history.insert(
            0,
            {
                "epoch": int(epoch),
                "pages": pages,
                "large_path": large_path,
                "weights_label": weights_label or self._sampling_weight_label(),
            },
        )
        del self.dashboard_history[max_history:]

    @staticmethod
    def _dashboard_slot_name(index: int) -> str:
        if index == 0:
            return "latest"
        return f"previous_{index}"

    def _report_dashboard_history(self) -> None:
        if self.clearml is None:
            return
        for slot_idx, entry in enumerate(self.dashboard_history):
            slot = self._dashboard_slot_name(slot_idx)
            epoch = int(entry["epoch"])
            weights_label = str(entry.get("weights_label", self._sampling_weight_label()))
            for page in entry["pages"]:
                page_idx = int(page["page_idx"])
                self.clearml.report_image(
                    title=f"dashboard/{slot}_conditioning_ablation/{weights_label}",
                    series=f"page_{page_idx:02d}",
                    path=page["path"],
                    iteration=epoch,
                )
            if entry.get("large_path"):
                self.clearml.report_image(
                    title=f"dashboard/{slot}_large_conditioned_unconditioned/{weights_label}",
                    series="case_0000",
                    path=entry["large_path"],
                    iteration=epoch,
                )

    @staticmethod
    def _chunks(values: list, size: int):
        size = max(int(size), 1)
        for start in range(0, len(values), size):
            yield start // size, values[start : start + size]

    @torch.no_grad()
    def report_unconditional_dashboard_sample(self, epoch: int):
        if self.config.training_objective == "structured_joint_state_flow":
            return
        if self.config.sample_every_n_epochs <= 0:
            return
        if epoch % self.config.sample_every_n_epochs != 0:
            return
        if _UNCONDITIONAL_DASHBOARD_EVERY_N_EPOCHS <= 0:
            return
        if epoch % _UNCONDITIONAL_DASHBOARD_EVERY_N_EPOCHS != 0:
            return

        weights_label = self._sampling_weight_label()
        _debug(f"unconditional dashboard sampling start epoch={epoch} weights={weights_label}")
        self.model.eval()
        cases = self._dashboard_cases()
        if not cases:
            _debug(f"unconditional dashboard skipped epoch={epoch}: no validation cases")
            return

        samples_dir = os.path.join(self.output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        dashboard_batch = self._dashboard_batch(cases[:1])
        zero_background = torch.zeros_like(dashboard_batch["background"])
        zero_obs_values = torch.zeros_like(dashboard_batch["obs_values"])
        zero_obs_mask = torch.zeros_like(dashboard_batch["obs_mask"])
        initial_noise = torch.randn_like(dashboard_batch["background"])

        with self._sampling_model() as sample_model:
            sampler = self._make_sampler(sample_model)
            unconditional = sampler.sample_conditioned(
                background=zero_background,
                background_mask=torch.zeros_like(dashboard_batch["background"]),
                obs_values=zero_obs_values,
                obs_mask=zero_obs_mask,
                water_mask=dashboard_batch["water_mask"],
                size=self.config.image_size,
                num_timesteps=self.config.num_sample_timesteps,
                device=self.accelerator.device,
                start_mode=self.config.sample_start_mode,
                start_noise_level=self.config.sample_start_noise_level,
                enforce_observations=self.config.sample_enforce_observations,
                valid_mask=dashboard_batch["valid_mask"],
                obs_guidance_scale=self.config.sample_obs_guidance_scale,
                obs_guidance_eps=self.config.sample_obs_guidance_eps,
                initial_noise=initial_noise,
                sample_target=self._sample_target(),
                **self._cfg_sampler_kwargs(),
                **self._sample_solver_kwargs(),
            )

        unconditional_physical = self._physical_metric_tensor(unconditional)[0].detach().cpu()
        water_mask = dashboard_batch["water_mask"][0].detach().cpu()
        valid_mask = dashboard_batch["valid_mask"][0].detach().cpu()
        display_mask = water_mask if water_mask.shape[0] == 1 else valid_mask

        import matplotlib.pyplot as plt
        import numpy as np

        n_channels = int(unconditional_physical.shape[0])
        fig, axes = plt.subplots(
            1,
            n_channels,
            figsize=(max(n_channels, 1) * _UNCONDITIONAL_PANEL_WIDTH, _UNCONDITIONAL_PANEL_HEIGHT),
            squeeze=False,
            constrained_layout=True,
        )
        fig.suptitle(
            f"fully unconditional sample, epoch {epoch}, {weights_label}",
            fontsize=16,
        )

        for channel in range(n_channels):
            field_name = self.fields[channel] if channel < len(self.fields) else f"ch{channel}"
            values = unconditional_physical[channel].numpy()
            if display_mask.ndim == 3:
                if display_mask.shape[0] == 1:
                    channel_mask = display_mask[0].numpy() > 0
                else:
                    channel_mask = display_mask[channel].numpy() > 0
                values = np.where(channel_mask, values, np.nan)

            if channel == 0:
                cmap, vmin, vmax = "Blues_r", 0.0, 1.0
            else:
                cmap = "viridis"
                finite = values[np.isfinite(values)]
                if finite.size:
                    vmin = float(np.nanpercentile(finite, 1.0))
                    vmax = float(np.nanpercentile(finite, 99.0))
                    if vmin == vmax:
                        vmin, vmax = None, None
                else:
                    vmin, vmax = None, None

            ax = axes[0, channel]
            image = ax.imshow(
                np.ma.masked_invalid(values),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_title(field_name, fontsize=14)
            ax.axis("off")
            fig.colorbar(image, ax=ax, fraction=0.035, pad=0.015)

        path = os.path.join(samples_dir, f"epoch_{epoch:04d}_dashboard_unconditional_large.png")
        fig.savefig(path, dpi=_DASHBOARD_DPI, bbox_inches="tight")
        plt.close(fig)

        if self.clearml is not None:
            self.clearml.report_image(
                title=f"dashboard/unconditional_large/{weights_label}",
                series="case_0000",
                path=path,
                iteration=epoch,
            )
        _debug(f"unconditional dashboard saved: {path}")

    @torch.no_grad()
    def report_dashboard_samples(self, epoch: int):
        if self.config.sample_every_n_epochs <= 0:
            return
        if self.config.training_objective == "structured_joint_state_flow":
            if (epoch + 1) % self.config.sample_every_n_epochs == 0:
                self._report_structured_trajectory_dashboard(epoch)
            return
        if epoch % self.config.sample_every_n_epochs != 0:
            return
        if _DASHBOARD_EVERY_N_EPOCHS <= 0:
            return
        if epoch % _DASHBOARD_EVERY_N_EPOCHS != 0:
            return
        weights_label = self._sampling_weight_label()
        _debug(f"dashboard sampling start epoch={epoch} weights={weights_label}")
        if self.clearml is not None:
            self.clearml.report_single_value("sampling_use_ema", float(bool(self.config.sample_use_ema)))
            self.clearml.report_single_value("ema_decay", float(self.config.ema_decay))
        self.model.eval()
        cases = self._dashboard_cases()
        if not cases:
            _debug(f"dashboard sampling skipped epoch={epoch}: no validation cases")
            return
        samples_dir = os.path.join(self.output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        dashboard_cases = []
        dashboard_batch = self._dashboard_batch(cases)

        with self._sampling_model() as sample_model:
            sampler = self._make_sampler(sample_model)
            initial_noise = torch.randn_like(dashboard_batch["background"])
            zero_background = torch.zeros_like(dashboard_batch["background"])
            zero_obs_values = torch.zeros_like(dashboard_batch["obs_values"])
            zero_obs_mask = torch.zeros_like(dashboard_batch["obs_mask"])
            sample_target = self._sample_target()

            def sample_variant(background, background_mask, obs_values, obs_mask):
                return sampler.sample_conditioned(
                    background=background,
                    background_mask=background_mask,
                    obs_values=obs_values,
                    obs_mask=obs_mask,
                    water_mask=dashboard_batch["water_mask"],
                    size=self.config.image_size,
                    num_timesteps=self.config.num_sample_timesteps,
                    device=self.accelerator.device,
                    start_mode=self.config.sample_start_mode,
                    start_noise_level=self.config.sample_start_noise_level,
                    enforce_observations=self.config.sample_enforce_observations,
                    valid_mask=dashboard_batch["valid_mask"],
                    obs_guidance_scale=self.config.sample_obs_guidance_scale,
                    obs_guidance_eps=self.config.sample_obs_guidance_eps,
                    initial_noise=initial_noise,
                    sample_target=sample_target,
                    **self._cfg_sampler_kwargs(),
                    **self._sample_solver_kwargs(),
                )

            assim_batch = sample_variant(
                dashboard_batch["background"],
                torch.ones_like(dashboard_batch["background"]),
                dashboard_batch["obs_values"],
                dashboard_batch["obs_mask"],
            )
            assim_background_only_batch = sample_variant(
                dashboard_batch["background"],
                torch.ones_like(dashboard_batch["background"]),
                zero_obs_values,
                zero_obs_mask,
            )
            assim_observation_only_batch = sample_variant(
                zero_background,
                torch.zeros_like(dashboard_batch["background"]),
                dashboard_batch["obs_values"],
                dashboard_batch["obs_mask"],
            )
            assim_neither_batch = sample_variant(
                zero_background,
                torch.zeros_like(dashboard_batch["background"]),
                zero_obs_values,
                zero_obs_mask,
            )

            for case_idx, one in enumerate(cases):
                assim = assim_batch[case_idx]
                assim_physical = self._physical_metric_tensor(assim.unsqueeze(0))[0]
                truth_physical = self._physical_metric_tensor(one["truth"])[0]
                background_physical = self._physical_metric_tensor(one["background"])[0]

                if self.clearml is not None:
                    for channel, field_name in enumerate(self.fields):
                        valid_mask = one["valid_mask"][0, channel : channel + 1]
                        analysis_mae, analysis_rmse = self._masked_error_metrics(
                            assim_physical[channel : channel + 1],
                            truth_physical[channel : channel + 1],
                            valid_mask,
                        )
                        background_mae, background_rmse = self._masked_error_metrics(
                            background_physical[channel : channel + 1],
                            truth_physical[channel : channel + 1],
                            valid_mask,
                        )
                        skill = 0.0 if background_rmse <= 0.0 else 1.0 - analysis_rmse / background_rmse
                        series = f"{field_name}/case_{case_idx:04d}"
                        self.clearml.report_scalar(
                            "sample_physical/rmse_analysis", series, analysis_rmse, epoch
                        )
                        self.clearml.report_scalar(
                            "sample_physical/rmse_background", series, background_rmse, epoch
                        )
                        self.clearml.report_scalar(
                            "sample_physical/mae_analysis", series, analysis_mae, epoch
                        )
                        self.clearml.report_scalar(
                            "sample_physical/mae_background", series, background_mae, epoch
                        )
                        self.clearml.report_scalar("sample_physical/rmse_skill", series, skill, epoch)

                dashboard_cases.append(
                    {
                        "case_idx": case_idx,
                        "case_label": one.get("case_label"),
                        "background": one["background"][0],
                        "obs_values": one["obs_values"][0],
                        "obs_mask": one["obs_mask"][0],
                        "assim": assim,
                        "assim_background_only": assim_background_only_batch[case_idx],
                        "assim_observation_only": assim_observation_only_batch[case_idx],
                        "assim_neither": assim_neither_batch[case_idx],
                        "truth": one["truth"][0],
                        "valid_mask": one["valid_mask"][0],
                        "water_mask": one["water_mask"][0],
                    }
                )

        import matplotlib.pyplot as plt

        current_pages = []
        cases_per_page = max(_DASHBOARD_CASES_PER_PAGE, 1)
        for page_idx, page_cases in self._chunks(dashboard_cases, cases_per_page):
            title = (
                f"conditioning ablation, shared initial noise, epoch {epoch}, "
                f"page {page_idx}, cases {len(page_cases)}, {weights_label}"
            )
            fig = make_multi_case_background_condition_assim_figure(
                cases=page_cases,
                fields=self.fields,
                means=self.channel_means,
                stds=self.channel_stds,
                channels=_DASHBOARD_CHANNELS,
                title=title,
                panel_width=_DASHBOARD_PANEL_WIDTH,
                panel_height=_DASHBOARD_PANEL_HEIGHT,
            )
            figure_path = os.path.join(
                samples_dir,
                f"epoch_{epoch:04d}_dashboard_page_{page_idx:02d}_conditioning_ablation.png",
            )
            fig.savefig(figure_path, dpi=_DASHBOARD_DPI, bbox_inches="tight")
            plt.close(fig)
            current_pages.append({"page_idx": page_idx, "path": figure_path})

        large_case = {
            key: value
            for key, value in dashboard_cases[0].items()
            if key not in ("assim_background_only", "assim_observation_only")
        }
        large_fig = make_multi_case_background_condition_assim_figure(
            cases=[large_case],
            fields=self.fields,
            means=self.channel_means,
            stds=self.channel_stds,
            channels=range(len(self.fields)),
            title=(f"large fully conditioned vs fully unconditioned sample, epoch {epoch}, {weights_label}"),
            panel_width=_DASHBOARD_LARGE_PANEL_WIDTH,
            panel_height=_DASHBOARD_LARGE_PANEL_HEIGHT,
        )
        large_path = os.path.join(
            samples_dir, f"epoch_{epoch:04d}_dashboard_large_conditioned_unconditioned.png"
        )
        large_fig.savefig(large_path, dpi=_DASHBOARD_DPI, bbox_inches="tight")
        plt.close(large_fig)

        self._remember_dashboard_epoch(
            epoch,
            current_pages,
            large_path=large_path,
            weights_label=weights_label,
        )
        self._report_dashboard_history()
        _debug(f"dashboard sampling done epoch={epoch}")

    @torch.no_grad()
    def _report_structured_trajectory_dashboard(self, epoch: int) -> None:
        with self._structured_diagnostic_rng(epoch, stream_index=41):
            self._report_structured_trajectory_dashboard_isolated(epoch)

    @torch.no_grad()
    def _report_structured_trajectory_dashboard_isolated(self, epoch: int) -> None:
        self.model.eval()
        try:
            raw_batch = next(iter(self.val_dataloader))
        except StopIteration:
            return
        batch = self._batch_to_device(raw_batch)
        one = {key: value[:1] for key, value in batch.items()}
        sample_seed = self._structured_diagnostic_seed(epoch, case_index=0, member_index=0, stream_index=23)
        generator = torch.Generator(device=self.accelerator.device)
        generator.manual_seed(sample_seed)
        initial_noise = torch.randn(
            (1, self.config.out_channels, *self.config.image_size),
            dtype=one["background"].dtype,
            device=self.accelerator.device,
            generator=generator,
        )
        with self._sampling_model() as sample_model:
            try:
                sample = sample_structured_batch(
                    self._make_sampler(sample_model),
                    one,
                    stats=self.config.structured_state_stats,
                    size=self.config.image_size,
                    num_timesteps=self.config.num_sample_timesteps,
                    device=self.accelerator.device,
                    method=self.config.sample_method,
                    rtol=self.config.sample_rtol,
                    atol=self.config.sample_atol,
                    initial_noise=initial_noise,
                )[0]
            except StructuredDecodeSaturationError as error:
                raise StructuredDecodeSaturationError(
                    f"dashboard epoch={epoch}: {error}",
                    {
                        **error.diagnostics,
                        "epoch": int(epoch),
                        "case_index": 0,
                        "member_index": 0,
                        "member_seed": int(sample_seed),
                    },
                ) from error
        samples_dir = os.path.join(self.output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        weights_label = self._sampling_weight_label()
        figure = make_structured_trajectory_figure(
            one["structured_physical_truth"][0],
            one["structured_physical_background"][0],
            sample,
            one["valid_mask"][0],
            title=(
                "joint SIC/SIT trajectory "
                f"leads={list(self.config.trajectory_lead_days)}, epoch {epoch}, {weights_label}"
            ),
            lead_days=self.config.trajectory_lead_days,
        )
        path = os.path.join(samples_dir, f"epoch_{epoch:04d}_structured_trajectory.png")
        figure.savefig(path, dpi=_DASHBOARD_DPI, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(figure)
        torch.save(
            {
                "sample": sample.detach().cpu(),
                "truth": one["structured_physical_truth"][0].detach().cpu(),
                "background": one["structured_physical_background"][0].detach().cpu(),
                "valid_mask": one["valid_mask"][0].detach().cpu(),
                "lag0_mask": one["structured_lag0_mask"][0].detach().cpu(),
                "values_space": "physical",
            },
            os.path.join(samples_dir, f"epoch_{epoch:04d}_structured_trajectory.pt"),
        )
        if self.clearml is not None:
            self.clearml.report_image(
                title="dashboard/structured_joint_trajectory",
                series="truth_background_member",
                path=path,
                iteration=epoch,
            )
        _debug(f"structured trajectory dashboard saved: {path}")

    def train_loop(self):
        start_epoch, global_step = self._training_loop_start()
        if not 0 <= start_epoch <= self.config.num_epochs or global_step < 0:
            raise ValueError("training loop start state is invalid")
        _debug(f"training start epochs={self.config.num_epochs} train_batches={len(self.train_dataloader)}")
        for epoch in range(start_epoch, self.config.num_epochs):
            train_dataset = getattr(self.train_dataloader, "dataset", None)
            if hasattr(train_dataset, "set_epoch"):
                train_dataset.set_epoch(epoch)
            self.model.train()
            _debug(f"epoch {epoch} start")
            progress_bar = tqdm(
                total=len(self.train_dataloader), disable=not self.accelerator.is_local_main_process
            )
            progress_bar.set_description(f"Epoch {epoch}")

            for batch_index, raw_batch in enumerate(self.train_dataloader):
                batch = self._batch_to_device(raw_batch)
                truth = batch["truth"]
                batch_size = truth.shape[0]
                timesteps = self._sample_timesteps(batch_size)
                background, background_mask, obs_values, obs_mask = self._conditioned_inputs(batch)
                model_state, v_real = self._make_training_pair(
                    truth, batch, timesteps, residual_background=background
                )

                with self.accelerator.accumulate(self.model):
                    model_input = self._make_model_input(
                        self._structured_model_state(model_state, timesteps),
                        batch,
                        background=background,
                        background_mask=background_mask,
                        obs_values=obs_values,
                        obs_mask=obs_mask,
                    )
                    model_output = self.model(model_input, timesteps * 1000, return_dict=False)[0]
                    v_pred = self._reconstruct_model_velocity(model_output, model_state, timesteps)
                    loss_full = self._flow_matching_loss(v_pred, v_real, batch)
                    loss_obs = torch.zeros_like(loss_full)
                    loss_smooth = torch.zeros_like(loss_full)
                    loss = loss_full
                    case_ids = raw_batch.get("meta", {}).get("case_id", "unavailable")
                    provenance = (
                        f"epoch={epoch} batch_index={batch_index} global_step={global_step} "
                        f"case_id={case_ids!r}"
                    )
                    _require_finite_loss(loss, provenance)
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                        _require_finite_gradients(self.model, provenance)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.ema_model.step(self.unwrapped_model.parameters())

                if batch_index == 0:
                    _debug(
                        f"epoch {epoch} first batch "
                        f"loss={loss.item():.6f} full={loss_full.item():.6f} "
                        f"obs={loss_obs.item():.6f} smooth={loss_smooth.item():.6f}"
                    )
                    with self._structured_diagnostic_rng(epoch, stream_index=31):
                        self._report_condition_diagnostics(raw_batch, global_step, "train")
                self._report_train_metrics(loss, loss_full, loss_obs, loss_smooth, global_step)
                if self.config.tracker:
                    self.accelerator.log(
                        {
                            "train_loss": loss.item(),
                            "train_loss_full": loss_full.item(),
                            "train_loss_obs": loss_obs.item(),
                            "train_loss_smoothness": loss_smooth.item(),
                        },
                        step=global_step,
                    )
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item(), obs=loss_obs.item())

            progress_bar.close()
            val_loss_full, val_loss_obs, val_loss = self.compute_val_loss()
            if not all(math.isfinite(value) for value in (val_loss_full, val_loss_obs, val_loss)):
                raise FloatingPointError(
                    f"non-finite validation loss at epoch={epoch} global_step={global_step}"
                )
            history_entry = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_loss_full": val_loss_full,
                "val_loss_obs": val_loss_obs,
                "step": global_step,
                "timestamp": datetime.now().isoformat(),
            }
            self.val_history.append(history_entry)
            if self.config.tracker:
                self.accelerator.log(
                    {
                        "val_loss": val_loss,
                        "val_loss_full": val_loss_full,
                        "val_loss_obs": val_loss_obs,
                    },
                    step=global_step,
                )
            self._report_val_metrics(val_loss_full, val_loss_obs, val_loss, global_step)
            _debug(f"epoch {epoch} validation total={val_loss:.6f}")
            logger.info(
                "Epoch %d val_loss %.6f full %.6f obs %.6f", epoch, val_loss, val_loss_full, val_loss_obs
            )

            if self.accelerator.is_main_process:
                _atomic_json(Path(self.output_dir) / "metrics.json", self.val_history)
                _debug("saved metrics.json")
                self.save_model_custom("last_model.pth")
                _debug("saved last checkpoint")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_model_custom("best_model.pth")
                    _debug("saved best checkpoint")
                    if self.clearml is not None:
                        self.clearml.report_single_value("best_val_loss", val_loss)
                # Commit the complete train state before any sampling, plotting, or
                # external reporting that may fail for reasons unrelated to optimization.
                self._save_structured_recovery(epoch + 1, global_step)

                with self._structured_diagnostic_rng(epoch, stream_index=37):
                    sample_metrics = {}
                    structured = self.config.training_objective == "structured_joint_state_flow"
                    metric_due = self.config.metric_every_n_epochs > 0 and (
                        (epoch + 1) % self.config.metric_every_n_epochs == 0
                        if structured
                        else epoch % self.config.metric_every_n_epochs == 0
                    )
                    dashboard_due = self.config.sample_every_n_epochs > 0 and (
                        (epoch + 1) % self.config.sample_every_n_epochs == 0
                        if structured
                        else (
                            _DASHBOARD_EVERY_N_EPOCHS > 0
                            and epoch % self.config.sample_every_n_epochs == 0
                            and epoch % _DASHBOARD_EVERY_N_EPOCHS == 0
                        )
                    )
                    structured_sampling_due = structured and (metric_due or dashboard_due)
                    try:
                        self._report_condition_diagnostics(
                            next(iter(self.val_dataloader)), global_step, "validation"
                        )
                    except StopIteration:
                        pass
                    if structured_sampling_due and global_step < self.config.diagnostic_min_optimizer_steps:
                        history_entry["structured_sampling_diagnostic"] = {
                            "status": "skipped_warmup",
                            "epoch": int(epoch),
                            "global_step": int(global_step),
                            "minimum_step": int(self.config.diagnostic_min_optimizer_steps),
                            "partial_metrics_permitted": False,
                        }
                        _atomic_json(Path(self.output_dir) / "metrics.json", self.val_history)
                    else:
                        try:
                            if metric_due:
                                sample_metrics = self.compute_sample_validation_metrics(epoch=epoch)
                            self.report_unconditional_dashboard_sample(epoch)
                            self.report_dashboard_samples(epoch)
                        except StructuredDecodeSaturationError as error:
                            if not structured:
                                raise
                            history_entry["structured_sampling_diagnostic"] = {
                                "status": "failed_decode_saturation",
                                "epoch": int(epoch),
                                "global_step": int(global_step),
                                "checkpoint": self.config.recovery_checkpoint_name,
                                "error": str(error),
                                "diagnostics": jsonable(error.diagnostics),
                                "partial_metrics_permitted": False,
                                "optimization_continues": True,
                                "publication_ready": False,
                            }
                            _atomic_json(Path(self.output_dir) / "metrics.json", self.val_history)
                            if self.clearml is not None:
                                self.clearml.report_single_value("structured_sampling_diagnostic_valid", 0.0)
                            _debug(
                                "structured sampling diagnostic failed nonfatally "
                                f"at epoch={epoch} step={global_step}: {error}"
                            )
                        else:
                            if metric_due:
                                history_entry.update(sample_metrics)
                                self._report_sample_validation_metrics(sample_metrics, global_step)
                            if structured_sampling_due:
                                history_entry["structured_sampling_diagnostic"] = {
                                    "status": "passed",
                                    "epoch": int(epoch),
                                    "global_step": int(global_step),
                                    "partial_metrics_permitted": False,
                                }
                                if self.clearml is not None:
                                    self.clearml.report_single_value(
                                        "structured_sampling_diagnostic_valid", 1.0
                                    )
                            if metric_due or structured_sampling_due:
                                _atomic_json(
                                    Path(self.output_dir) / "metrics.json",
                                    self.val_history,
                                )
            self._after_training_epoch(epoch, global_step)

        if self.accelerator.is_main_process:
            if self.config.training_objective == "structured_joint_state_flow":
                diagnostic_records = [
                    entry["structured_sampling_diagnostic"]
                    for entry in self.val_history
                    if "structured_sampling_diagnostic" in entry
                ]
                physical_sampling_validated = _latest_structured_sampling_valid(diagnostic_records)
                completion = {
                    "optimization_complete": True,
                    "completed_optimizer_steps": int(global_step),
                    "physical_sampling_validated": physical_sampling_validated,
                    "diagnostic_records": diagnostic_records,
                    "publication_ready": False,
                    "publication_blocker": (
                        "standalone frozen publication evaluation and equal-information "
                        "baselines remain required"
                    ),
                }
                _atomic_json(
                    Path(self.output_dir) / "structured_training_completion.json",
                    completion,
                )
            _debug("uploading ClearML artifacts")
            self._upload_clearml_artifacts()
            if self.clearml is not None:
                self.clearml.close()
        _debug("training finished")
        self.accelerator.end_training()
        return self.output_dir
