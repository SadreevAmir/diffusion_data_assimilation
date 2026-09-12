import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from diffusers.training_utils import EMAModel
from torch import nn

from assim_lib.config import TrainingConfig
from assim_lib.direct_dynamics_cascade import masked_block_average, project_detail
from assim_lib.direct_dynamics_cascade_contract import forecast_contract_sha256
from assim_lib.direct_dynamics_cascade_fine import generated_coarse_condition
from assim_lib.direct_dynamics_cascade_fine import (
    ProjectedDetailModel,
    load_fine_cascade_sampler,
    write_fine_manifest,
)
from assim_lib.direct_dynamics_cascade_fine_preconditioned import (
    PRECONDITIONING_KIND,
    VariancePreconditionedFineCascadeDynamicsTrainer,
    VariancePreconditionedFineCascadeSampler,
    _PreconditionedVelocityModel,
    preconditioned_flow_target,
    preconditioned_model_state,
    preconditioning_coefficients,
    reconstruct_preconditioned_velocity,
    load_variance_preconditioned_fine_cascade_sampler,
    validate_preconditioning_contract,
    write_preconditioning_manifest,
)
from assim_lib.model_io import build_unet


SIGMA = (0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
BETA = (0.81, 0.82, 0.83, 0.84, 0.85, 0.86)


class ZeroNeuralBranch(nn.Module):
    def forward(self, model_input, timestep, return_dict=False):
        del timestep, return_dict
        return (torch.zeros_like(model_input[:, :6]),)


class EchoNeuralBranch(nn.Module):
    def forward(self, model_input, timestep, return_dict=False):
        del timestep, return_dict
        return (model_input[:, :6],)


class VariancePreconditionedFineTests(unittest.TestCase):
    def test_coefficients_have_exact_gaussian_endpoints(self) -> None:
        state = torch.ones((2, 6, 8, 10), dtype=torch.float64)
        sigma = torch.tensor(SIGMA, dtype=torch.float64).view(1, 6, 1, 1)
        beta = torch.tensor(BETA, dtype=torch.float64).view(1, 6, 1, 1)
        for time_value, expected_scale, expected_analytic, expected_neural in (
            (0.0, sigma, -torch.ones_like(sigma), beta),
            (1.0, beta, torch.ones_like(beta), sigma),
        ):
            scale, analytic, neural = preconditioning_coefficients(
                state, torch.full((2,), time_value), SIGMA, BETA
            )
            self.assertTrue(torch.allclose(scale, expected_scale.expand_as(scale), atol=1e-14))
            self.assertTrue(
                torch.allclose(analytic, expected_analytic.expand_as(analytic), atol=1e-14)
            )
            self.assertTrue(
                torch.allclose(neural, expected_neural.expand_as(neural), atol=1e-14)
            )

    def test_target_reconstructs_the_original_flow_velocity(self) -> None:
        generator = torch.Generator().manual_seed(31)
        state = torch.randn((3, 6, 8, 10), generator=generator, dtype=torch.float64)
        velocity = torch.randn((3, 6, 8, 10), generator=generator, dtype=torch.float64)
        time = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
        target = preconditioned_flow_target(velocity, state, time, SIGMA, BETA)
        reconstructed, _ = reconstruct_preconditioned_velocity(
            target, state, time, SIGMA, BETA
        )
        self.assertTrue(torch.allclose(reconstructed, velocity, atol=2e-14, rtol=0))
        normalized = preconditioned_model_state(state, time, SIGMA, BETA)
        self.assertTrue(torch.isfinite(normalized).all())

    def test_zero_neural_oracle_transports_base_scale_to_target_scale(self) -> None:
        generator = torch.Generator().manual_seed(37)
        raw = torch.randn((1, 6, 12, 10), generator=generator)
        valid = torch.ones((1, 1, 12, 10))
        causal = torch.zeros((1, 15, 12, 10))
        causal[:, 2:3] = valid
        condition = generated_coarse_condition(
            causal, torch.zeros((1, 6, 6, 5)), valid
        )
        zeros = torch.zeros((1, 6, 12, 10))
        sampler = VariancePreconditionedFineCascadeSampler(
            ZeroNeuralBranch(), SIGMA, BETA
        )
        result = sampler.sample_conditioned(
            background=zeros,
            background_mask=torch.ones_like(zeros),
            obs_values=zeros[:, :2],
            obs_mask=zeros[:, :2],
            water_mask=valid,
            valid_mask=valid,
            state_mask=valid.expand_as(zeros),
            size=(12, 10),
            num_timesteps=33,
            device=torch.device("cpu"),
            method="rk4",
            start_mode="noise",
            initial_noise=raw,
            sample_target="state",
            model_conditioning=condition,
            state_channels=6,
            end_time=0.0,
        )
        projected = project_detail(raw, valid)
        expected = projected * (
            torch.tensor(SIGMA).view(1, 6, 1, 1)
            / torch.tensor(BETA).view(1, 6, 1, 1)
        )
        self.assertTrue(torch.allclose(result, expected, atol=3e-5, rtol=3e-4))
        coarse, fraction = masked_block_average(result, valid)
        self.assertLess(
            float(coarse[fraction.expand_as(coarse) > 0].abs().max()), 2e-6
        )

    def test_sampler_adapter_matches_training_parameterization_and_named_mask(self) -> None:
        generator = torch.Generator().manual_seed(41)
        state = project_detail(
            torch.randn((2, 6, 12, 10), generator=generator),
            torch.ones((2, 1, 12, 10)),
        )
        time = torch.tensor([0.2, 0.8])
        model_input = torch.zeros((2, 29, 12, 10))
        model_input[:, :6] = state
        model_input[:, 10:11] = 1
        adapter = _PreconditionedVelocityModel(EchoNeuralBranch(), SIGMA, BETA)
        actual = adapter(model_input, time * 1000)[0]
        normalized = preconditioned_model_state(state, time, SIGMA, BETA)
        expected, _ = reconstruct_preconditioned_velocity(
            project_detail(normalized, model_input[:, 10:11]),
            state,
            time,
            SIGMA,
            BETA,
        )
        self.assertTrue(torch.allclose(actual, expected, atol=3e-6, rtol=0))

    def test_actual_trainer_loss_backpropagates_through_neural_branch(self) -> None:
        generator = torch.Generator().manual_seed(43)
        valid = torch.ones((2, 1, 12, 10))
        state = project_detail(
            torch.randn((2, 6, 12, 10), generator=generator), valid
        )
        target_velocity = project_detail(
            torch.randn((2, 6, 12, 10), generator=generator), valid
        )
        time = torch.tensor([0.25, 0.75])
        model = nn.Conv2d(6, 6, 1)
        trainer = VariancePreconditionedFineCascadeDynamicsTrainer.__new__(
            VariancePreconditionedFineCascadeDynamicsTrainer
        )
        trainer._target_residual_rms = SIGMA
        trainer._projected_base_rms = BETA
        trainer._loss_inverse_neural_scale = None
        model_state = trainer._structured_model_state(state, time)
        neural = project_detail(model(model_state), valid)
        predicted_velocity = trainer._reconstruct_model_velocity(
            neural, state, time
        )
        loss = trainer._flow_matching_loss(
            predicted_velocity, target_velocity, {"valid_mask": valid}
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.weight.grad.abs().sum()), 0.0)
        self.assertIsNone(trainer._loss_inverse_neural_scale)

    def test_contract_requires_hashed_train_only_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            audit = {
                "split": "train_only",
                "selected_case_ids_sha256": "case-hash",
                "measures": {
                    "target": {"rms": list(SIGMA)},
                    "base": {"rms": list(BETA)},
                },
            }
            path.write_text(json.dumps(audit), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            contract = {
                "kind": PRECONDITIONING_KIND,
                "target_residual_rms": list(SIGMA),
                "projected_base_rms": list(BETA),
                "source_audit_path": str(path),
                "source_audit_sha256": digest,
            }
            validated = validate_preconditioning_contract(contract)
            self.assertEqual(validated["source_split"], "train_only")
            self.assertEqual(validated["source_case_ids_sha256"], "case-hash")
            broken = dict(contract, source_audit_sha256="0" * 64)
            with self.assertRaises(ValueError):
                validate_preconditioning_contract(broken)
            audit["split"] = "valid"
            path.write_text(json.dumps(audit), encoding="utf-8")
            wrong_split = dict(
                contract,
                source_audit_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            with self.assertRaises(ValueError):
                validate_preconditioning_contract(wrong_split)

    def test_raw_and_ema_reload_cannot_silently_use_plain_velocity(self) -> None:
        config = TrainingConfig(
            image_size=(320, 256),
            in_channels=29,
            out_channels=6,
            block_out_channels=(8,),
            down_block_types=("DownBlock2D",),
            up_block_types=("UpBlock2D",),
            norm_num_groups=4,
            add_attention=False,
            clearml_enabled=False,
        )
        model = build_unet(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_path = root / "audit.json"
            audit = {
                "split": "train_only",
                "selected_case_ids_sha256": "case-hash",
                "measures": {
                    "target": {"rms": list(SIGMA)},
                    "base": {"rms": list(BETA)},
                },
            }
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            contract = validate_preconditioning_contract(
                {
                    "kind": PRECONDITIONING_KIND,
                    "target_residual_rms": list(SIGMA),
                    "projected_base_rms": list(BETA),
                    "source_audit_path": str(audit_path),
                    "source_audit_sha256": hashlib.sha256(
                        audit_path.read_bytes()
                    ).hexdigest(),
                }
            )
            forecast_contract = {"unit_test_contract": "preconditioned-fine"}
            forecast_sha256 = forecast_contract_sha256(forecast_contract)
            commit = "b" * 40
            write_fine_manifest(root, config, {"git_commit": commit}, forecast_contract)
            write_preconditioning_manifest(root, contract, commit)
            torch.save(model.state_dict(), root / "raw.pth")
            ema = EMAModel(model.parameters())
            torch.save(ema.state_dict(), root / "ema.pth")
            model_config = dict(config.__dict__)
            valid = torch.ones((1, 1, 8, 8))
            causal = torch.zeros((1, 15, 8, 8))
            causal[:, 2:3] = valid
            condition = generated_coarse_condition(
                causal, torch.zeros((1, 6, 4, 4)), valid
            )
            initial_noise = torch.randn(
                (1, 6, 8, 8), generator=torch.Generator().manual_seed(47)
            )
            zeros = torch.zeros((1, 6, 8, 8))
            sample_kwargs = {
                "background": zeros,
                "background_mask": torch.ones_like(zeros),
                "obs_values": zeros[:, :2],
                "obs_mask": zeros[:, :2],
                "water_mask": valid,
                "valid_mask": valid,
                "state_mask": valid.expand_as(zeros),
                "size": (8, 8),
                "num_timesteps": 3,
                "device": torch.device("cpu"),
                "method": "rk4",
                "start_mode": "noise",
                "initial_noise": initial_noise,
                "sample_target": "state",
                "model_conditioning": condition,
                "state_channels": 6,
                "end_time": 0.0,
            }
            expected = VariancePreconditionedFineCascadeSampler(
                model, SIGMA, BETA
            ).sample_conditioned(**sample_kwargs)
            for checkpoint in ("raw.pth", "ema.pth"):
                checkpoint_sha256 = hashlib.sha256(
                    (root / checkpoint).read_bytes()
                ).hexdigest()
                loaded = load_variance_preconditioned_fine_cascade_sampler(
                    directory,
                    checkpoint,
                    model_config,
                    checkpoint_sha256,
                    commit,
                    forecast_sha256,
                    device=torch.device("cpu"),
                )
                self.assertIsInstance(
                    loaded, VariancePreconditionedFineCascadeSampler
                )
                actual = loaded.sample_conditioned(**sample_kwargs)
                self.assertTrue(
                    torch.allclose(actual, expected, atol=2e-6, rtol=0)
                )
                with self.assertRaisesRegex(
                    ValueError, "velocity parameterization"
                ):
                    load_fine_cascade_sampler(
                        directory,
                        checkpoint,
                        model_config,
                        checkpoint_sha256,
                        commit,
                        forecast_sha256,
                        device=torch.device("cpu"),
                    )


if __name__ == "__main__":
    unittest.main()
