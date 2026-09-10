import unittest

import torch
from torch import nn

from assim_lib.direct_dynamics_cascade import masked_block_average, project_detail
from assim_lib.direct_dynamics_cascade_coarse import CoarseCascadeSampler, coarse_target
from assim_lib.direct_dynamics_cascade_end_to_end import CascadePredictor
from assim_lib.direct_dynamics_cascade_fine import FineCascadeSampler


class ConstantVelocityModel(nn.Module):
    def __init__(self, velocity: torch.Tensor):
        super().__init__()
        self.register_buffer("velocity", velocity)

    def forward(self, model_input, timestep, return_dict=False):
        del timestep, return_dict
        return (self.velocity.expand(model_input.shape[0], -1, -1, -1),)


class ZeroModel(nn.Module):
    def forward(self, model_input, timestep, return_dict=False):
        del timestep, return_dict
        return (torch.zeros_like(model_input[:, :6]),)


class EndToEndCascadeTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(712)
        self.condition = torch.randn(2, 15, 8, 10, generator=generator)
        self.mask = torch.ones(2, 1, 8, 10)
        self.mask[:, :, :2, :2] = 0
        self.condition[:, 2:3] = self.mask
        self.condition[:, :2] *= self.mask
        self.condition[:, 7:11] = 1.0 * self.mask
        self.condition[:, 3:7] *= self.condition[:, 7:11]
        for channel in range(11, 15):
            self.condition[:, channel] = self.condition[:, channel, :1, :1]
        self.truth = torch.randn(2, 6, 8, 10, generator=generator)
        self.case_ids = ("case-a", "case-b")

    def _predictor_for_member(self, member: int) -> CascadePredictor:
        # Reproduce the predictor's member-bound noise to construct exact oracle velocities.
        from assim_lib.direct_dynamics_cascade_end_to_end import _casewise_noise

        coarse_noise, _ = _casewise_noise(
            self.case_ids,
            member,
            "coarse",
            (6, 4, 5),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        coarse_truth, _, _ = coarse_target(self.truth, self.mask)
        coarse_model = ConstantVelocityModel(coarse_noise - coarse_truth)
        fine_noise, _ = _casewise_noise(
            self.case_ids,
            member,
            "fine",
            (6, 8, 10),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        fine_detail = project_detail(self.truth, self.mask)
        fine_model = ConstantVelocityModel(project_detail(fine_noise, self.mask) - fine_detail)
        return CascadePredictor(CoarseCascadeSampler(coarse_model), FineCascadeSampler(fine_model))

    def test_generated_coarse_to_fine_recovers_oracle_without_truth_input(self):
        result = self._predictor_for_member(3).sample_member(
            structured_conditioning=self.condition,
            valid_mask=self.mask,
            case_ids=self.case_ids,
            member=3,
            coarse_num_timesteps=5,
            fine_num_timesteps=5,
            device=torch.device("cpu"),
        )
        valid = self.mask.expand_as(self.truth) > 0
        self.assertTrue(torch.allclose(result.forecast[valid], self.truth[valid], atol=4e-5, rtol=0))
        recovered, _ = masked_block_average(result.forecast, self.mask)
        self.assertTrue(torch.allclose(recovered, result.coarse, atol=3e-6, rtol=0))
        residual_coarse, fraction = masked_block_average(result.residual, self.mask)
        self.assertLess(float(residual_coarse[fraction > 0].abs().max()), 3e-6)
        self.assertNotEqual(result.coarse_seeds, result.fine_seeds)

    def test_case_streams_are_batch_order_invariant(self):
        predictor = CascadePredictor(CoarseCascadeSampler(ZeroModel()), FineCascadeSampler(ZeroModel()))
        forward = predictor.sample_member(
            structured_conditioning=self.condition,
            valid_mask=self.mask,
            case_ids=self.case_ids,
            member=1,
            coarse_num_timesteps=3,
            fine_num_timesteps=3,
            device=torch.device("cpu"),
        )
        order = torch.tensor([1, 0])
        reverse = predictor.sample_member(
            structured_conditioning=self.condition[order],
            valid_mask=self.mask[order],
            case_ids=tuple(self.case_ids[index] for index in order.tolist()),
            member=1,
            coarse_num_timesteps=3,
            fine_num_timesteps=3,
            device=torch.device("cpu"),
        )
        for name in ("forecast", "coarse", "residual", "raw_coarse_noise", "raw_fine_noise"):
            self.assertTrue(torch.equal(getattr(forward, name)[order], getattr(reverse, name)))

        pieces = []
        for index, case_id in enumerate(self.case_ids):
            pieces.append(
                predictor.sample_member(
                    structured_conditioning=self.condition[index : index + 1],
                    valid_mask=self.mask[index : index + 1],
                    case_ids=(case_id,),
                    member=1,
                    coarse_num_timesteps=3,
                    fine_num_timesteps=3,
                    device=torch.device("cpu"),
                )
            )
        for name in ("forecast", "coarse", "residual"):
            combined = torch.cat([getattr(item, name) for item in pieces])
            self.assertTrue(torch.equal(getattr(forward, name), combined))

    def test_ensemble_member_indices_are_chunk_invariant(self):
        predictor = CascadePredictor(CoarseCascadeSampler(ZeroModel()), FineCascadeSampler(ZeroModel()))
        kwargs = dict(
            structured_conditioning=self.condition[:1],
            valid_mask=self.mask[:1],
            case_ids=self.case_ids[:1],
            coarse_num_timesteps=2,
            fine_num_timesteps=2,
            device=torch.device("cpu"),
        )
        joint = predictor.sample_ensemble(member_indices=(2, 5), **kwargs)
        left = predictor.sample_ensemble(member_indices=(2,), **kwargs)
        right = predictor.sample_ensemble(member_indices=(5,), **kwargs)
        for name in ("forecast", "coarse", "residual"):
            self.assertTrue(torch.equal(joint[name], torch.cat((left[name], right[name]), dim=1)))

    def test_rejects_nonzero_flow_endpoint(self):
        predictor = CascadePredictor(CoarseCascadeSampler(ZeroModel()), FineCascadeSampler(ZeroModel()))
        with self.assertRaisesRegex(ValueError, "end_time=0"):
            predictor.sample_member(
                structured_conditioning=self.condition[:1],
                valid_mask=self.mask[:1],
                case_ids=self.case_ids[:1],
                member=0,
                coarse_num_timesteps=2,
                fine_num_timesteps=2,
                device=torch.device("cpu"),
                end_time=0.01,
            )

    def test_rejects_duplicate_case_identity(self):
        with self.assertRaisesRegex(ValueError, "unique case"):
            self._predictor_for_member(0).sample_member(
                structured_conditioning=self.condition,
                valid_mask=self.mask,
                case_ids=("same", "same"),
                member=0,
                coarse_num_timesteps=2,
                fine_num_timesteps=2,
                device=torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
