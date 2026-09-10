import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from assim_lib.visual_review_log import (
    write_sample_provenance_manifest,
    write_visual_evidence_manifest,
    write_visual_review,
)


class VisualReviewLogTests(unittest.TestCase):
    def _evidence(self, root: Path) -> Path:
        checkpoint = root / "model.pth"
        sample = root / "samples.pt"
        image = root / "member.png"
        checkpoint.write_bytes(b"checkpoint")
        sample.write_bytes(b"sample")
        image.write_bytes(b"image")
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        provenance = write_sample_provenance_manifest(
            manifest_path=root / "sample_provenance.json",
            source_commit="a" * 40,
            checkpoint_bindings=[{"stage": "direct", "role": "ema", "path": checkpoint}],
            sample_path=sample,
            case_ids=["case-a"],
            member_indices=[0],
            solver={"method": "rk4", "num_timesteps": 65, "end_time": 0.0},
            replay_identity={
                "replay_code_commit": "a" * 40,
                "direct_checkpoint_sha256": checkpoint_sha256,
            },
        )
        return write_visual_evidence_manifest(
            manifest_path=root / "visual_evidence.json",
            sample_provenance_path=provenance,
            image_paths=[image],
            panel_bindings=[
                {
                    "image_index": 0,
                    "case_id": "case-a",
                    "member": 0,
                    "lead_days": [3, 6, 9],
                    "fields": ["sic", "sit"],
                }
            ],
            renderer="structured_trajectory_figure_v1",
            origin="upper",
            scales="active-ocean-only",
            display="raw",
        )

    @staticmethod
    def _review_kwargs(root: Path, evidence: Path) -> dict:
        return {
            "run_dir": root,
            "review_id": "fixed-member-00",
            "evidence_manifest_path": evidence,
            "inspected_panel_indices": [0],
            "reviewer": "codex-visual-inspection",
            "ratings": {
                "macro_coherence": "pass",
                "grain_or_pits": "fail",
                "coastline": "not_assessed",
                "false_low_ice_sit": "not_assessed",
                "support": "pass",
            },
            "observations": ["Visible pixel-scale grain in all forecast leads."],
            "verdict": "fail",
        }

    def test_review_is_evidence_bound_and_cannot_be_overwritten(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = self._evidence(root)
            kwargs = self._review_kwargs(root, evidence)
            review_dir = write_visual_review(**kwargs)
            payload = json.loads((review_dir / "visual_review.json").read_text())
            self.assertFalse(payload["training_authorized"])
            self.assertEqual(payload["visual_verdict"], "fail")
            self.assertEqual(payload["inspected_panels"][0]["case_id"], "case-a")
            self.assertTrue((review_dir / "visual_review.md").is_file())
            with self.assertRaises(FileExistsError):
                write_visual_review(**kwargs)

    def test_review_rejects_modified_bound_artifact(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = self._evidence(root)
            (root / "member.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                write_visual_review(**self._review_kwargs(root, evidence))

    def test_rejects_contradictory_verdict(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = self._evidence(root)
            kwargs = self._review_kwargs(root, evidence)
            kwargs["verdict"] = "pass"
            with self.assertRaisesRegex(ValueError, "contradicts"):
                write_visual_review(**kwargs)

    def test_rejects_contradictory_sample_replay_identity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pth"
            sample = root / "samples.pt"
            checkpoint.write_bytes(b"checkpoint")
            sample.write_bytes(b"sample")
            with self.assertRaisesRegex(ValueError, "checkpoint SHA"):
                write_sample_provenance_manifest(
                    manifest_path=root / "sample_provenance.json",
                    source_commit="a" * 40,
                    checkpoint_bindings=[{"stage": "direct", "role": "ema", "path": checkpoint}],
                    sample_path=sample,
                    case_ids=["case-a"],
                    member_indices=[0],
                    solver={"method": "rk4", "num_timesteps": 65, "end_time": 0.0},
                    replay_identity={
                        "replay_code_commit": "a" * 40,
                        "direct_checkpoint_sha256": "0" * 64,
                    },
                )


if __name__ == "__main__":
    unittest.main()
