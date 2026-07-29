import tempfile
import unittest
from pathlib import Path
import os
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from few_shot.harness.models import (
    INPFormerWrapper,
    _official_gaussian_kernel,
    discover_official_checkpoints,
    official_checkpoint_directory,
)


class _FakeINPModel(nn.Module):
    def forward(self, images):
        batch_size = images.shape[0]
        encoder_a = torch.tensor(
            [
                [[1.0, 0.0], [0.5, 0.5]],
                [[0.0, 1.0], [0.5, -0.5]],
            ]
        ).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        decoder_a = torch.tensor(
            [
                [[1.0, 1.0], [0.5, -0.5]],
                [[0.0, 0.0], [0.5, 0.5]],
            ]
        ).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        encoder_b = torch.flip(encoder_a, dims=(-1,))
        decoder_b = torch.flip(decoder_a, dims=(-2,))
        return [encoder_a, encoder_b], [decoder_a, decoder_b], torch.tensor(0.0)


def _direct_official_postprocess(raw_map):
    anomaly_map = F.interpolate(
        raw_map[:, None],
        size=(392, 392),
        mode="bilinear",
        align_corners=True,
    )
    anomaly_map = F.interpolate(
        anomaly_map,
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    )
    return F.conv2d(
        anomaly_map,
        _official_gaussian_kernel(5, 4),
        padding=2,
    )[:, 0]


class INPFormerOfficialScoringTest(unittest.TestCase):
    def test_official_import_path_forces_broken_xformers_branch_off(self):
        previous = os.environ.pop("XFORMERS_DISABLED", None)
        previous_sys_path = sys.path.copy()
        try:
            INPFormerWrapper._prepare_imports(Path(temporary_directory := tempfile.gettempdir()))
            self.assertEqual(os.environ["XFORMERS_DISABLED"], "1")
            self.assertEqual(str(temporary_directory), sys.path[0])
        finally:
            sys.path[:] = previous_sys_path
            if previous is None:
                os.environ.pop("XFORMERS_DISABLED", None)
            else:
                os.environ["XFORMERS_DISABLED"] = previous

    def test_forward_matches_official_cosine_map_gaussian_and_top_one_percent(self):
        wrapper = INPFormerWrapper(shot=1, device="cpu")
        wrapper.model = _FakeINPModel()
        wrapper.active_checkpoint = Path("model.pth")

        images = [Image.new("RGB", (17, 23), color=(10, 20, 30))]
        actual_scores, actual_raw_maps = wrapper.forward_raw_batch(images)

        with torch.no_grad():
            en, de, _ = wrapper.model(torch.zeros((1, 3, 392, 392)))
            expected_raw = torch.stack([
                1.0 - F.cosine_similarity(source, target, dim=1)
                for source, target in zip(en, de)
            ], dim=1).mean(dim=1)
            expected_map = _direct_official_postprocess(expected_raw)
            flattened = expected_map.flatten(1)
            top_count = int(flattened.shape[1] * 0.01)
            expected_score = torch.sort(
                flattened, dim=1, descending=True
            )[0][:, :top_count].mean(dim=1)

        np.testing.assert_allclose(
            actual_raw_maps, expected_raw.numpy(), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            actual_scores, expected_score.numpy(), rtol=1e-6, atol=1e-6
        )
        self.assertEqual(actual_raw_maps.shape, (1, 2, 2))

    def test_metric_map_and_mask_use_official_256_coordinate_space(self):
        wrapper = INPFormerWrapper(shot=4, device="cpu")
        raw_map = np.arange(28 * 28, dtype=np.float32).reshape(28, 28)
        mask = np.zeros((100, 200), dtype=np.float32)
        mask[20:80, 50:150] = 1

        metric_map = wrapper.prepare_metric_map(raw_map)
        metric_mask = wrapper.prepare_metric_mask(mask)

        self.assertEqual(metric_map.shape, (256, 256))
        self.assertEqual(metric_mask.shape, (256, 256))
        self.assertEqual(set(np.unique(metric_mask)), {0.0, 1.0})

    def test_discovery_requires_dataset_and_shot_specific_directories(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for shot in (1, 2, 4):
                for dataset in ("mvtec", "visa"):
                    checkpoint = (
                        root
                        / official_checkpoint_directory(shot, dataset)
                        / "model.pth"
                    )
                    checkpoint.parent.mkdir(parents=True)
                    checkpoint.touch()

            discovered = discover_official_checkpoints(str(root))

        self.assertEqual(set(discovered), {1, 2, 4})
        for shot in (1, 2, 4):
            self.assertEqual(set(discovered[shot]), {"mvtec", "visa"})
            self.assertIn(f"Few-Shot-{shot}", discovered[shot]["mvtec"])
            self.assertIn("dataset=VisA", discovered[shot]["visa"])

    def test_discovery_accepts_only_the_selected_shot_directories(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for dataset in ("mvtec", "visa"):
                checkpoint = (
                    root
                    / official_checkpoint_directory(2, dataset)
                    / "model.pth"
                )
                checkpoint.parent.mkdir(parents=True)
                checkpoint.touch()

            discovered = discover_official_checkpoints(str(root), shots=[2])

        self.assertEqual(set(discovered), {2})
        self.assertEqual(set(discovered[2]), {"mvtec", "visa"})

    def test_discovery_requires_only_the_selected_target_dataset_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = (
                root
                / official_checkpoint_directory(4, "visa")
                / "model.pth"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()

            discovered = discover_official_checkpoints(
                str(root), shots=[4], datasets=["visa"]
            )

        self.assertEqual(set(discovered), {4})
        self.assertEqual(set(discovered[4]), {"visa"})
        self.assertIn("dataset=VisA", discovered[4]["visa"])


if __name__ == "__main__":
    unittest.main()
