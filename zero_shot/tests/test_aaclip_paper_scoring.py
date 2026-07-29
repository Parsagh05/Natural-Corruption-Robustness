import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from zero_shot.harness.aaclip_scoring import postprocess_aaclip_industrial_condition
from zero_shot.harness.config import DatasetConfig, HarnessConfig
from zero_shot.harness.metrics import resize_anomaly_map
from zero_shot.harness.models import AACLIPWrapper, AFCLIPWrapper
from zero_shot.harness.runner import RobustnessRunner


class _FakeAACLIPModel:
    def __init__(self, patch_features, detection_features):
        self.patch_features = patch_features
        self.detection_features = detection_features

    def __call__(self, image_tensor):
        batch_size = image_tensor.shape[0]
        return (
            [feature[:batch_size] for feature in self.patch_features],
            self.detection_features[:batch_size],
        )


class _TwoSampleDataset:
    def __init__(self):
        self.samples = [
            {"mask_path": None, "image_path": Path(f"/dataset/sample_{index}.png")}
            for index in range(2)
        ]
        self.items = [
            {
                "image": object(),
                "mask": np.asarray(
                    [[0.0, 0.0], [0.0, float(index)]], dtype=np.float32
                ),
                "label": index,
                "sample_id": f"widget/sample_{index}.png",
                "selected_corruption": None,
            }
            for index in range(2)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class _RunnerModel:
    metric_map_align_corners = False
    condition_postprocessing = False

    def __init__(self, paper_postprocessing):
        self.model_name = "AA-CLIP" if paper_postprocessing else "OtherModel"
        self.metric_map_align_corners = paper_postprocessing
        self.condition_postprocessing = paper_postprocessing

    def forward_raw_batch(self, images, category):
        return (
            np.asarray([2.0, 4.0], dtype=np.float32),
            np.asarray(
                [
                    [[0.0, 1.0], [2.0, 3.0]],
                    [[4.0, 5.0], [6.0, 7.0]],
                ],
                dtype=np.float32,
            ),
        )

    def postprocess_condition_outputs(self, scores, anomaly_maps):
        return postprocess_aaclip_industrial_condition(scores, anomaly_maps)


class AACLIPPaperScoringTest(unittest.TestCase):
    def test_raw_image_and_patch_scores_match_official_formulas(self):
        patch_feature = torch.tensor(
            [
                [[0.10, 0.20], [0.30, 0.10], [-0.10, 0.40], [0.00, 0.00]],
                [[0.20, 0.00], [0.10, 0.30], [0.40, -0.20], [-0.10, 0.10]],
            ],
            dtype=torch.float32,
        )
        detection_feature = torch.tensor(
            [[0.20, 0.60], [-0.40, -0.20]], dtype=torch.float32
        )
        text_features = torch.eye(2, dtype=torch.float32)

        wrapper = AACLIPWrapper(device="cpu", apply_score_blur=False)
        wrapper.model = _FakeAACLIPModel([patch_feature], detection_feature)
        wrapper.preprocess = lambda image: torch.zeros(3, 2, 2)
        wrapper._get_text_features = lambda category: text_features
        images = [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))]

        scores, maps = wrapper.forward_raw_batch(images, category="bottle")

        expected_scores = ((detection_feature[:, 1] + 1.0) / 2.0).numpy()
        patch_scores = 100.0 * torch.matmul(patch_feature, text_features)
        patch_scores = patch_scores.permute(0, 2, 1).reshape(2, 2, 2, 2)
        expected_maps = (
            (patch_scores[:, 1] + 1.0 - patch_scores[:, 0]) / 2.0
        ).numpy()
        np.testing.assert_allclose(scores, expected_scores, rtol=0, atol=1e-6)
        np.testing.assert_allclose(maps, expected_maps, rtol=0, atol=1e-6)

    def test_industrial_postprocessing_matches_official_normalize_and_fuse(self):
        image_scores = np.asarray([2.0, 4.0], dtype=np.float32)
        maps = [
            np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
            np.asarray([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32),
        ]
        original_maps = np.stack([anomaly_map.copy() for anomaly_map in maps])

        fused_scores, normalized_maps = postprocess_aaclip_industrial_condition(
            image_scores, maps
        )

        expected_maps = original_maps / 7.0
        expected_scores = np.asarray([1.5 / 7.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(
            np.stack(normalized_maps), expected_maps, rtol=0, atol=1e-6
        )
        np.testing.assert_allclose(fused_scores, expected_scores, rtol=0, atol=1e-6)

    def test_constant_inputs_remain_finite(self):
        fused_scores, normalized_maps = postprocess_aaclip_industrial_condition(
            np.asarray([0.5, 0.5], dtype=np.float32),
            [
                np.full((2, 2), 3.0, dtype=np.float32),
                np.full((2, 2), 3.0, dtype=np.float32),
            ],
        )

        self.assertTrue(np.isfinite(fused_scores).all())
        self.assertTrue(np.isfinite(np.stack(normalized_maps)).all())
        np.testing.assert_array_equal(fused_scores, np.zeros(2, dtype=np.float32))

    def test_only_aaclip_opts_into_paper_specific_shared_hooks(self):
        self.assertTrue(AACLIPWrapper.metric_map_align_corners)
        self.assertTrue(AACLIPWrapper.condition_postprocessing)
        self.assertFalse(AFCLIPWrapper.metric_map_align_corners)
        self.assertFalse(AFCLIPWrapper.condition_postprocessing)

    def test_aaclip_map_resize_uses_official_align_corners_mode(self):
        anomaly_map = np.asarray([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)
        actual = resize_anomaly_map(
            anomaly_map, 5, 5, align_corners=AACLIPWrapper.metric_map_align_corners
        )
        expected = torch.nn.functional.interpolate(
            torch.from_numpy(anomaly_map)[None, None],
            size=(5, 5),
            mode="bilinear",
            align_corners=True,
        )[0, 0].numpy()
        default = resize_anomaly_map(anomaly_map, 5, 5)

        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
        self.assertFalse(np.array_equal(actual, default))

    def test_runner_changes_only_opted_in_aaclip_scores(self):
        pixel_metrics = {
            "auroc_px": 50.0,
            "f1_px": 50.0,
            "aupro_px": 50.0,
            "threshold_px": 0.5,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "outputs"
            runner = RobustnessRunner(
                HarnessConfig(
                    output_root=output_root,
                    datasets=[],
                    models=[],
                    include_clean=False,
                    device="cpu",
                    batch_size=2,
                )
            )
            dataset_config = DatasetConfig(
                name="TestData",
                root_path=Path(temporary_directory),
                categories=["widget"],
            )

            with patch(
                "zero_shot.harness.runner.compute_pixel_metrics",
                return_value=pixel_metrics,
            ):
                for paper_postprocessing in (False, True):
                    runner._run_category(
                        model_wrapper=_RunnerModel(paper_postprocessing),
                        dataset_config=dataset_config,
                        category="widget",
                        corruption_type="clean",
                        severity=0,
                        dataset=_TwoSampleDataset(),
                    )

            other_scores = np.load(
                output_root
                / "OtherModel/TestData/widget/clean/level_0/raw_scores.npy"
            )
            aaclip_scores = np.load(
                output_root / "AA-CLIP/TestData/widget/clean/level_0/raw_scores.npy"
            )

            np.testing.assert_array_equal(
                other_scores, np.asarray([2.0, 4.0], dtype=np.float32)
            )
            np.testing.assert_allclose(
                aaclip_scores,
                np.asarray([1.5 / 7.0, 1.0], dtype=np.float32),
                rtol=0,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
