import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from harness.config import DatasetConfig, HarnessConfig
from harness.dataset import AnomalyDetectionDataset
from harness.runner import RobustnessRunner
from harness.storage import ArtifactStorage


class CleanBaselineTest(unittest.TestCase):
    def test_clean_condition_is_first_by_default_and_only_added_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = HarnessConfig(
                output_root=Path(temp_dir) / "outputs",
                corruption_types=["clean", "brightness"],
                severity_levels=[1, 2],
            )

            self.assertEqual(
                config.evaluation_conditions,
                [("clean", 0), ("brightness", 1), ("brightness", 2)],
            )

    def test_empty_corruption_list_runs_clean_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = HarnessConfig(
                output_root=Path(temp_dir) / "outputs",
                corruption_types=[],
                include_clean=True,
            )

            self.assertEqual(config.evaluation_conditions, [("clean", 0)])

    def test_clean_condition_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = HarnessConfig(
                output_root=Path(temp_dir) / "outputs",
                corruption_types=["brightness"],
                severity_levels=[1, 2],
                include_clean=False,
            )

            self.assertEqual(
                config.evaluation_conditions,
                [("brightness", 1), ("brightness", 2)],
            )

    def test_clean_condition_returns_original_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "mvtec"
            image_dir = dataset_root / "bottle" / "test" / "good"
            image_dir.mkdir(parents=True)
            pixels = np.array(
                [
                    [[1, 2, 3], [4, 5, 6]],
                    [[7, 8, 9], [10, 11, 12]],
                ],
                dtype=np.uint8,
            )
            Image.fromarray(pixels).save(image_dir / "sample.png")

            dataset = AnomalyDetectionDataset(
                config=DatasetConfig(
                    name="mvtec",
                    root_path=dataset_root,
                    categories=["bottle"],
                ),
                corruption_type="clean",
                severity=0,
                category="bottle",
            )

            self.assertEqual(len(dataset), 1)
            np.testing.assert_array_equal(np.asarray(dataset[0]["image"]), pixels)

    def test_categorized_run_does_not_apply_a_category_to_clean_images(self):
        dataset_calls = []

        class FakeDataset:
            def __init__(self, **kwargs):
                dataset_calls.append(kwargs)

            def __len__(self):
                return 1

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            plan_path = temp_path / "plan.csv"
            plan_path.touch()
            dataset_config = DatasetConfig(
                name="mvtec",
                root_path=temp_path,
                categories=["bottle"],
            )
            config = HarnessConfig(
                output_root=temp_path / "outputs",
                datasets=[dataset_config],
                corruption_types=["noise"],
                severity_levels=[1],
                categorized_corruptions=True,
                categorized_corruption_plans={"mvtec": plan_path},
                device="cpu",
            )
            runner = RobustnessRunner(config)

            with (
                patch("harness.runner.AnomalyDetectionDataset", FakeDataset),
                patch.object(RobustnessRunner, "_run_category"),
                patch.object(ArtifactStorage, "cleanup_memory"),
            ):
                runner._run_model_on_dataset(
                    SimpleNamespace(model_name="fake-model"), dataset_config
                )

        self.assertEqual(
            [call["corruption_type"] for call in dataset_calls],
            ["clean", "noise"],
        )
        self.assertFalse(dataset_calls[0]["categorized_corruptions"])
        self.assertTrue(dataset_calls[1]["categorized_corruptions"])


if __name__ == "__main__":
    unittest.main()
