import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from harness.config import DatasetConfig, HarnessConfig
from harness.runner import RobustnessRunner

from scripts.extract_categorized_fine_grained import (
    CATEGORY_CORRUPTIONS,
    _metric_header,
    _metric_rows,
    _parse_artifact_path,
    _relative_dataset_path,
)


class CategorizedFineGrainedExtractionTest(unittest.TestCase):
    def test_metric_rows_preserve_protocol_order_and_compute_means(self):
        categories = ["class_a", "class_b"]
        metric_names = ("auroc_sp", "threshold_sp")
        results = {
            ("shot_noise", 2): {
                "class_a": {"auroc_sp": 80.0, "threshold_sp": 0.2},
                "class_b": {"auroc_sp": 60.0, "threshold_sp": 0.4},
            },
            ("gaussian_noise", 1): {
                "class_a": {"auroc_sp": 90.0, "threshold_sp": 0.1},
                "class_b": {"auroc_sp": 70.0, "threshold_sp": 0.3},
            },
        }

        rows = _metric_rows(results, categories, metric_names)

        self.assertEqual(
            _metric_header(categories, metric_names),
            [
                "transformation_level",
                "class_a_auroc_sp",
                "class_a_threshold_sp",
                "class_b_auroc_sp",
                "class_b_threshold_sp",
                "mean_auroc_sp",
                "mean_threshold_sp",
            ],
        )
        self.assertEqual(rows[0], ["gaussian_noise_level 1", 90.0, 0.1, 70.0, 0.3, 80.0, 0.2])
        self.assertEqual(rows[1], ["shot_noise_level 2", 80.0, 0.2, 60.0, 0.4, 70.0, 0.3])

    def test_artifact_path_parsing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metadata_path = root / "bottle" / "noise" / "level_3" / "metadata.json"
            self.assertEqual(
                _parse_artifact_path(root, metadata_path),
                ("bottle", "noise", 3),
            )

    def test_category_mapping_covers_requested_noise_types(self):
        self.assertEqual(
            CATEGORY_CORRUPTIONS["noise"],
            ("gaussian_noise", "shot_noise", "impulse_noise"),
        )

    def test_dataset_relative_path_is_recovered_for_geometric_mask_seed(self):
        self.assertEqual(
            _relative_dataset_path(
                "/kaggle/input/mvtec/mvtec_anomaly_detection/"
                "bottle/test/broken_large/000.png",
                "bottle",
            ),
            "bottle/test/broken_large/000.png",
        )


class _FakeCategorizedDataset:
    def __init__(self):
        corruptions = [
            "gaussian_noise",
            "gaussian_noise",
            "shot_noise",
            "shot_noise",
            "impulse_noise",
            "impulse_noise",
        ]
        self.samples = [
            {
                "image_path": Path(f"/dataset/widget/test/{index}.png"),
                "mask_path": (
                    Path(f"/dataset/widget/masks/{index}.png")
                    if index % 2
                    else None
                ),
            }
            for index in range(6)
        ]
        self.items = [
            {
                "image": object(),
                "mask": np.full((2, 2), index % 2, dtype=np.float32),
                "label": index % 2,
                "sample_id": f"widget/sample_{index}.png",
                "selected_corruption": corruptions[index],
            }
            for index in range(6)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class _FakeCategorizedModel:
    model_name = "TestModel"

    def forward_raw_batch(self, images, category):
        return (
            np.asarray([0.1, 0.9, 0.2, 0.8, 0.3, 0.7], dtype=np.float32),
            np.stack([
                np.full((2, 2), index / 10, dtype=np.float32)
                for index in range(6)
            ]),
        )


class CategorizedPipelineOutputTest(unittest.TestCase):
    def test_runner_exports_fine_grained_sp_px_and_per_image_json(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "outputs"
            config = HarnessConfig(
                output_root=output_root,
                datasets=[],
                models=[],
                corruption_types=["noise"],
                severity_levels=[1],
                include_clean=False,
                categorized_corruptions=True,
                categorized_corruption_plans={},
                batch_size=6,
                device="cpu",
            )
            runner = RobustnessRunner(config)
            dataset_config = DatasetConfig(
                name="TestData",
                root_path=Path("/dataset"),
                categories=["widget"],
            )

            pixel_metrics = {
                "auroc_px": 75.0,
                "aupro_px": 65.0,
                "f1_px": 55.0,
                "threshold_px": 0.25,
            }
            with patch(
                "harness.runner.compute_pixel_metrics",
                return_value=pixel_metrics,
            ) as compute_pixel_mock:
                clean_dataset = _FakeCategorizedDataset()
                for item in clean_dataset.items:
                    item["selected_corruption"] = None
                runner._run_category(
                    model_wrapper=_FakeCategorizedModel(),
                    dataset_config=dataset_config,
                    category="widget",
                    corruption_type="clean",
                    severity=0,
                    dataset=clean_dataset,
                )
                runner._run_category(
                    model_wrapper=_FakeCategorizedModel(),
                    dataset_config=dataset_config,
                    category="widget",
                    corruption_type="noise",
                    severity=1,
                    dataset=_FakeCategorizedDataset(),
                )

            self.assertEqual(compute_pixel_mock.call_count, 5)
            px_path, sp_path, json_path = (
                runner.eval_harness.export_categorized_fine_grained(
                    "TestModel", dataset_config
                )
            )
            self.assertTrue(px_path.is_file())
            self.assertTrue(sp_path.is_file())
            self.assertTrue(json_path.is_file())

            with sp_path.open(newline="", encoding="utf-8") as input_file:
                sp_rows = list(csv.DictReader(input_file))
            with px_path.open(newline="", encoding="utf-8") as input_file:
                px_rows = list(csv.DictReader(input_file))
            self.assertEqual(
                [row["transformation_level"] for row in sp_rows],
                [
                    "clean_level 0",
                    "gaussian_noise_level 1",
                    "shot_noise_level 1",
                    "impulse_noise_level 1",
                ],
            )
            self.assertEqual(
                [row["transformation_level"] for row in px_rows],
                [
                    "clean_level 0",
                    "gaussian_noise_level 1",
                    "shot_noise_level 1",
                    "impulse_noise_level 1",
                ],
            )
            self.assertEqual(float(px_rows[0]["widget_auroc_px"]), 75.0)

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["records"]), 12)
            self.assertEqual(
                payload["records"][0]["selected_corruption"],
                "clean",
            )
            self.assertEqual(
                payload["records"][0]["transformation_level"],
                "clean_level 0",
            )
            self.assertAlmostEqual(payload["records"][0]["image_score"], 0.1)


if __name__ == "__main__":
    unittest.main()
