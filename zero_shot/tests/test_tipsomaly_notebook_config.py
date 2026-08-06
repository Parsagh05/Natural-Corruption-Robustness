import json
import math
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from zero_shot.harness.config import COMPLETED_MODELS
from zero_shot.harness.models import MODEL_REGISTRY, TipsomalyWrapper, get_model


class TipsomalyNotebookConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook_path = (
            Path(__file__).resolve().parents[1] / "kaggle_final_tipsomaly.ipynb"
        )
        cls.notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        )

    def test_all_code_cells_compile(self):
        for index, cell in enumerate(self.notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            with self.subTest(cell=index):
                compile("".join(cell.get("source", [])), f"cell_{index}", "exec")

    def test_official_source_and_repository_checkpoints_are_used(self):
        self.assertIn(
            "https://github.com/Alireza99Salehi/Tipsomaly.git", self.source
        )
        self.assertIn(
            'WEIGHT_DATASET = "visa" if IS_MVTEC else "mvtec"', self.source
        )
        self.assertIn(
            'f"trained_on_{WEIGHT_DATASET}_default"', self.source
        )
        self.assertIn('"learnable_params_2.pth"', self.source)
        self.assertIn('MODEL_VERSION = "l14h"', self.source)

    def test_notebook_runs_shared_robustness_pipeline(self):
        self.assertIn("run_evaluation(", self.source)
        self.assertIn("corruption_plan_path", self.source)
        self.assertIn(
            "categorized_corruptions=USE_CATEGORIZED_CORRUPTIONS", self.source
        )
        self.assertIn("include_clean=INCLUDE_CLEAN_BASELINE", self.source)


class TipsomalyRegistryTest(unittest.TestCase):
    def test_tipsomaly_is_an_executable_registered_model(self):
        self.assertIs(MODEL_REGISTRY["Tipsomaly"], TipsomalyWrapper)
        self.assertIn("Tipsomaly", COMPLETED_MODELS)
        wrapper = get_model("Tipsomaly", device="cpu", dataset_name="visa")
        self.assertIsInstance(wrapper, TipsomalyWrapper)
        self.assertEqual(wrapper.device, "cpu")

    def test_wrapper_enforces_cross_dataset_weights(self):
        self.assertEqual(
            TipsomalyWrapper(dataset_name="mvtec")._cross_dataset_weight_name(),
            "visa",
        )
        self.assertEqual(
            TipsomalyWrapper(dataset_name="visa")._cross_dataset_weight_name(),
            "mvtec",
        )
        with self.assertRaisesRegex(ValueError, "other dataset"):
            TipsomalyWrapper(
                dataset_name="mvtec", weight_dataset="mvtec"
            )._cross_dataset_weight_name()

    def test_forward_uses_spatial_token_and_strongest_local_evidence(self):
        class StubVisionEncoder:
            def __call__(self, images):
                batch_size = images.shape[0]
                object_token = torch.tensor([[[1.0, 0.0]]]).repeat(
                    batch_size, 1, 1
                )
                spatial_token = torch.tensor([[[0.0, 1.0]]]).repeat(
                    batch_size, 1, 1
                )
                patches = torch.tensor(
                    [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]]
                ).repeat(batch_size, 1, 1)
                return object_token, spatial_token, patches

        class StubTextEncoder:
            def __call__(self, texts, device, learned=False):
                del texts, device, learned
                return torch.eye(2).unsqueeze(0)

        wrapper = TipsomalyWrapper(device="cpu", dataset_name="visa")
        wrapper.model = StubVisionEncoder()
        wrapper.text_encoder = StubTextEncoder()
        wrapper.preprocess = lambda image: torch.zeros((3, 2, 2))
        wrapper.temperature = torch.tensor(1.0)
        wrapper._learnable_text_features = torch.eye(2).unsqueeze(0)

        score, anomaly_map = wrapper.forward_raw(
            Image.new("RGB", (2, 2)), category="pcb1"
        )

        abnormal_probability = math.e / (1.0 + math.e)
        self.assertAlmostEqual(score, 2 * abnormal_probability, places=6)
        np.testing.assert_allclose(
            anomaly_map,
            np.array(
                [
                    [1 - abnormal_probability, abnormal_probability],
                    [1 - abnormal_probability, abnormal_probability],
                ],
                dtype=np.float32,
            ),
            rtol=1e-6,
        )

    def test_metric_map_matches_regrid_smooth_contract(self):
        wrapper = TipsomalyWrapper(
            device="cpu", dataset_name="visa", image_size=518, sigma=4
        )
        lowres_map = np.zeros((37, 37), dtype=np.float32)
        lowres_map[18, 18] = 1.0

        metric_map = wrapper.prepare_metric_map(lowres_map)

        self.assertEqual(metric_map.shape, (518, 518))
        self.assertEqual(metric_map.dtype, np.float32)
        self.assertGreater(float(metric_map.max()), 0.0)
        self.assertLessEqual(float(metric_map.max()), 1.0)

        metric_mask = wrapper.prepare_metric_mask(
            np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        )
        self.assertEqual(metric_mask.shape, (518, 518))
        self.assertEqual(metric_mask.dtype, np.float32)
        self.assertEqual(set(np.unique(metric_mask)), {0.0, 1.0})


if __name__ == "__main__":
    unittest.main()
