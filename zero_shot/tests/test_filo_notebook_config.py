import json
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from zero_shot.harness.config import COMPLETED_MODELS
from zero_shot.harness.models import FiLoWrapper, MODEL_REGISTRY, get_model


class FiLoNotebookConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook_path = Path(__file__).resolve().parents[1] / "kaggle_final_filo.ipynb"
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

    def test_official_sources_and_released_checkpoints_are_used(self):
        self.assertIn("https://github.com/CASIA-LMC-Lab/FiLo.git", self.source)
        self.assertIn('HF_REPOSITORY = "FantasticGNU/FiLo"', self.source)
        self.assertIn('f"filo_train_on_{WEIGHT_DATASET}.pth"', self.source)
        self.assertIn('f"grounding_train_on_{WEIGHT_DATASET}.pth"', self.source)
        self.assertIn("hf_hub_download", self.source)
        self.assertNotIn('"pip", "install", "-q", "-e"', self.source)

    def test_checkpoint_mapping_is_cross_dataset(self):
        self.assertIn(
            'WEIGHT_DATASET = "visa" if IS_MVTEC else "mvtec"', self.source
        )
        self.assertIn('"dataset_name": DATASET_NAME', self.source)
        self.assertIn("BATCH_SIZE = 1", self.source)

    def test_notebook_runs_the_shared_robustness_pipeline(self):
        self.assertIn("run_evaluation(", self.source)
        self.assertIn("corruption_plan_path", self.source)
        self.assertIn("categorized_corruptions=USE_CATEGORIZED_CORRUPTIONS", self.source)
        self.assertIn("include_clean=INCLUDE_CLEAN_BASELINE", self.source)


class FiLoRegistryTest(unittest.TestCase):
    def test_filo_is_an_executable_registered_model(self):
        self.assertIs(MODEL_REGISTRY["FiLo"], FiLoWrapper)
        self.assertIn("FiLo", COMPLETED_MODELS)
        wrapper = get_model("FiLo", device="cpu", dataset_name="visa")
        self.assertIsInstance(wrapper, FiLoWrapper)
        self.assertEqual(wrapper.device, "cpu")

    def test_wrapper_uses_cross_dataset_weights(self):
        self.assertEqual(
            FiLoWrapper(dataset_name="mvtec")._cross_dataset_weight_name(),
            "visa",
        )
        self.assertEqual(
            FiLoWrapper(dataset_name="visa")._cross_dataset_weight_name(),
            "mvtec",
        )

    def test_forward_matches_official_score_and_box_reweighting(self):
        class StubFiLo:
            def __call__(self, items, with_adapter, positions):
                self.items = items
                self.with_adapter = with_adapter
                self.positions = positions
                text_probs = torch.tensor([[[0.2, 0.6]]], dtype=torch.float32)
                anomaly_maps = [
                    torch.tensor(
                        [[[[0.2, 0.2], [0.2, 0.2]], [[0.8, 0.8], [0.8, 0.8]]]],
                        dtype=torch.float32,
                    )
                ]
                return text_probs, anomaly_maps

        wrapper = FiLoWrapper(device="cpu", dataset_name="mvtec")
        wrapper.model = StubFiLo()
        wrapper.preprocess = lambda image: torch.zeros((3, 2, 2))
        wrapper.grounding_model = object()
        wrapper._gaussian_blur = lambda anomaly_map: anomaly_map
        wrapper._localize = lambda image, category: (
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            ["top left"],
        )

        score, anomaly_map = wrapper.forward_raw(
            Image.new("RGB", (2, 2)), category="metal_nut"
        )

        # Official map formula: (abnormal - normal + 1) / 2 = 0.8.
        # The selected rectangle retains 0.8 and the rest is multiplied by 0.7.
        np.testing.assert_allclose(
            anomaly_map,
            np.array([[0.8, 0.56], [0.56, 0.56]], dtype=np.float32),
            rtol=1e-6,
        )
        self.assertAlmostEqual(score, 0.7, places=6)
        self.assertEqual(wrapper.model.items["cls_name"], ["metal nut"])
        self.assertEqual(wrapper.model.positions, ["top left"])
        self.assertTrue(wrapper.model.with_adapter)

    def test_artifact_maps_are_compact_without_changing_metric_maps(self):
        wrapper = FiLoWrapper(device="cpu", dataset_name="visa")
        full_maps = np.linspace(
            0.0, 1.0, num=2 * 518 * 518, dtype=np.float32
        ).reshape(2, 518, 518)

        compact_maps = wrapper.prepare_artifact_maps(full_maps)

        self.assertEqual(compact_maps.shape, (2, 37, 37))
        self.assertEqual(compact_maps.dtype, np.float32)
        self.assertEqual(full_maps.shape, (2, 518, 518))
        self.assertGreaterEqual(float(compact_maps.min()), 0.0)
        self.assertLessEqual(float(compact_maps.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
