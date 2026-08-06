import json
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from zero_shot.harness.config import COMPLETED_MODELS
from zero_shot.harness.models import FBCLIPWrapper, MODEL_REGISTRY, get_model


class FBCLIPNotebookConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook_path = (
            Path(__file__).resolve().parents[1] / "kaggle_final_fbclip.ipynb"
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

    def test_official_source_and_checkpoint_ids_are_used(self):
        self.assertIn(
            "https://github.com/Xi-Mu-Yu/FB-CLIP.git", self.source
        )
        self.assertIn("1Qw0w-5WeYcVbOlQvrJnjcAgjP9SLhMTC", self.source)
        self.assertIn("1hzKUafDEpF1KUk6psKnA3anrAGR2nGj4", self.source)
        self.assertIn('"mvtec_epoch_1_model.pth"', self.source)
        self.assertIn('"visa_epoch_2_model.pth"', self.source)
        self.assertIn("gdown.download(", self.source)

    def test_checkpoint_mapping_is_cross_dataset(self):
        self.assertIn(
            'WEIGHT_DATASET = "visa" if IS_MVTEC else "mvtec"', self.source
        )
        self.assertIn('"dataset_name": DATASET_NAME', self.source)
        self.assertIn('"weight_dataset": WEIGHT_DATASET', self.source)
        self.assertIn('"feature_layers": [1, 6, 12, 18, 24]', self.source)
        self.assertIn('"image_size": 518', self.source)
        self.assertIn('"sigma": 4', self.source)

    def test_notebook_runs_shared_robustness_pipeline(self):
        self.assertIn("run_evaluation(", self.source)
        self.assertIn("corruption_plan_path", self.source)
        self.assertIn(
            "categorized_corruptions=USE_CATEGORIZED_CORRUPTIONS", self.source
        )
        self.assertIn("include_clean=INCLUDE_CLEAN_BASELINE", self.source)
        self.assertIn("BATCH_SIZE = 1", self.source)


class FBCLIPRegistryTest(unittest.TestCase):
    def test_fbclip_is_an_executable_registered_model(self):
        self.assertIs(MODEL_REGISTRY["FB-CLIP"], FBCLIPWrapper)
        self.assertIn("FB-CLIP", COMPLETED_MODELS)
        wrapper = get_model(
            "FB-CLIP",
            device="cpu",
            dataset_name="visa",
            weight_dataset="mvtec",
        )
        self.assertIsInstance(wrapper, FBCLIPWrapper)
        self.assertEqual(wrapper.device, "cpu")

    def test_wrapper_enforces_cross_dataset_weights(self):
        self.assertEqual(
            FBCLIPWrapper(
                dataset_name="mvtec", weight_dataset="visa"
            )._cross_dataset_weight_name(),
            "visa",
        )
        self.assertEqual(
            FBCLIPWrapper(
                dataset_name="visa", weight_dataset="mvtec"
            )._cross_dataset_weight_name(),
            "mvtec",
        )
        with self.assertRaisesRegex(ValueError, "other dataset"):
            FBCLIPWrapper(
                dataset_name="mvtec", weight_dataset="mvtec"
            )._cross_dataset_weight_name()

    def test_forward_matches_official_fb_encode_outputs(self):
        class StubPromptLearner:
            def __call__(self, cls_id=None):
                self.cls_id = cls_id
                return "prompts", "tokens", "compound"

        class StubModel:
            def FB_encode(self, images, args, **prompt_inputs):
                self.images = images
                self.args = args
                self.prompt_inputs = prompt_inputs
                batch_size = images.shape[0]
                scores = torch.tensor([0.25, 0.75], dtype=torch.float32)[
                    :batch_size
                ]
                maps = torch.arange(
                    batch_size * 4, dtype=torch.float32
                ).reshape(batch_size, 1, 2, 2)
                return scores, maps, torch.tensor(0.0)

        wrapper = FBCLIPWrapper(
            device="cpu", dataset_name="visa", weight_dataset="mvtec"
        )
        wrapper.model = StubModel()
        wrapper.prompt_learner = StubPromptLearner()
        wrapper.preprocess = lambda image: torch.zeros((3, 4, 4))
        wrapper._args = object()

        scores, anomaly_maps = wrapper.forward_raw_batch(
            [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))]
        )

        np.testing.assert_allclose(scores, np.array([0.25, 0.75]))
        np.testing.assert_allclose(
            anomaly_maps,
            np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        )
        self.assertIsNone(wrapper.prompt_learner.cls_id)
        self.assertEqual(
            wrapper.model.prompt_inputs,
            {
                "prompts": "prompts",
                "tokenized_prompts": "tokens",
                "compound_prompts_text": "compound",
            },
        )

    def test_metric_map_matches_official_resize_and_gaussian_filter(self):
        wrapper = FBCLIPWrapper(
            device="cpu",
            dataset_name="visa",
            weight_dataset="mvtec",
            image_size=518,
            sigma=4,
            use_gaussian_filter=True,
        )
        lowres_map = np.zeros((37, 37), dtype=np.float32)
        lowres_map[18, 18] = 1.0

        metric_map = wrapper.prepare_metric_map(lowres_map)

        self.assertEqual(metric_map.shape, (518, 518))
        self.assertEqual(metric_map.dtype, np.float32)
        self.assertGreater(float(metric_map.max()), 0.0)
        self.assertLess(float(metric_map.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
