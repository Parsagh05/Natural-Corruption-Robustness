import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from few_shot.harness.models import (
    AFCLIP_DATASET_CATEGORIES,
    APRILGANFewShotWrapper,
    MODEL_REGISTRY,
    discover_aprilgan_checkpoints,
)


class APRILGANFewShotTests(unittest.TestCase):
    def test_wrapper_is_registered(self):
        self.assertIs(MODEL_REGISTRY["APRIL-GAN"], APRILGANFewShotWrapper)
        wrapper = APRILGANFewShotWrapper(shot=4, device="cpu")
        self.assertEqual(wrapper.model_name, "APRIL-GAN-4-shot")
        self.assertTrue(wrapper.condition_postprocessing)

    def test_checkpoint_discovery_requires_both_released_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "mvtec_pretrained.pth").touch()
            (root / "visa_pretrained.pth").touch()
            discovered = discover_aprilgan_checkpoints(str(root))
            self.assertEqual(set(discovered), {"mvtec", "visa"})
            self.assertTrue(discovered["mvtec"].endswith("mvtec_pretrained.pth"))

    def test_support_selection_reproduces_seeded_torch_randint(self):
        wrapper = APRILGANFewShotWrapper(shot=4, reference_seed=42, device="cpu")
        candidates = tuple(Path(f"normal-{index}.png") for index in range(2))
        wrapper._normal_training_paths = lambda dataset, category: candidates
        first = wrapper._select_support_paths("mvtec")
        second = wrapper._select_support_paths("mvtec")
        self.assertEqual(first, second)
        self.assertEqual(tuple(first), AFCLIP_DATASET_CATEGORIES["mvtec"])
        self.assertTrue(
            any(len(set(paths)) < len(paths) for paths in first.values()),
            "Sampling with replacement should permit duplicate support draws.",
        )

    def test_official_image_score_fusion_is_condition_level(self):
        wrapper = APRILGANFewShotWrapper(shot=1, device="cpu")
        scores = np.asarray([0.2, 0.8], dtype=np.float32)
        maps = [
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            np.asarray([[3.0, 0.0]], dtype=np.float32),
        ]
        fused, returned_maps = wrapper.postprocess_condition_outputs(scores, maps)
        np.testing.assert_allclose(fused, np.asarray([0.1, 0.9], dtype=np.float32))
        self.assertIs(returned_maps, maps)

    def test_kaggle_notebook_runs_the_few_shot_suite(self):
        notebook_path = Path(__file__).resolve().parents[1] / "kaggle_aprilgan.ipynb"
        payload = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in payload["cells"]
        )
        self.assertIn("APRILGANFewShotWrapper", source)
        self.assertIn("run_aprilgan_evaluations", source)
        self.assertIn("SHOTS_TO_RUN = [1, 2, 4]", source)
        self.assertIn("REFERENCE_SEEDS = [42]", source)
        self.assertIn("torch.randint", source)


if __name__ == "__main__":
    unittest.main()
