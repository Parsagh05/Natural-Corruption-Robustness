import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from zero_shot.harness.config import COMPLETED_MODELS
from zero_shot.harness.models import APRILGANWrapper, MODEL_REGISTRY


class APRILGANZeroShotTests(unittest.TestCase):
    def test_wrapper_is_registered_as_executable(self):
        self.assertIs(MODEL_REGISTRY["APRIL-GAN"], APRILGANWrapper)
        self.assertIn("APRIL-GAN", COMPLETED_MODELS)

    def test_cross_dataset_checkpoint_selection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {}
            for dataset, value in (("mvtec", 1.0), ("visa", 2.0)):
                path = root / f"{dataset}_pretrained.pth"
                layer = torch.nn.Linear(1, 1)
                with torch.no_grad():
                    layer.weight.fill_(value)
                    layer.bias.zero_()
                torch.save({"trainable_linearlayer": layer.state_dict()}, path)
                paths[dataset] = str(path)

            wrapper = APRILGANWrapper(checkpoint_paths=paths, device="cpu")
            wrapper.model = torch.nn.Identity()
            wrapper.linear_layer = torch.nn.Linear(1, 1)
            wrapper.prepare_for_dataset("MVTec AD")
            self.assertEqual(wrapper.active_weight_dataset, "visa")
            self.assertAlmostEqual(wrapper.linear_layer.weight.item(), 2.0)
            wrapper.prepare_for_dataset("VisA")
            self.assertEqual(wrapper.active_weight_dataset, "mvtec")
            self.assertAlmostEqual(wrapper.linear_layer.weight.item(), 1.0)

    def test_full_metric_map_is_compacted_only_for_artifacts(self):
        wrapper = APRILGANWrapper(device="cpu")
        maps = np.zeros((2, 518, 518), dtype=np.float32)
        self.assertEqual(wrapper.prepare_metric_map(maps[0]).shape, (518, 518))
        self.assertEqual(wrapper.prepare_artifact_maps(maps).shape, (2, 37, 37))

    def test_kaggle_notebook_is_valid_and_pins_official_source(self):
        notebook_path = Path(__file__).resolve().parents[1] / "kaggle_final_aprilgan.ipynb"
        payload = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in payload["cells"]
        )
        self.assertIn("ByChelsea/VAND-APRIL-GAN", source)
        self.assertIn(APRILGANWrapper.OFFICIAL_SOURCE_COMMIT, source)
        self.assertIn("mvtec_pretrained.pth", source)
        self.assertIn("visa_pretrained.pth", source)
        self.assertIn('models=["APRIL-GAN"]', source)
        self.assertIn("WEIGHT_DATASET = \"visa\" if IS_MVTEC else \"mvtec\"", source)


if __name__ == "__main__":
    unittest.main()
