import json
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from few_shot.harness.models import (
    PROMPTAD_DATASET_CATEGORIES,
    PromptADWrapper,
    discover_promptad_checkpoints,
)
from few_shot.harness.runner import run_promptad_evaluations


class _FakePromptADModel:
    def __init__(self):
        self.loaded_value = None

    @staticmethod
    def transform(image):
        pixels = np.asarray(image, dtype=np.float32)
        return torch.from_numpy(pixels.copy()).permute(2, 0, 1)

    def load_state_dict(self, state, strict=False):
        self.loaded_value = float(state["text_features"][0, 0])

    def __call__(self, images, task):
        batch_size = images.shape[0]
        if task == "cls":
            return [self.loaded_value] * batch_size, [np.full((2, 2), 0.5)] * batch_size

    @staticmethod
    def encode_image(images):
        return images

    def calculate_textual_anomaly_score(self, visual_features, task):
        return torch.full((visual_features.shape[0], 1, 15, 15), self.loaded_value * 2)

    def calculate_visual_anomaly_score(self, visual_features):
        return torch.full((visual_features.shape[0], 1, 15, 15), self.loaded_value * 2)


def _fake_state(value):
    return {
        "feature_gallery1": torch.zeros((225, 896)),
        "feature_gallery2": torch.zeros((225, 896)),
        "text_features": torch.full((2, 640), float(value)),
    }


class PromptADOfficialScoringTest(unittest.TestCase):
    def test_kaggle_notebook_is_clean_and_exposes_promptad_suite(self):
        notebook_path = Path(__file__).resolve().parents[1] / "kaggle_promptad.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                ast.parse("".join(cell["source"]), filename=f"cell-{index}")
        required = (
            "parsagholami/promptad-few-shot-checkpoints-mvtec-ad-and-visa",
            'PROMPTAD_COMMIT = "0f86ce0dc1ed59007d51348d8d566aed31360cf9"',
            "discover_promptad_checkpoints",
            "run_promptad_evaluations",
            "VERIFY_CHECKPOINT_HASHES = True",
            'DATASET_NAME = "visa"',
            "USE_CATEGORIZED_CORRUPTIONS = True",
            "INCLUDE_CLEAN_BASELINE = True",
            "SEVERITY_LEVELS = [1, 2, 3, 4]",
            "SHOTS_TO_RUN = [1, 2, 4]",
            "CORRUPTION_CACHE_ROOT = None",
            "STRICT_SOURCE_COMMIT = True",
            "Checkpoint preflight passed",
            "Dataset preflight passed",
            "Corruption smoke test passed",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)

    def test_discovery_reads_complete_class_task_index(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            package = Path(temporary_directory) / "promptad_retrained_visa_1shot"
            index = {}
            for category in PROMPTAD_DATASET_CATEGORIES["visa"]:
                for task in ("cls", "seg"):
                    relative = Path("result") / f"{task}-{category}.pt"
                    checkpoint = package / relative
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint.touch()
                    index[f"visa/1-shot/{category}/{task}"] = {
                        "path": relative.as_posix()
                    }
            (package / "checkpoint_index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )

            discovered = discover_promptad_checkpoints(
                temporary_directory, shots=[1], datasets=["visa"]
            )

        self.assertEqual(set(discovered), {1})
        self.assertEqual(
            set(discovered[1]["visa"]),
            set(PROMPTAD_DATASET_CATEGORIES["visa"]),
        )
        self.assertEqual(
            set(discovered[1]["visa"]["candle"]), {"cls", "seg"}
        )

    def test_cls_score_and_seg_map_use_their_paired_buffers(self):
        wrapper = PromptADWrapper(shot=1, device="cuda")
        wrapper.device = "cpu"
        wrapper.model = _FakePromptADModel()
        wrapper.active_dataset = "visa"
        wrapper.active_category = "candle"
        wrapper._cls_state = _fake_state(0.25)
        wrapper._seg_state = _fake_state(0.75)

        image = Image.new("RGB", (2, 2), color=(10, 20, 30))
        scores, maps = wrapper.forward_raw_batch([image], category="candle")

        expected_image_score = 1.0 / (1.0 / 0.5 + 1.0 / 0.25)
        np.testing.assert_allclose(scores, [expected_image_score])
        np.testing.assert_allclose(maps, 0.75)
        self.assertEqual(maps.shape, (1, 15, 15))

    def test_preprocess_reproduces_released_bgr_test_path(self):
        wrapper = PromptADWrapper(shot=1, device="cuda")
        wrapper.model = _FakePromptADModel()
        image = Image.new("RGB", (1, 1), color=(10, 20, 30))

        tensor = wrapper._preprocess_image(image)

        self.assertEqual(tensor[:, 0, 0].tolist(), [30.0, 20.0, 10.0])
        self.assertEqual(tuple(tensor.shape), (3, 1024, 1024))

    def test_metric_map_and_mask_use_official_400_space(self):
        wrapper = PromptADWrapper(shot=4, device="cuda")
        raw_map = np.arange(15 * 15, dtype=np.float32).reshape(15, 15) / 225
        metric_map = wrapper.prepare_metric_map(raw_map)
        expected_map = F.interpolate(
            torch.from_numpy(raw_map)[None, None],
            size=(400, 400),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        expected_map = gaussian_filter(expected_map, sigma=4)
        mask = np.zeros((83, 117), dtype=np.float32)
        mask[20:60, 30:90] = 1
        metric_mask = wrapper.prepare_metric_mask(mask)

        self.assertEqual(metric_map.shape, (400, 400))
        np.testing.assert_allclose(metric_map, expected_map, rtol=1e-6, atol=1e-6)
        self.assertEqual(metric_mask.shape, (400, 400))
        self.assertEqual(set(np.unique(metric_mask)), {0.0, 1.0})

    def test_suite_launches_selected_shot_with_class_pairs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            visa_root = root / "visa"
            source_root = root / "PromptAD-source"
            visa_root.mkdir()
            (source_root / "PromptAD" / "CLIPAD").mkdir(parents=True)
            (source_root / "PromptAD" / "model.py").touch()
            (source_root / "PromptAD" / "CLIPAD" / "factory.py").touch()
            mapping = {1: {"visa": {}}}
            for category in PROMPTAD_DATASET_CATEGORIES["visa"]:
                mapping[1]["visa"][category] = {}
                for task in ("cls", "seg"):
                    checkpoint = root / "weights" / f"{task}-{category}.pt"
                    checkpoint.parent.mkdir(exist_ok=True)
                    checkpoint.touch()
                    mapping[1]["visa"][category][task] = str(checkpoint)

            with patch("few_shot.harness.runner.run_evaluation") as run_mock:
                run_promptad_evaluations(
                    mvtec_root=None,
                    visa_root=str(visa_root),
                    output_root=str(root / "outputs"),
                    promptad_root=str(source_root),
                    checkpoint_paths=mapping,
                    shots=[1],
                    datasets=["visa"],
                )

        run_mock.assert_called_once()
        kwargs = run_mock.call_args.kwargs
        self.assertEqual(kwargs["models"], ["PromptAD"])
        self.assertEqual(kwargs["dataset"], "visa")
        self.assertEqual(
            len(kwargs["model_kwargs"]["PromptAD"]["checkpoint_paths"]["visa"]),
            12,
        )


if __name__ == "__main__":
    unittest.main()
