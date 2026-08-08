import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F

from few_shot.harness.models import (
    AFCLIP_DATASET_CATEGORIES,
    AFCLIPFewShotWrapper,
    MODEL_REGISTRY,
    discover_afclip_checkpoints,
)
from few_shot.harness.runner import run_afclip_evaluations
from zero_shot.harness.config import DatasetConfig


class _FakeAFCLIPModel:
    def __init__(self):
        self.memorybank = None
        self.support_shape = None

    def store_memory(self, images, args):
        self.support_shape = tuple(images.shape)
        self.memorybank = [torch.ones((1, 2))]

    @staticmethod
    def detect_forward(images, args):
        batch = images.shape[0]
        scores = torch.arange(1, batch + 1, dtype=torch.float32)
        maps = torch.arange(batch * 9, dtype=torch.float32).reshape(batch, 1, 3, 3)
        return scores, maps


class AFCLIPOfficialScoringTest(unittest.TestCase):
    def test_wrapper_is_registered_and_not_a_separate_fewshot_checkpoint(self):
        self.assertIs(MODEL_REGISTRY["AF-CLIP+"], AFCLIPFewShotWrapper)
        wrapper = AFCLIPFewShotWrapper(shot=4, device="cpu")
        self.assertEqual(wrapper.model_name, "AF-CLIP+-4-shot")
        self.assertEqual(wrapper.OFFICIAL_ALPHA, 0.1)
        self.assertEqual(wrapper.OFFICIAL_MEMORY_LAYERS, (6, 12, 18, 24))

    def test_discovery_finds_exact_released_prompt_and_adaptor_suite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            weight_root = Path(temporary_directory) / "AF-CLIP" / "weight"
            weight_root.mkdir(parents=True)
            for dataset in ("mvtec", "visa"):
                for component in ("prompt", "adaptor"):
                    (weight_root / f"{dataset}_{component}.pt").touch()
            discovered = discover_afclip_checkpoints(temporary_directory)

        self.assertEqual(set(discovered), {"mvtec", "visa"})
        self.assertEqual(set(discovered["mvtec"]), {"prompt", "adaptor"})
        self.assertTrue(discovered["visa"]["prompt"].endswith("visa_prompt.pt"))

    def test_mvtec_support_selection_is_seeded_normal_only_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for category in AFCLIP_DATASET_CATEGORIES["mvtec"]:
                normal_dir = root / category / "train" / "good"
                normal_dir.mkdir(parents=True)
                for index in range(6):
                    Image.new("RGB", (4, 4), color=(index, 0, 0)).save(
                        normal_dir / f"{index:03d}.png"
                    )
                bad_dir = root / category / "test" / "bad"
                bad_dir.mkdir(parents=True)
                Image.new("RGB", (4, 4)).save(bad_dir / "bad.png")

            first = AFCLIPFewShotWrapper(
                shot=2,
                reference_seed=123,
                dataset_roots={"mvtec": str(root)},
                device="cpu",
            )._select_support_paths("mvtec")
            second = AFCLIPFewShotWrapper(
                shot=2,
                reference_seed=123,
                dataset_roots={"mvtec": str(root)},
                device="cpu",
            )._select_support_paths("mvtec")

        self.assertEqual(first, second)
        self.assertEqual(set(first), set(AFCLIP_DATASET_CATEGORIES["mvtec"]))
        self.assertTrue(all(len(paths) == 2 for paths in first.values()))
        self.assertTrue(all("train" in path.parts and "good" in path.parts for paths in first.values() for path in paths))

    def test_visa_support_reader_uses_only_normal_training_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_root = root / "candle" / "Data" / "Images"
            normal_root = image_root / "Normal"
            anomaly_root = image_root / "Anomaly"
            normal_root.mkdir(parents=True)
            anomaly_root.mkdir(parents=True)
            for name in ("a.JPG", "b.JPG", "test.JPG"):
                Image.new("RGB", (4, 4)).save(normal_root / name)
            Image.new("RGB", (4, 4)).save(anomaly_root / "bad.JPG")
            split_root = root / "split_csv"
            split_root.mkdir()
            (split_root / "1cls.csv").write_text(
                "object,split,label,image,mask\n"
                "candle,train,normal,candle/Data/Images/Normal/a.JPG,\n"
                "candle,train,normal,candle/Data/Images/Normal/b.JPG,\n"
                "candle,train,anomaly,candle/Data/Images/Anomaly/bad.JPG,x\n"
                "candle,test,normal,candle/Data/Images/Normal/test.JPG,\n",
                encoding="utf-8",
            )
            wrapper = AFCLIPFewShotWrapper(
                shot=2,
                dataset_roots={"visa": str(root)},
                device="cpu",
            )
            paths = wrapper._normal_training_paths("visa", "candle")

        self.assertEqual({path.name for path in paths}, {"a.JPG", "b.JPG"})

    def test_forward_builds_memory_from_clean_support_and_preserves_raw_map(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            support_paths = []
            for index in range(2):
                path = Path(temporary_directory) / f"support-{index}.png"
                Image.new("RGB", (20, 10), color=(10 + index, 20, 30)).save(path)
                support_paths.append(path)

            wrapper = AFCLIPFewShotWrapper(shot=2, device="cpu")
            wrapper.model = _FakeAFCLIPModel()
            wrapper._args = object()
            wrapper.active_dataset = "mvtec"
            wrapper.support_paths = {"bottle": tuple(support_paths)}
            images = [Image.new("RGB", (11, 17)), Image.new("RGB", (19, 13))]
            scores, maps = wrapper.forward_raw_batch(images, category="bottle")

        np.testing.assert_array_equal(scores, np.array([1, 2], dtype=np.float32))
        np.testing.assert_array_equal(
            maps,
            np.arange(18, dtype=np.float32).reshape(2, 3, 3),
        )
        self.assertEqual(wrapper.model.support_shape, (2, 3, 518, 518))

    def test_metric_map_matches_official_bilinear_then_gaussian_path(self):
        wrapper = AFCLIPFewShotWrapper(shot=1, device="cpu")
        raw = np.arange(9, dtype=np.float32).reshape(3, 3)
        actual = wrapper.prepare_metric_map(raw)
        resized = F.interpolate(
            torch.from_numpy(raw)[None, None],
            size=(518, 518),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        expected = gaussian_filter(resized, sigma=4).astype(np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

        mask = np.zeros((13, 21), dtype=np.float32)
        mask[2:9, 5:16] = 1
        metric_mask = wrapper.prepare_metric_mask(mask)
        self.assertEqual(metric_mask.shape, (518, 518))
        self.assertEqual(set(np.unique(metric_mask)), {0.0, 1.0})

    def test_suite_runner_keeps_shot_seed_outputs_distinct(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "AF-CLIP"
            (source / "clip").mkdir(parents=True)
            (source / "clip" / "clip.py").touch()
            (source / "clip" / "model.py").touch()
            weights = source / "weight"
            weights.mkdir()
            for dataset in ("mvtec", "visa"):
                for component in ("prompt", "adaptor"):
                    (weights / f"{dataset}_{component}.pt").touch()
            checkpoints = discover_afclip_checkpoints(str(weights))
            mvtec_root = root / "mvtec"
            visa_root = root / "visa"
            mvtec_root.mkdir()
            visa_root.mkdir()
            configs = [
                DatasetConfig("MVTec", mvtec_root, []),
                DatasetConfig("VisA", visa_root, []),
            ]

            with patch(
                "few_shot.harness.runner.build_dataset_configs",
                return_value=configs,
            ), patch("few_shot.harness.runner.run_evaluation") as run_mock:
                run_afclip_evaluations(
                    mvtec_root=str(mvtec_root),
                    visa_root=str(visa_root),
                    output_root=str(root / "outputs"),
                    afclip_root=str(source),
                    checkpoint_paths=checkpoints,
                    shots=[1, 2],
                    reference_seeds=[7, 8],
                    datasets=["mvtec", "visa"],
                    corruption_types=[],
                    severity_levels=[],
                    categorized_corruptions=False,
                    include_clean=True,
                )

        self.assertEqual(run_mock.call_count, 4)
        result_names = {
            call.kwargs["model_kwargs"]["AF-CLIP+"]["result_name"]
            for call in run_mock.call_args_list
        }
        self.assertEqual(
            result_names,
            {
                "AF-CLIP+-1-shot-seed-7",
                "AF-CLIP+-1-shot-seed-8",
                "AF-CLIP+-2-shot-seed-7",
                "AF-CLIP+-2-shot-seed-8",
            },
        )

    def test_kaggle_notebook_is_clean_and_exposes_complete_suite(self):
        notebook_path = Path(__file__).resolve().parents[1] / "kaggle_afclip.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                ast.parse("".join(cell["source"]), filename=f"cell-{index}")
        required = (
            'AFCLIP_COMMIT = "bb7edec4128a76f29cb573cd3002538bf250b2fe"',
            "discover_afclip_checkpoints",
            "run_afclip_evaluations",
            'DATASET_NAME = "visa"',
            "SHOTS_TO_RUN = [1, 2, 4]",
            "REFERENCE_SEEDS = [111]",
            "USE_CATEGORIZED_CORRUPTIONS = True",
            "INCLUDE_CLEAN_BASELINE = True",
            "Checkpoint preflight passed",
            "Support preflight passed",
            "Corruption smoke test passed",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)


if __name__ == "__main__":
    unittest.main()
