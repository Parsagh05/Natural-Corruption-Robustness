import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from few_shot.harness.runner import run_official_evaluations
from few_shot.harness.runner import RobustnessRunner


class SixEvaluationSuiteTest(unittest.TestCase):
    def test_strict_model_failure_aborts_instead_of_recording_zero(self):
        class FailingModel:
            model_name = "INP-Former"
            fail_on_inference_error = True

            @staticmethod
            def forward_raw_batch(images, category=None):
                raise RuntimeError("batch failure")

            @staticmethod
            def forward_raw(image, category=None):
                raise RuntimeError("single failure")

        class OneSampleDataset:
            samples = [{
                "image_path": "widget/test/bad/001.png",
                "mask_path": "widget/ground_truth/bad/001_mask.png",
            }]

            def __len__(self):
                return 1

            def __getitem__(self, index):
                return {
                    "image": object(),
                    "mask": [[1]],
                    "label": 1,
                    "sample_id": "widget/bad/001.png",
                }

        runner = object.__new__(RobustnessRunner)
        runner.config = SimpleNamespace(
            batch_size=1,
            default_map_resolution=256,
            device="cpu",
        )
        with self.assertRaisesRegex(RuntimeError, "Strict inference failed"):
            runner._run_category(
                model_wrapper=FailingModel(),
                dataset_config=SimpleNamespace(name="MVTec"),
                category="widget",
                corruption_type="noise",
                severity=1,
                dataset=OneSampleDataset(),
            )

    def test_strict_few_shot_protocol_rejects_anomaly_without_mask(self):
        dataset = SimpleNamespace(samples=[{
            "is_anomaly": True,
            "mask_path": None,
            "sample_id": "widget/bad/001.png",
        }])

        with self.assertRaisesRegex(FileNotFoundError, "001.png"):
            RobustnessRunner._validate_anomaly_masks(
                dataset, "MVTec", "widget"
            )

    def test_suite_launches_three_shots_with_two_datasets_and_categorized_protocol(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            mvtec_root = root / "mvtec"
            visa_root = root / "visa"
            source_root = root / "INP-Former"
            mvtec_root.mkdir()
            visa_root.mkdir()
            (source_root / "models").mkdir(parents=True)
            (source_root / "dinov2" / "models").mkdir(parents=True)
            (source_root / "models" / "uad.py").touch()
            (source_root / "dinov2" / "models" / "vision_transformer.py").touch()
            checkpoints = {}
            for shot in (1, 2, 4):
                checkpoints[shot] = {}
                for dataset in ("mvtec", "visa"):
                    checkpoint = root / "weights" / str(shot) / dataset / "model.pth"
                    checkpoint.parent.mkdir(parents=True)
                    checkpoint.touch()
                    checkpoints[shot][dataset] = str(checkpoint)

            with patch("few_shot.harness.runner.run_evaluation") as run_mock:
                run_official_evaluations(
                    mvtec_root=str(mvtec_root),
                    visa_root=str(visa_root),
                    output_root=str(root / "outputs"),
                    inpformer_root=str(source_root),
                    checkpoint_paths=checkpoints,
                    device="cpu",
                )

        self.assertEqual(run_mock.call_count, 3)
        observed_shots = []
        for call in run_mock.call_args_list:
            kwargs = call.kwargs
            observed_shots.append(kwargs["shot"])
            self.assertEqual(kwargs["dataset"], "both")
            self.assertTrue(kwargs["categorized_corruptions"])
            self.assertEqual(
                kwargs["corruption_types"],
                ["noise", "blur", "photometric", "geometric"],
            )
            self.assertEqual(kwargs["severity_levels"], [1, 2, 3, 4])
            self.assertEqual(
                set(kwargs["model_kwargs"]["INP-Former"]["checkpoint_paths"]),
                {"mvtec", "visa"},
            )
        self.assertEqual(observed_shots, [1, 2, 4])
        self.assertEqual(len(observed_shots) * 2, 6)

    def test_suite_can_launch_one_selected_shot_for_both_datasets(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            mvtec_root = root / "mvtec"
            visa_root = root / "visa"
            source_root = root / "INP-Former"
            mvtec_root.mkdir()
            visa_root.mkdir()
            (source_root / "models").mkdir(parents=True)
            (source_root / "dinov2" / "models").mkdir(parents=True)
            (source_root / "models" / "uad.py").touch()
            (source_root / "dinov2" / "models" / "vision_transformer.py").touch()
            checkpoint_mapping = {2: {}}
            for dataset in ("mvtec", "visa"):
                checkpoint = root / "weights" / "2" / dataset / "model.pth"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.touch()
                checkpoint_mapping[2][dataset] = str(checkpoint)

            with patch("few_shot.harness.runner.run_evaluation") as run_mock:
                run_official_evaluations(
                    mvtec_root=str(mvtec_root),
                    visa_root=str(visa_root),
                    output_root=str(root / "outputs"),
                    inpformer_root=str(source_root),
                    checkpoint_paths=checkpoint_mapping,
                    shots=[2],
                    device="cpu",
                )

        run_mock.assert_called_once()
        self.assertEqual(run_mock.call_args.kwargs["shot"], 2)

    def test_suite_forwards_uncategorized_controls(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            mvtec_root = root / "mvtec"
            visa_root = root / "visa"
            source_root = root / "INP-Former"
            mvtec_root.mkdir()
            visa_root.mkdir()
            (source_root / "models").mkdir(parents=True)
            (source_root / "dinov2" / "models").mkdir(parents=True)
            (source_root / "models" / "uad.py").touch()
            (source_root / "dinov2" / "models" / "vision_transformer.py").touch()
            checkpoint_mapping = {1: {}}
            for dataset in ("mvtec", "visa"):
                checkpoint = root / "weights" / "1" / dataset / "model.pth"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.touch()
                checkpoint_mapping[1][dataset] = str(checkpoint)

            with patch("few_shot.harness.runner.run_evaluation") as run_mock:
                run_official_evaluations(
                    mvtec_root=str(mvtec_root),
                    visa_root=str(visa_root),
                    output_root=str(root / "outputs"),
                    inpformer_root=str(source_root),
                    checkpoint_paths=checkpoint_mapping,
                    shots=[1],
                    device="cpu",
                    corruption_types=["brightness", "contrast"],
                    severity_levels=[2, 4],
                    categorized_corruptions=False,
                    include_clean=False,
                )

        run_mock.assert_called_once()
        kwargs = run_mock.call_args.kwargs
        self.assertEqual(kwargs["corruption_types"], ["brightness", "contrast"])
        self.assertEqual(kwargs["severity_levels"], [2, 4])
        self.assertFalse(kwargs["categorized_corruptions"])
        self.assertIsNone(kwargs["categorized_corruption_plans"])
        self.assertFalse(kwargs["include_clean"])

    def test_kaggle_notebook_exposes_official_six_run_launcher(self):
        notebook_path = (
            Path(__file__).resolve().parents[1] / "kaggle_inpformer.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

        self.assertIn("run_official_evaluations", source)
        self.assertIn("OFFICIAL_CHECKPOINT_URLS", source)
        self.assertIn("DOWNLOAD_FROM_GOOGLE_DRIVE = True", source)
        self.assertIn("ATTACHED_CHECKPOINT_ROOT", source)
        self.assertIn("resume=True", source)
        self.assertIn("valid_torch_checkpoint_file", source)
        self.assertIn("zipfile.is_zipfile", source)
        self.assertIn("checkpoint_zip.testzip()", source)
        self.assertIn("Corruption smoke test passed", source)
        self.assertIn("Dataset preflight passed", source)
        self.assertIn("STRICT_SOURCE_COMMIT = True", source)
        self.assertIn("INCLUDE_CLEAN_BASELINE = True", source)
        self.assertIn("USE_CATEGORIZED_CORRUPTIONS = True", source)
        self.assertIn("UNCATEGORIZED_CORRUPTION_TYPES = [", source)
        self.assertIn("CATEGORIZED_CORRUPTION_TYPES = [", source)
        self.assertIn("CORRUPTION_TYPES = (", source)
        self.assertIn("SEVERITY_LEVELS = [1, 2, 3, 4]", source)
        self.assertIn("SHOTS_TO_RUN = [1, 2, 4]", source)
        self.assertIn("CORRUPTION_CACHE_ROOT = None", source)
        self.assertIn("shots=SHOTS_TO_RUN", source)
        self.assertIn("corruption_types=CORRUPTION_TYPES", source)
        self.assertIn("severity_levels=SEVERITY_LEVELS", source)
        self.assertIn(
            "categorized_corruptions=USE_CATEGORIZED_CORRUPTIONS", source
        )


if __name__ == "__main__":
    unittest.main()
