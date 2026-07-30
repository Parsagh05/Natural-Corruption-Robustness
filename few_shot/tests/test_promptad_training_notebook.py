import ast
import json
from pathlib import Path
import unittest


class PromptADTrainingNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(__file__).resolve().parents[1] / "kaggle_train_promptad.ipynb"
        cls.notebook = json.loads(cls.path.read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell.get("source", [])) for cell in cls.notebook["cells"]
        )

    def test_notebook_is_clean_and_all_code_cells_parse(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            ast.parse("".join(cell["source"]), filename=f"cell-{index}")

    def test_notebook_pins_official_source_and_paper_setup(self):
        required_fragments = (
            'PROMPTAD_COMMIT = "0f86ce0dc1ed59007d51348d8d566aed31360cf9"',
            'SEED = 111',
            'IMAGE_RESIZE = 240',
            'IMAGE_CROP = 240',
            'METRIC_RESOLUTION = 400',
            'BATCH_SIZE = 400',
            'BACKBONE = "ViT-B-16-plus-240"',
            'PRETRAINED_DATASET = "laion400m_e32"',
            'EPOCHS = 100',
            'LEARNING_RATE = 0.002',
            'MOMENTUM = 0.9',
            'WEIGHT_DECAY = 0.0005',
            'ALIGNMENT_WEIGHT = 0.001',
            'NORMAL_CONTEXT_TOKENS = 4',
            'ANOMALY_CONTEXT_TOKENS = 1',
            'LEARNABLE_ANOMALY_SUFFIXES = 4',
            'NORMAL_PROTOTYPES_BY_TASK = {"cls": 3, "seg": 1}',
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.source)

    def test_notebook_preserves_official_suite_and_exports_auditable_weights(self):
        required_fragments = (
            'SHOTS_TO_TRAIN = [1, 2, 4]',
            'TASKS_TO_TRAIN = ["cls", "seg"]',
            'selected_samples_per_run.txt',
            'train_cls.py',
            'train_seg.py',
            'RESUME_RESULT_ROOT',
            'MAX_JOBS_THIS_SESSION',
            'INSTRUMENTED_SCRIPTS',
            'train_{task_name}_tqdm.py',
            'for epoch in tqdm(range(args.Epoch)',
            'desc="PromptAD checkpoints"',
            'expected_keys = {"feature_gallery1", "feature_gallery2", "text_features"}',
            'checkpoint_index.json',
            'training_manifest.json',
            'sha256',
            'archive.testzip()',
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.source)


if __name__ == "__main__":
    unittest.main()
