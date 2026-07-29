import json
import unittest
from pathlib import Path


class AACLIPNotebookConfigTest(unittest.TestCase):
    def _code_source(self, notebook_name):
        zero_shot_root = Path(__file__).resolve().parents[1]
        notebook = json.loads(
            (zero_shot_root / notebook_name).read_text(encoding="utf-8")
        )
        return "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        )

    def test_zero_shot_checkpoint_mapping_is_cross_dataset(self):
        source = self._code_source("kaggle_final_aaclip.ipynb")

        self.assertIn(
            'AACLIP_TRAIN_DATASET = "VisA" if IS_MVTEC else "MVTec"',
            source,
        )
        self.assertIn(
            'AACLIP_TRAIN_DIR = f"TrainOn{AACLIP_TRAIN_DATASET}"',
            source,
        )
        self.assertIn(
            "if AACLIP_TRAIN_DATASET.lower() == DATASET_NAME:",
            source,
        )
        self.assertNotIn(
            'AACLIP_TRAIN_DIR = "TrainOnMVTec" if IS_MVTEC else "TrainOnVisA"',
            source,
        )

    def test_evaluation_notebooks_load_harness_and_shared_assets(self):
        benchmark_path = "/kaggle/working/Natural-Corruption-Robustness"
        for notebook_name in (
            "kaggle_final_aaclip.ipynb",
            "kaggle_final_afclip.ipynb",
            "kaggle_final_anomalyclip.ipynb",
        ):
            with self.subTest(notebook=notebook_name):
                source = self._code_source(notebook_name)
                self.assertIn(benchmark_path, source)
                self.assertIn("zero_shot", source)
                self.assertIn("corruption_plan_path", source)


if __name__ == "__main__":
    unittest.main()
