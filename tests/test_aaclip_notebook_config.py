import json
import unittest
from pathlib import Path


class AACLIPNotebookConfigTest(unittest.TestCase):
    def test_zero_shot_checkpoint_mapping_is_cross_dataset(self):
        repository_root = Path(__file__).resolve().parents[1]
        notebook_path = repository_root / "kaggle_final_aaclip.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        )

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


if __name__ == "__main__":
    unittest.main()
