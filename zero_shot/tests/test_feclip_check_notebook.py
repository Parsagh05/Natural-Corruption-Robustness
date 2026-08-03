import ast
import json
import unittest
from pathlib import Path


class FECLIPCheckNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook_path = Path(__file__).parents[1] / "kaggle_check_feclip.ipynb"
        cls.notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell.get("source", [])) for cell in cls.notebook["cells"]
        )

    def test_all_python_cells_parse(self):
        for index, cell in enumerate(self.notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(
                line
                for line in cell.get("source", [])
                if not line.lstrip().startswith(("%", "!"))
            )
            with self.subTest(cell=index):
                ast.parse(source, filename=f"cell_{index}")

    def test_final_hugging_face_archive_is_pinned(self):
        self.assertIn(
            "HF_REPO_ID = 'Parsagh1383/FE-CLIP_Reproduction_Checkpoint'",
            self.source,
        )
        self.assertIn("HF_REPO_TYPE = 'dataset'", self.source)
        self.assertIn(
            "7821723a70ad54720f78d46f46d0e812f26b0d394280c8eaa5d8a76a19751499",
            self.source,
        )

    def test_cross_dataset_checkpoint_mapping_is_correct(self):
        self.assertIn(
            "'MVTec': checkpoint_root / 'train_on_visa_seed_111'",
            self.source,
        )
        self.assertIn(
            "'VisA': checkpoint_root / 'train_on_mvtec_seed_111'",
            self.source,
        )
        self.assertIn("EXPECTED_SOURCE = {'MVTec': 'VisA', 'VisA': 'MVTec'}", self.source)

    def test_paper_targets_are_exactly_encoded(self):
        expected = (
            "'MVTec': {'image_auroc': 91.9, 'image_ap': 96.5, "
            "'pixel_auroc': 92.6, 'pixel_pro': 88.3}",
            "'VisA':  {'image_auroc': 84.6, 'image_ap': 86.6, "
            "'pixel_auroc': 95.9, 'pixel_pro': 92.8}",
        )
        for fragment in expected:
            self.assertIn(fragment, self.source)

    def test_paper_family_evaluation_protocol_is_present(self):
        self.assertIn("GAUSSIAN_SIGMA = 4.0", self.source)
        self.assertIn("PRO_MAX_STEPS = 200", self.source)
        self.assertIn("PRO_FPR_LIMIT = 0.3", self.source)
        self.assertIn("def cal_pro_score(", self.source)
        self.assertIn("measure.regionprops(measure.label(mask))", self.source)
        self.assertIn("roc_auc_score(masks.reshape(-1)", self.source)
        self.assertIn("average_precision_score(labels, scores)", self.source)

    def test_metrics_are_macro_averaged_across_categories(self):
        self.assertIn("category_metrics(all_labels, all_scores, all_masks, all_maps)", self.source)
        self.assertIn("float(category_frame[metric].mean())", self.source)
        self.assertIn(
            "'aggregation': 'metric_per_category_then_unweighted_category_mean'",
            self.source,
        )

    def test_structural_and_numerical_verdicts_are_separate(self):
        self.assertIn("'structural_integrity': 'PASS'", self.source)
        self.assertIn("EXACT_AT_PAPER_PRECISION", self.source)
        self.assertIn("CLOSE_ONE_SEED_REPRODUCTION", self.source)
        self.assertIn("NOT_NUMERICALLY_REPRODUCED_WITHIN_TOLERANCE", self.source)
        self.assertIn("CLOSE_TOLERANCE_PP = 2.0", self.source)

    def test_checkpoint_components_restore_strictly(self):
        self.assertIn("model.ffe_adapters.load_state_dict", self.source)
        self.assertIn("model.lfs_adapters.load_state_dict", self.source)
        self.assertIn("model.patch_projection.load_state_dict", self.source)
        self.assertIn("strict=True", self.source)
        self.assertIn("trainable_count == 9_184_000", self.source)


if __name__ == "__main__":
    unittest.main()
