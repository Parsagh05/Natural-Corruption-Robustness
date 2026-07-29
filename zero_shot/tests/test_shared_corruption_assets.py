import unittest
from pathlib import Path

from shared import CORRUPTION_PLAN_ROOT, corruption_plan_path
from shared.corruption import apply_corruption, stable_corruption_seed
from shared.imagenet_c.corruptions import brightness


class SharedCorruptionAssetsTest(unittest.TestCase):
    def test_dataset_plans_resolve_to_shared_directory(self):
        for dataset in ("mvtec", "visa"):
            with self.subTest(dataset=dataset):
                plan_path = corruption_plan_path(dataset)
                self.assertEqual(plan_path.parent, CORRUPTION_PLAN_ROOT)
                self.assertTrue(plan_path.is_file())

    def test_legacy_zero_shot_asset_locations_are_empty(self):
        zero_shot_root = Path(__file__).resolve().parents[1]
        self.assertFalse((zero_shot_root / "imagenet_c").exists())
        self.assertFalse((zero_shot_root / "mvtec_corruption_plan.csv").exists())
        self.assertFalse((zero_shot_root / "visa_corruption_plan.csv").exists())

    def test_shared_corruption_package_is_importable(self):
        self.assertTrue(callable(brightness))
        self.assertTrue(callable(apply_corruption))
        self.assertEqual(
            stable_corruption_seed(111, "sample.png", "brightness", 1),
            stable_corruption_seed(111, "sample.png", "brightness", 1),
        )


if __name__ == "__main__":
    unittest.main()
