import unittest
from pathlib import Path
from unittest.mock import patch

from shared import CORRUPTION_PLAN_ROOT, corruption_plan_path
from shared.corruption import apply_corruption, stable_corruption_seed
from shared.imagenet_c.corruptions import brightness


class _FakeGrayscaleMotionImage:
    def __init__(self, blob):
        self.blob = blob

    def motion_blur(self, radius, sigma, angle):
        return None

    def make_blob(self):
        import cv2
        import numpy as np

        grayscale = np.arange(200 * 300, dtype=np.uint8).reshape(200, 300)
        success, encoded = cv2.imencode(".png", grayscale)
        if not success:
            raise RuntimeError("Test PNG encoding failed.")
        return encoded.tobytes()


class SharedCorruptionAssetsTest(unittest.TestCase):
    def test_motion_blur_preserves_non_224_grayscale_image_dimensions(self):
        from PIL import Image
        from shared.imagenet_c import corruptions

        image = Image.new("RGB", (300, 200), color=(128, 128, 128))
        with patch.object(
            corruptions, "MotionImage", _FakeGrayscaleMotionImage
        ):
            result = corruptions.motion_blur(image, severity=1)

        self.assertEqual(result.shape, (200, 300, 3))
        self.assertTrue((result[..., 0] == result[..., 1]).all())
        self.assertTrue((result[..., 1] == result[..., 2]).all())

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
