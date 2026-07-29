import csv
import tempfile
import unittest
from pathlib import Path

from zero_shot.harness.result_order import condition_sort_key, parse_condition
from zero_shot.scripts.normalize_result_csvs import normalize_csv


class ResultConditionOrderTest(unittest.TestCase):
    def test_protocol_order_places_clean_first_and_sorts_severity_numerically(self):
        conditions = [
            "contrast_level 4",
            "geometric_level 1",
            "gaussian_noise_level 10",
            "clean_level 0",
            "brightness_level 1",
            "gaussian_noise_level 2",
            "blur_level 2",
            "noise_level 1",
            "shot_noise_level 1",
            "gaussian_noise_level 1",
        ]

        self.assertEqual(
            sorted(conditions, key=condition_sort_key),
            [
                "clean_level 0",
                "gaussian_noise_level 1",
                "gaussian_noise_level 2",
                "gaussian_noise_level 10",
                "shot_noise_level 1",
                "brightness_level 1",
                "contrast_level 4",
                "noise_level 1",
                "blur_level 2",
                "geometric_level 1",
            ],
        )

    def test_unknown_corruptions_are_supported_after_configured_corruptions(self):
        conditions = [
            "zeta_effect_level 1",
            "brightness_level 1",
            "alpha_effect_level 2",
            "alpha_effect_level 1",
        ]

        self.assertEqual(
            sorted(conditions, key=condition_sort_key),
            [
                "brightness_level 1",
                "alpha_effect_level 1",
                "alpha_effect_level 2",
                "zeta_effect_level 1",
            ],
        )

    def test_categorized_geometric_subtypes_follow_protocol_order(self):
        conditions = [
            "shifting_level 1",
            "zooming_level 1",
            "rotation_level 1",
        ]

        self.assertEqual(
            sorted(conditions, key=condition_sort_key),
            [
                "rotation_level 1",
                "zooming_level 1",
                "shifting_level 1",
            ],
        )

    def test_parse_condition_rejects_malformed_values(self):
        with self.assertRaises(ValueError):
            parse_condition("brightness severity 1")

    def test_normalizer_preserves_every_nonempty_cell_exactly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "Model_MVTec_PX.csv"
            csv_path.write_text(
                "transformation_level,mean_auroc_px,mean_threshold_px\n"
                "contrast_level 2,91.20,0.123400\n"
                "clean_level 0,99.50,1.000000\n"
                ",,\n"
                "contrast_level 1,93.00,0.234500\n",
                encoding="utf-8",
            )

            with csv_path.open("r", encoding="utf-8", newline="") as source:
                before = [
                    row
                    for row in csv.reader(source)
                    if row and any(value for value in row)
                ]

            changed, removed_empty_rows = normalize_csv(csv_path)

            with csv_path.open("r", encoding="utf-8", newline="") as source:
                after = list(csv.reader(source))

            self.assertTrue(changed)
            self.assertEqual(removed_empty_rows, 1)
            self.assertEqual(after[0], before[0])
            self.assertCountEqual(after[1:], before[1:])
            self.assertEqual(
                [row[0] for row in after[1:]],
                ["clean_level 0", "contrast_level 1", "contrast_level 2"],
            )
            self.assertIn(["contrast_level 2", "91.20", "0.123400"], after)
            self.assertIn(["clean_level 0", "99.50", "1.000000"], after)


if __name__ == "__main__":
    unittest.main()
