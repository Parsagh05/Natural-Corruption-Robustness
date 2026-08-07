import json
import tempfile
import unittest
from pathlib import Path

from zero_shot.scripts.finalize_result_files import (
    build_merged_document,
    discover_split_json_groups,
    merge_json_parts,
)


def _document(records):
    return {
        "model": "FB-CLIP",
        "dataset": "VisA",
        "protocol": "categorized_fine_grained_assigned_subsets",
        "description": "test",
        "records": records,
    }


def _record(condition, sample_id, score):
    return {
        "model": "FB-CLIP",
        "dataset": "VisA",
        "class_name": "candle",
        "transformation_level": condition,
        "sample_id": sample_id,
        "score": score,
    }


class FinalizeResultFilesTest(unittest.TestCase):
    def _write(self, path, document):
        path.write_text(json.dumps(document), encoding="utf-8")

    def test_discovery_includes_canonical_and_both_numbered_styles(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base = root / "FB-CLIP_VisA_FINE_GRAINED_PER_IMAGE"
            canonical = base.with_suffix(".json")
            part_one = root / f"{base.name}1.json"
            part_two = root / f"{base.name}_2.json"
            for path in (canonical, part_one, part_two):
                self._write(path, _document([]))

            groups = discover_split_json_groups([root])

            self.assertEqual(
                groups[canonical.resolve()],
                [canonical.resolve(), part_one.resolve(), part_two.resolve()],
            )

    def test_merge_preserves_canonical_records_and_deduplicates_parts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            canonical = root / "FB-CLIP_VisA_FINE_GRAINED_PER_IMAGE.json"
            continuation = root / "FB-CLIP_VisA_FINE_GRAINED_PER_IMAGE1.json"
            clean = _record("clean_level 0", "normal.png", 0.1)
            noise = _record("gaussian_noise_level 1", "bad.png", 0.8)
            self._write(canonical, _document([clean]))
            self._write(continuation, _document([clean, noise]))

            merged, duplicate_count = build_merged_document(
                [canonical, continuation]
            )

            self.assertEqual(duplicate_count, 1)
            self.assertEqual(merged["records"], [clean, noise])

            changed, record_count, duplicate_count = merge_json_parts(
                [canonical, continuation], canonical
            )
            self.assertTrue(changed)
            self.assertEqual(record_count, 2)
            self.assertEqual(duplicate_count, 1)
            written = json.loads(canonical.read_text(encoding="utf-8"))
            self.assertEqual(written, merged)

            changed, record_count, duplicate_count = merge_json_parts(
                [canonical, continuation], canonical
            )
            self.assertFalse(changed)
            self.assertEqual(record_count, 2)
            self.assertEqual(duplicate_count, 2)


if __name__ == "__main__":
    unittest.main()
