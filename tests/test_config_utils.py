from pathlib import Path
import sys
import unittest
import warnings


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.config_utils import round_to_nearest_multiple


class RoundToNearestMultipleTest(unittest.TestCase):
    def test_preserves_exact_multiple_without_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            self.assertEqual(round_to_nearest_multiple(12, 3, name="count"), 12)
        self.assertEqual(caught, [])

    def test_rounds_to_nearest_multiple_and_warns(self):
        with self.assertWarnsRegex(RuntimeWarning, "count=13.*using 12"):
            self.assertEqual(round_to_nearest_multiple(13, 4, name="count"), 12)
        with self.assertWarnsRegex(RuntimeWarning, "count=15.*using 16"):
            self.assertEqual(round_to_nearest_multiple(15, 4, name="count"), 16)

    def test_rounds_ties_up_and_keeps_result_positive(self):
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(round_to_nearest_multiple(6, 4), 8)
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(round_to_nearest_multiple(1, 4), 4)

    def test_rejects_nonpositive_inputs(self):
        with self.assertRaisesRegex(ValueError, "count must be positive"):
            round_to_nearest_multiple(0, 2, name="count")
        with self.assertRaisesRegex(ValueError, "multiple must be positive"):
            round_to_nearest_multiple(2, 0)

    def test_cross_validation_defaults_normalize_for_every_fold_size(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for task_count in (6, 7, 9):
                for value in (1_536, 256, 1_000_000):
                    normalized = round_to_nearest_multiple(value, task_count)
                    self.assertGreater(normalized, 0)
                    self.assertEqual(normalized % task_count, 0)


if __name__ == "__main__":
    unittest.main()
