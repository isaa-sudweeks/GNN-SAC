"""Check common-distribution aggregation and explicit missing-cohort handling."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('failure_report',
    Path(__file__).resolve().parents[1] / 'scripts/report_failure_balanced_replay.py')
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class FailureReplayReportTests(unittest.TestCase):
    def test_empty_cohort_is_not_a_zero_error_pass(self):
        result = report.prediction_summary([], 'terminal_ending')
        self.assertEqual(result['windows'], 0)
        self.assertIsNone(result['progress_mae'])
        self.assertEqual(report.ratio(0, 0), 'n/a (empty cohort)')

    def test_errors_are_weighted_by_windows_not_cell_count(self):
        def cell(count, error):
            return dict(horizon=10, cohort='ordinary', windows=count, true_collapses=0,
                any_imagined={'false_negatives': 0, 'false_positives': 1},
                endpoint={'false_negatives': 0}, nonfinite_windows=0,
                true_geometry_label_disagreements=0, progress_mae=error,
                position_rmse=error, velocity_rmse=error)
        result = report.prediction_summary([cell(1, 1.), cell(3, 3.)], 'ordinary')
        self.assertEqual(result['windows'], 4)
        self.assertEqual(result['progress_mae'], 2.5)
        self.assertAlmostEqual(result['position_rmse'], 7. ** .5)
        self.assertEqual(result['any_false_positives'], 2)


if __name__ == '__main__':
    unittest.main()
