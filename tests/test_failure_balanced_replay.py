"""Regression tests for failure strata, exact uniform draws and sampler restarts."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location('failure_helpers',
    Path(__file__).resolve().parents[1] / 'scripts/failure_balanced_helpers.py')
helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helpers)


def sample_arrays():
    return dict(episode=np.repeat([0, 1, 2], 12),
                terminated=np.isin(np.arange(36), [11, 35]),
                truncated=np.isin(np.arange(36), [23]))


class FailureBalancedReplayTests(unittest.TestCase):
    def test_band_excludes_timeout_and_cross_episode(self):
        risk = helpers.failure_band(sample_arrays(), 3)
        np.testing.assert_equal(np.flatnonzero(risk), [8, 9, 10, 11, 32, 33, 34, 35])

    def test_recursive_samples_are_positions_in_valid_starts(self):
        arrays = sample_arrays()
        sampler = helpers.ReplaySampler(arrays, 5, 'balanced', np.random.default_rng(4).bit_generator.state, distance=3)
        sampler.integers(36, size=32)
        positions = sampler.integers(len(sampler.valid), size=32)
        endpoints = sampler.valid[positions] + 4
        self.assertEqual(int(helpers.failure_band(arrays, 3)[endpoints].sum()), 16)
        self.assertFalse(np.isin(sampler.valid, [8, 9, 10, 11, 20, 21, 22, 23]).any())

    def test_uniform_delegation_is_exact(self):
        rng = np.random.default_rng(123)
        sampler = helpers.ReplaySampler(sample_arrays(), 5, 'uniform', rng.bit_generator.state)
        for _ in range(3):
            for high in sampler.highs:
                np.testing.assert_equal(sampler.integers(high, size=32), rng.integers(high, size=32))

    def test_balanced_counts_and_replacement(self):
        sampler = helpers.ReplaySampler(sample_arrays(), 5, 'balanced', np.random.default_rng(1).bit_generator.state, distance=0)
        values = sampler.integers(36, size=32)
        self.assertEqual(int(sampler.risk[0][values].sum()), 16)
        self.assertEqual(sampler.counts[0]['draws'], 32)
        self.assertLess(len(np.unique(values[sampler.risk[0][values]])), 16)

    def test_empty_stratum_fallback_is_explicit(self):
        arrays = sample_arrays(); arrays['terminated'][:] = False
        sampler = helpers.ReplaySampler(arrays, 5, 'balanced', np.random.default_rng(1).bit_generator.state)
        sampler.integers(36, size=32)
        self.assertEqual(sampler.counts[0]['fallback_calls'], 1)

    def test_mid_phase_resume_is_exact(self):
        arrays = sample_arrays(); initial = np.random.default_rng(3).bit_generator.state
        sampler = helpers.ReplaySampler(arrays, 5, 'balanced', initial)
        sampler.integers(36, size=32)
        saved = sampler.state_dict()
        expected = sampler.integers(sampler.highs[1], size=32)
        resumed = helpers.ReplaySampler(arrays, 5, 'balanced', initial)
        resumed.load_state_dict(saved)
        np.testing.assert_equal(expected, resumed.integers(resumed.highs[1], size=32))
        self.assertEqual(resumed.counts, sampler.counts)

    def test_unexpected_call_is_rejected(self):
        sampler = helpers.ReplaySampler(sample_arrays(), 5, 'balanced', np.random.default_rng(1).bit_generator.state)
        with self.assertRaises(AssertionError):
            sampler.integers(1, size=32)

    def test_resume_discards_progress_after_durable_checkpoint(self):
        self.assertEqual(helpers.restored_history([{'round': 400}, {'round': 500},
            {'round': 600}, {'round': 700}], 500), [{'round': 400}, {'round': 500}])


if __name__ == '__main__':
    unittest.main()
