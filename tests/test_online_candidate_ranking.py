"""Decision metrics and restoration contracts for frozen candidate diagnostics."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

PATH = Path(__file__).resolve().parents[1] / 'scripts/run_online_candidate_ranking.py'
SPEC = importlib.util.spec_from_file_location('candidate_diagnostic', PATH)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class CandidateDiagnosticTests(unittest.TestCase):
    def test_short_episode_deduplicates_strata(self):
        self.assertEqual(diagnostic.selected_states(1, True),
                         [dict(timestep=0, strata=['initial', 'precollapse'])])
        self.assertEqual([r['timestep'] for r in diagnostic.selected_states(80, False)], [0, 20, 70])
        self.assertEqual([r['timestep'] for r in diagnostic.selected_states(25, True)], [0, 20])

    def test_known_spearman_and_ties(self):
        np.testing.assert_equal(diagnostic.average_ranks(np.array([3., 1., 1.])), [2., .5, .5])
        result = diagnostic.ranking_metrics(np.array([3., 2., 1.]), np.zeros(3, bool),
            np.array([1., 2., 3.]), np.zeros(3, bool), np.ones(3))
        self.assertAlmostEqual(result['spearman'], -1)
        self.assertEqual(result['regret'], 2.)

    def test_safe_alternative_and_false_negative(self):
        result = diagnostic.ranking_metrics(np.array([1., 0.]), np.array([False, False]),
            np.array([-1., .2]), np.array([True, False]), np.array([.1, .2]))
        self.assertTrue(result['avoidable_selected_collapse'])
        self.assertEqual(result['false_negatives'], 1)
        self.assertEqual(result['positive_progress_safe_candidates'], 1)

    def test_all_failures_are_not_called_avoidable(self):
        result = diagnostic.ranking_metrics(np.ones(2), np.ones(2, bool),
            np.zeros(2), np.ones(2, bool), np.zeros(2))
        self.assertFalse(result['avoidable_selected_collapse'])
        self.assertFalse(result['safe_opportunity'])
        self.assertIsNone(result['spearman'])

    def test_nonfinite_results_are_not_dropped(self):
        with self.assertRaises(ValueError):
            diagnostic.ranking_metrics(np.array([np.nan]), np.array([False]),
                np.array([0.]), np.array([False]), np.array([0.]))

    def test_exact_cem_tie_selection(self):
        learner = SimpleNamespace(metadata={'action_mask': [True]}, cem=None,
                                  score=lambda state, candidates: np.ones(len(candidates)))
        def fake_cem(score, *args, **kwargs):
            candidates = np.array([[[0.]], [[1.]]])
            score(candidates)
            return SimpleNamespace(actions=candidates[1])
        candidates, chosen = diagnostic.propose(learner, None, 1, None, fake_cem)
        self.assertEqual(chosen, 1)
        self.assertEqual(len(candidates), 2)

    def test_branch_discount_and_true_terminal_stop(self):
        branch = SimpleNamespace(planning_env=None, restore=lambda snapshot: None)
        count = []
        def step(env, metadata, mapping, action, cfg):
            count.append(1)
            row = dict(state=np.zeros((1, 6)), next_state=np.zeros((1, 6)),
                progress=np.array([.2, 0.]), reward=-.8 if len(count) == 2 else .2,
                terminated=len(count) == 2, truncated=False)
            return row, {'dt': .2}, len(count) == 2
        result = diagnostic.branch_rollout(branch, None, np.zeros((10, 1)),
            {'node_to_physical': [0], 'initial_diagonal': 2}, {'gamma': .5}, step)
        self.assertEqual(len(count), 2)
        self.assertAlmostEqual(result['score'], -.2)
        self.assertTrue(result['collapse'])
        self.assertAlmostEqual(result['displacement'], .2)

    def test_branch_rejects_artificial_truncation(self):
        branch = SimpleNamespace(planning_env=None, restore=lambda snapshot: None)
        def step(*args):
            return dict(state=np.zeros((1, 6)), next_state=np.zeros((1, 6)),
                progress=np.zeros(2), reward=0., terminated=False, truncated=True), {'dt': .2}, True
        with self.assertRaises(AssertionError):
            diagnostic.branch_rollout(branch, None, np.zeros((1, 1)),
                {'node_to_physical': [0], 'initial_diagonal': 2}, {'gamma': .5}, step)


if __name__ == '__main__':
    unittest.main()
