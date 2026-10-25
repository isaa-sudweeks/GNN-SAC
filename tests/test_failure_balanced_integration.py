"""Native-update parity, full-round restart, and nominal simulator smoke tests."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('failure_runner', ROOT / 'scripts/run_failure_balanced_replay.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def equal(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


@unittest.skipUnless(os.environ.get('GNN_DYNAMICS_RUNTIME') and os.environ.get('GNN_DYNAMICS_CAMPAIGN'),
                     'Requires trusted original dynamics runtime and checkpoint artifacts')
class FailureBalancedIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = runner.bootstrap(Path(os.environ['GNN_DYNAMICS_RUNTIME']))
        cls.campaign = Path(os.environ['GNN_DYNAMICS_CAMPAIGN'])
        cls.options = dict(batch_size=32, failure_distance=10, failure_fraction=.5)

    def load(self, arm):
        return runner.load_group(self.api, self.campaign, 0, arm, self.options)

    def test_uniform_native_full_rounds_are_bitwise_identical(self):
        _, wrapped, replay_a, _ = self.load('uniform')
        _, native, replay_b, _ = self.load('uniform')
        for learner in native.values():
            learner.rng = learner.rng.rng
        for _ in range(3):
            self.api['update_round'](wrapped, replay_a)
            self.api['update_round'](native, replay_b)
        for topology in wrapped:
            for name in runner.MODULES:
                self.assertTrue(equal(getattr(wrapped[topology], name).state_dict(),
                                      getattr(native[topology], name).state_dict()), (topology, name))

    def test_balanced_full_round_checkpoint_resume_is_exact(self):
        _, learners, replays, samplers = self.load('balanced')
        for _ in range(2):
            self.api['update_round'](learners, replays)
        identity = dict(rounds=3, source='trusted-test-input')
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            runner.save_fit(self.api, folder, 2, learners, samplers, identity, 0.)
            self.api['update_round'](learners, replays)
            expected = {name: {key: deepcopy(getattr(l, key).state_dict()) for key in runner.MODULES}
                        for name, l in learners.items()}
            expected_sampler = {name: s.state_dict() for name, s in samplers.items()}
            del learners, replays, samplers
            _, resumed, replay, sampler = self.load('balanced')
            round_number, _ = runner.restore_fit(self.api, folder, resumed, sampler, identity)
            self.assertEqual(round_number, 2)
            self.api['update_round'](resumed, replay)
            for name, learner in resumed.items():
                for key in runner.MODULES:
                    self.assertTrue(equal(getattr(learner, key).state_dict(), expected[name][key]), (name, key))
                self.assertTrue(equal(sampler[name].state_dict(), expected_sampler[name]))
            first = next(iter(resumed.values()))
            self.assertEqual({int(s['step']) for s in first.optimizer.state.values()}, {60009})

    def test_literal_paired_reset_boundary_and_validation_branch_smoke(self):
        cfg, learners, _, _ = self.load('uniform')
        learner = next(iter(learners.values()))
        cfg = dict(cfg, episode_steps=2)
        with tempfile.TemporaryDirectory() as tmp:
            first = runner.collect(self.api, learner, cfg, [23000000], Path(tmp) / 'first', True)
            second = runner.collect(self.api, learner, cfg, [23000000], Path(tmp) / 'second', False)
            self.assertEqual(first[0]['seed'], 23000000)
            self.assertEqual(first[0]['initial_state_sha256'], second[0]['initial_state_sha256'])
            self.assertTrue(first[0]['restoration']['exact_continuation'])
            with np.load(Path(tmp) / 'first' / first[0]['trace']) as trace:
                self.assertTrue(trace['truncated'][-1])
                self.assertEqual(len(trace['state']), 2)
            result = runner.validate(self.api, learner, Path(tmp) / 'first')
            self.assertEqual(result['rankings'][0]['pooled']['candidates'], 41)


if __name__ == '__main__':
    unittest.main()
