"""Guard experiment budgets, development holdouts, and diagnostic sampling."""
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.launch_routing_ablation import build_commands
from scripts.evaluate_routing_ablation import replay_samples, select_checkpoints
from tests.test_tensor_gnn_buffer import config, TensorGNNBuffer, transition


class RoutingExperimentTest(unittest.TestCase):
    def test_nine_jobs_hold_everything_but_edge_semantics_fixed(self):
        command, jobs = build_commands(Path('/tmp/routing-dry'), Path('/tmp/routing-cache'), [1, 2, 3])
        self.assertEqual(len(jobs), 9)
        self.assertEqual([j['arm'] for j in jobs[:3]], ['baseline', 'edge_types', 'signed'])
        self.assertEqual({j['steps'] for j in jobs}, {2000010})
        self.assertEqual(len({tuple(j['training_topologies']) for j in jobs}), 1)
        self.assertEqual(len({tuple(j['heldout_topologies']) for j in jobs}), 1)
        for job in jobs:
            self.assertFalse(set(job['training_topologies']) & set(job['heldout_topologies']))
            self.assertEqual(len(job['training_topologies']), 15)
            self.assertEqual(len(job['heldout_topologies']), 4)
            self.assertTrue(job['graph_features']['node_roles'])
            self.assertTrue(job['graph_features']['edge_distance'])
            self.assertFalse(any('_n7_' in t for t in job['training_topologies']+job['heldout_topologies']))
        self.assertIn('distillation.decay_steps=75000000', command)

    def test_replay_sampling_is_reproducible_and_handles_wrapped_ring(self):
        cfg = config(task='graph', multitask=False, buffer_size=4)
        buffer = TensorGNNBuffer(cfg)
        for marker in range(8):
            buffer.add(transition(marker, 3, metadata=True))
        replay = next(iter(buffer.state_dict()['buffers'].values()))
        first, second = replay_samples(replay, 32, 17), replay_samples(replay, 32, 17)
        self.assertEqual(len(first), 32)
        for a, b in zip(first, second):
            torch.testing.assert_close(a.x, b.x)
            self.assertGreaterEqual(float(a.x[0, 0]), 4)
            self.assertLess(float(a.x[0, 0]), 8)

    def test_checkpoint_selection_uses_steps_not_scores_or_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('distillation.pt', 'step_800190.pt', 'step_1600380.pt',
                         'step_2000010.pt', 'step_800190.agent.pt', 'latest.pt'):
                (root/name).touch()
            self.assertEqual([(stage, path.name) for stage, path in select_checkpoints(root)], [
                ('offline', 'distillation.pt'), ('800k', 'step_800190.pt'), ('final', 'step_2000010.pt')])


if __name__ == '__main__':
    unittest.main()
