import sys
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sac'))
from dynamics.model import ObservableDynamicsGNN, physical_graph_loss
from dynamics.training import fitting_statistics, target_deltas, tensor_graph, fit_model, _cache, rollout_training_loss, validation_progress_loss, evaluate_recursive


def episode():
    state = np.zeros((3, 3, 6), dtype=np.float32)
    state[:, :, :3] = [[-.5, 0, .2], [.5, 0, .2], [-.5, 0, .2]]
    nxt = state.copy(); nxt[:, :, 2] += .01
    return dict(state=state, next_state=nxt, action=np.ones((3, 3, 1), np.float32),
                progress=np.tile([.1, .0], (3, 1)), dt=np.full(3, .2),
                edge_index=np.array([[0, 1, 2, 1], [1, 0, 1, 2]]), edge_static=np.zeros((4, 2)),
                action_mask=np.ones(3), physical_mask=np.array([1, 1, 0], bool),
                node_to_physical=np.array([0, 1, 0]), initial_diagonal=1.0,
                role='fitting', topology='tetrahedron')


class DynamicsModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_permutation_equivariance_and_alias_refresh(self):
        e = episode(); graph = tensor_graph(e)
        model = ObservableDynamicsGNN(width=16)
        state = torch.tensor(e['state'][:1]); action = torch.tensor(e['action'][:1])
        permutation = torch.tensor([2, 0, 1]); inverse = torch.argsort(permutation)
        permuted = dict(graph)
        for key in ('physical_mask', 'action_mask', 'multiplicity'):
            permuted[key] = graph[key][permutation]
        permuted['edge_index'] = inverse[graph['edge_index']]
        permuted['representative_index'] = inverse[graph['representative_index'][permutation]]
        expected = model(state, action, graph)[:, permutation]
        actual = model(state[:, permutation], action[:, permutation], permuted)
        torch.testing.assert_close(actual, expected)
        future, progress = model.step(state, action, graph)
        torch.testing.assert_close(future[:, 0], future[:, 2])
        torch.testing.assert_close(future[:, graph['physical_mask'], :2].mean(1), torch.zeros_like(progress), atol=1e-6, rtol=0)

    def test_loss_ignores_duplicate_nodes_and_balances_graphs(self):
        predicted = torch.zeros(2, 3, 6); target = torch.ones_like(predicted)
        target[0, :2] *= 2; target[:, 2] = 1000
        actual = physical_graph_loss(predicted, target, torch.tensor([1, 1, 0], dtype=torch.bool), torch.ones(6))
        self.assertEqual(float(actual), 2.5)

    def test_target_preserves_mean_translation(self):
        e = episode(); delta = target_deltas(e)
        np.testing.assert_allclose(delta[:, :, 0], .1)
        np.testing.assert_allclose(delta[:, :, 2], .01, atol=1e-7)

    def test_normalization_rejects_validation_and_ignores_alias_duplicates(self):
        e = episode(); stats = fitting_statistics([e])
        expected = e['state'][:, :2].reshape(-1, 6).mean(0)
        np.testing.assert_allclose(stats['mean'], expected)
        e['role'] = 'evaluation'
        with self.assertRaisesRegex(ValueError, 'fitting-role'):
            fitting_statistics([e])

    def test_recursive_step_refreshes_geometry_without_future_truth(self):
        e = episode(); graph = tensor_graph(e)
        model = ObservableDynamicsGNN(width=16)
        state = torch.tensor(e['state'][:1]); action = torch.tensor(e['action'][:1])
        first = model(state, action, graph)
        altered = state.clone(); altered[:, 1, 2] += 1
        second = model(altered, action, graph)
        self.assertGreater(float((first - second).abs().max().detach()), 1e-6)
        with patch.object(model, 'forward', wraps=model.forward) as tracked:
            predicted, _ = model.step(state, action, graph)
            model.step(predicted, action, graph)
            torch.testing.assert_close(tracked.call_args_list[1].args[0], predicted)

    def test_routing_features_use_known_commands_and_static_routes(self):
        e = episode(); e['routing_incidence'] = np.array([[1, -1, 0]], np.float32)
        e['ctrlrange'] = np.array([[-.05, .05]]); e['speed'] = .05
        graph = tensor_graph(e); model = ObservableDynamicsGNN(width=16)
        state = torch.tensor(e['state'][:1]); action = torch.tensor(e['action'][:1])
        before = model(state, action, graph)
        changed = dict(graph); changed['routing_incidence'] = -graph['routing_incidence']
        action[:, 1] = 0
        positive = model(state, action, graph); negative = model(state, action, changed)
        self.assertGreater(float((positive - negative).abs().max().detach()), 1e-7)
        self.assertTrue(torch.isfinite(before).all())

    def test_multistep_loss_backpropagates_and_rejects_episode_crossing(self):
        cached = _cache([episode()], "cpu")[0]
        model = ObservableDynamicsGNN(width=16)
        with patch.object(model, 'step', wraps=model.step) as tracked:
            loss = rollout_training_loss(model, cached, torch.tensor([0, 1]), 2)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(tracked.call_count, 2)
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.parameters()))
        with self.assertRaisesRegex(ValueError, 'boundary'):
            rollout_training_loss(model, cached, torch.tensor([2]), 2)
        self.assertTrue(np.isfinite(validation_progress_loss(model, [cached], 2)))

    def test_checkpoint_selection_rejects_overlapping_seed(self):
        fitting = episode(); fitting['seed'] = 12000
        validation = episode(); validation.update(role='validation', seed=12000)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'identities overlap'):
                fit_model([fitting], [validation], {}, Path(folder))

    def test_fixed_budget_overrides_patience_and_averages_all_source_batches(self):
        import contextlib
        import io
        import json
        fitting = [episode(), episode()]
        fitting[0].update(seed=12000, topology='tetrahedron')
        fitting[1].update(seed=12100, topology='henneberg_n6_1tube_2')
        validation = episode(); validation.update(role='validation', seed=22000)
        settings = dict(seed=2, width=8, layers=1, batch_size=2, updates=4,
                        learning_rate=.001, weight_decay=0., evaluate_every=2,
                        patience=1, minimum_updates=1, device='cpu', threads=1,
                        early_stopping=False)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            with patch('dynamics.training.validation_loss', return_value=1.):
                _, result = fit_model(fitting, [validation], settings, Path(folder))
            self.assertEqual(result['updates_completed'], 4)
            for row in result['history']:
                self.assertEqual(row['training_batches_averaged'], 2)
                self.assertEqual(set(row['train_loss_by_topology']), {e['topology'] for e in fitting})
                expected = np.mean([v['mean'] for v in row['train_loss_by_topology'].values()])
                self.assertAlmostEqual(row['train_loss_mean'], expected)
            final = torch.load(Path(folder) / 'last-model.pt', weights_only=False)
            self.assertEqual(final['update'], 4)
            protocol = json.loads((Path(folder) / 'fit-protocol.json').read_text())
            self.assertIn('disabled', protocol['stopping'])

    def test_nonfinite_recursive_windows_are_retained_as_failures(self):
        e = episode(); e.update(role='evaluation', terminated=np.zeros(3, bool))
        model = ObservableDynamicsGNN(width=16)
        def bad_step(state, action, graph):
            return torch.full_like(state, float('nan')), torch.full((len(state), 2), float('nan'))
        with patch.object(model, 'step', side_effect=bad_step):
            report = evaluate_recursive(model, [e], [1])
        gnn = next(r for r in report['summary'] if r['predictor'] == 'gnn')
        self.assertEqual(gnn['windows'], 3)
        self.assertEqual(gnn['nonfinite_windows'], 3)
        self.assertEqual(gnn['progress_mae'], 100.)
