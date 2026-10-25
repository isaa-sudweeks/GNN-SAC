"""Regression checks for conservative replay-audit eligibility."""
import unittest
from pathlib import Path
import tempfile
import torch
from scripts.audit_dynamics_replay import audit_buffer, audit_checkpoint, classify_label_contract, valid_indices


class ReplayAuditTests(unittest.TestCase):
    def test_wrapped_ring_excludes_no_valid_rows(self):
        self.assertEqual(valid_indices(4, 4, 2), [2, 3, 0, 1])
        self.assertEqual(valid_indices(4, 2, 2), [0, 1])

    def test_inconsistent_partial_ring_rejected(self):
        with self.assertRaises(ValueError):
            valid_indices(4, 2, 3)

    def test_finite_rows_do_not_prove_trajectories(self):
        buffer = dict(capacity=3, size=2, idx=2)
        for field in ('obs', 'next_obs', 'action', 'reward', 'terminated'):
            buffer[field] = [torch.zeros(2, 3), torch.ones(2, 3), None]
        report = audit_buffer(buffer, seed=4, sample_count=10)
        self.assertEqual(report['sampled_indices'], [0, 1])
        self.assertFalse(report['trajectory_eligible'])
        self.assertEqual(report['one_step_status'], 'candidate_pending_state_and_physics_audit')
        buffer['next_obs'][1][0, 0] = float('nan')
        self.assertEqual(audit_buffer(buffer, seed=4, sample_count=10)['one_step_status'], 'insufficient')

    def test_noise_contract_does_not_reject_missing_targets(self):
        report = classify_label_contract({'sim_backend': 'mujoco'})
        self.assertFalse(report['fresh_nominal_fallback_justified'])
        self.assertFalse(report['hidden_targets_required'])
        self.assertFalse(report['nominal_native_eligible'])
        self.assertEqual(report['clean_label_status'], 'requires_collection_trace')

    def test_noisy_observation_settings_justify_clean_label_fallback(self):
        report = classify_label_contract({
            'sim_backend': 'mjx', 'domain_randomization': True,
            'domain_randomization_params': {
                'observation_noise': {'enabled': True, 'std': 0.02},
                'action_noise': {'enabled': True, 'std': 0.1}}})
        self.assertTrue(report['fresh_nominal_fallback_justified'])
        self.assertTrue(report['observation_noise_enabled'])
        self.assertTrue(report['action_noise_enabled'])
        report = classify_label_contract({
            'sim_backend': 'mujoco', 'domain_randomization': True,
            'domain_randomization_params': {'action_noise': {'enabled': True, 'std': 0.1}}})
        self.assertFalse(report['fresh_nominal_fallback_justified'])

    def test_generic_specialist_identity_comes_from_config(self):
        buffer = dict(capacity=2, size=1, idx=1)
        for field in ('obs', 'next_obs', 'action', 'reward', 'terminated'):
            buffer[field] = [torch.zeros(2, 3), None]
        checkpoint = dict(config={'truss_topology': 'tetrahedron', 'truss_topologies': None},
                          buffer={'buffers': {'truss-graph': buffer}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.pt'
            torch.save(checkpoint, path)
            report = audit_checkpoint(path, sample_count=1)
        self.assertEqual(report['tasks']['tetrahedron']['stored_task_key'], 'truss-graph')

    def test_holdout_rows_are_excluded_before_inspection(self):
        checkpoint = dict(config={'truss_topology': 'octahedron'},
                          buffer={'buffers': {'truss-graph': {'not_a_valid_buffer': True}}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'holdout.pt'
            torch.save(checkpoint, path)
            report = audit_checkpoint(path, sample_count=1)
        self.assertEqual(report['tasks'], {})
        self.assertEqual(report['excluded_without_row_inspection'], ['truss-graph'])


if __name__ == '__main__':
    unittest.main()
