"""End-to-end matched diagnostic using tiny CPU teachers and disjoint replay."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'sac', ROOT / 'scripts'):
    sys.path.insert(0, str(path))

from common.graph_transforms import graph_feature_schema
from env import make_env
from gnn_sac import GNNSAC
from prepare_tube_replay import prepare_replay
from run_teacher_student_diagnostic import prepare_validation, train_student, evaluate_student
from tests.test_gnn_mujoco_truss_gen_smoke import graph_test_cfg


class TeacherStudentDiagnosticTest(unittest.TestCase):
    def test_separate_shared_initialization_budget_and_fresh_evaluation(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            topologies = ['tetrahedron', 'octahedron']
            training, validation = {}, {}
            for topology in topologies:
                cfg = graph_test_cfg(truss_topology=topology, max_steps=2, nsubsteps=1,
                                     mpl_dims=[8], message_hidden_dims=[8], head_hidden_dims=[8])
                env = make_env(cfg)
                try:
                    agent = GNNSAC(cfg)
                    source = root / f'{topology}.pt'
                    torch.save(dict(config=vars(cfg), agent=dict(model=agent.model.state_dict(),
                               graph_feature_schema=graph_feature_schema(cfg))), source)
                finally:
                    env.close()
                training[topology] = str(root / f'{topology}-train.pt')
                validation[topology] = str(root / f'{topology}-validation.pt')
                prepare_replay(source, Path(training[topology]), 8, 17)
            jobs = [dict(label='separate', seed=1, topologies=topologies[:1], directory=str(root / 'separate')),
                    dict(label='shared', seed=1, topologies=topologies, directory=str(root / 'shared'))]
            manifest = dict(topologies=topologies, training_replay=training, validation_replay=validation,
                            validation_samples=8, validation_seed=29, run_root=str(root),
                            reset_seeds=[1000], target_cache=str(root / 'targets'),
                            evaluation_samples=2, batch_size=2, updates=2, snapshots=[2],
                            device='cpu', wandb=False, jobs=jobs,
                            overrides=['max_steps=2', 'nsubsteps=1', 'mpl_dims=[8]',
                                       'message_hidden_dims=[8]', 'head_hidden_dims=[8]'])
            for index in range(2):
                prepare_validation(manifest, index)
                train_student(manifest, index)
                evaluate_student(manifest, index)
            identities = [json.loads((Path(job['directory']) / 'identity.json').read_text()) for job in jobs]
            self.assertEqual(identities[0]['initial_actor_sha256'], identities[1]['initial_actor_sha256'])
            self.assertEqual(identities[0]['updates_per_topology'], identities[1]['updates_per_topology'])
            for job in jobs:
                results = json.loads((Path(job['directory']) / 'evaluation.json').read_text())['2']
                self.assertEqual(set(results['topologies']), set(job['topologies']))
                for topology, metrics in results['topologies'].items():
                    self.assertTrue(torch.isfinite(torch.tensor(metrics['fresh_teacher_kl'])))
                    self.assertEqual(len(metrics['rollouts']), 1)
                    self.assertAlmostEqual(results['actor_gradient_cosine'][topology][topology], 1, places=5)
                source = torch.load(training[job['topologies'][0]], weights_only=False)
                fresh = torch.load(validation[job['topologies'][0]], weights_only=False)
                self.assertNotEqual(source['tube_replay_provenance']['seed'], fresh['tube_replay_provenance']['seed'])
