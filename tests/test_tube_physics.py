"""Physical lengths, cache/replay contracts and the three-arm experiment."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'sac'):
    sys.path.insert(0, str(path))

from common.graph_transforms import prepare_graph, tube_physics_features, tube_components, graph_feature_flags, graph_feature_schema
from common.tensor_gnn_buffer import DenseGraphGroup, build_graph_static, dense_graph_batch, TensorGNNBuffer
from common.distillation import replay_observations, reconstruct_control_metadata, attach_control_metadata
from env.mujoco_gen.topology_envs import MujocoPresetGraphEnv
from tests.test_tube_virtual_nodes import FLAGS, fixture
from tests.test_tensor_gnn_buffer import config as replay_config, transition


def physics_graph():
    graph = fixture()
    graph.x = torch.tensor([[0.,0,0],[.5,0,0],[.5,1/3,0],
                            [0,0,0],[0,0,.5],[1,0,.5]])
    graph.tube_position_scale = torch.tensor([[2.,3,4]])
    graph.tube_reference_length = torch.tensor([2.,2,2,5,5,5])
    graph.tube_segment_weight = torch.tensor([.5] * 8 + [0.,0])
    return graph


class TubePhysicsTest(unittest.TestCase):
    def test_inverse_anisotropic_normalization_and_fixed_reference(self):
        graph = physics_graph()
        flags = dict(FLAGS, use_tube_physics=True)
        hubs = prepare_graph(graph, **flags).x[7:, -7:-3]
        torch.testing.assert_close(hubs, torch.tensor([[2.,2,1,0],[2,5,.8,-.2]]))
        changed = graph.clone(); changed.x *= 2
        other = prepare_graph(changed, **flags).x[7:, -7:-3]
        torch.testing.assert_close(other[:, 1], hubs[:, 1])
        torch.testing.assert_close(other[:, 2], 2 * hubs[:, 2])
        invalid = graph.clone(); del invalid.tube_position_scale
        with self.assertRaisesRegex(ValueError, 'metadata'):
            prepare_graph(invalid, **flags)
        with self.assertRaisesRegex(ValueError, 'requires'):
            prepare_graph(graph, **dict(flags, use_tube_nodes=False))

    def test_node_relabeling_preserves_physical_features(self):
        graph = physics_graph(); perm = torch.tensor([5,4,3,2,1,0])
        inverse = torch.argsort(perm)
        shuffled = graph.clone(); shuffled.x = graph.x[perm]
        shuffled.action_mask = graph.action_mask[perm]
        shuffled.tube_reference_length = graph.tube_reference_length[perm]
        shuffled.edge_index = inverse[graph.edge_index]
        flags = dict(FLAGS, use_tube_physics=True)
        torch.testing.assert_close(prepare_graph(shuffled, **flags).x[7:, -7:-3],
                                   prepare_graph(graph, **flags).x[7:, -7:-3].flip(0))

    def test_dense_dynamic_features_and_replay_roundtrip(self):
        graph = physics_graph()
        cfg = replay_config(multitask=False, task='graph', obs_dim=3, use_virtual_node=True,
                            graph_features=dict(tube_nodes=True, tube_physics=True, node_roles=True,
                                                edge_roles=True, edge_direction=True, edge_distance=True))
        graphs = [graph, graph.clone()]; graphs[1].x *= 1.5
        static = build_graph_static(cfg, graph, torch.zeros(6,1))
        group = DenseGraphGroup(static, torch.stack([g.x for g in graphs]), torch.stack([g.rigidity for g in graphs]))
        dense = dense_graph_batch(cfg, [group], 'cpu')
        expected = Batch.from_data_list([prepare_graph(g, use_virtual_node=True, **graph_feature_flags(cfg)) for g in graphs])
        for field in ['x','edge_index','edge_attr','action_mask','global_node_mask']:
            torch.testing.assert_close(dense[field], expected[field])
        buffer = TensorGNNBuffer(cfg)
        episode = transition(1, 6)
        for step in episode:step['obs'] = graph.clone()
        buffer.add(episode)
        restored = TensorGNNBuffer(cfg); restored.load_state_dict(buffer.state_dict())
        replay = next(iter(restored.state_dict()['buffers'].values()))
        for raw in replay_observations(replay):
            torch.testing.assert_close(prepare_graph(raw, use_virtual_node=True, **graph_feature_flags(cfg)).x,
                                       prepare_graph(graph, use_virtual_node=True, **graph_feature_flags(cfg)).x)

    def test_compiled_route_lengths_all_development_presets(self):
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
            groups = compose(config_name='config', overrides=['cross_validation=random_5fold']).cross_validation.groups
        for topology in [t for group in groups.values() for t in group]:
            with self.subTest(topology=topology):
                cfg = dict(truss_topology=topology, use_control_graph=True, truss_realistic=False,
                           normalize_observations=True, domain_randomization=False, nsubsteps=1,
                           graph_features=dict(tube_nodes=True, tube_physics=True, edge_roles=True))
                env = MujocoPresetGraphEnv(cfg)
                try:
                    for seed in [0,1]:
                        observation, _ = env.reset(seed=seed)
                        graph = Data(**{k:torch.as_tensor(v) for k,v in observation.items()})
                        values = tube_physics_features(graph.x, graph.edge_index, tube_components(graph), graph)
                        model, data = env.mj_model.model, env.mj_model.data
                        ids = [int(model.eq_obj1id[i]) for i in range(model.neq)]
                        ids += [int(model.eq_obj2id[i]) for i in range(model.neq) if model.eq_obj2id[i] >= 0]
                        np.testing.assert_allclose(float((values[:,1] * values[:,2]).sum()),
                                                   data.ten_length[ids].sum(), rtol=2e-5, atol=2e-5)
                        np.testing.assert_allclose(float(values[:,1].sum()), model.tendon_length0[ids].sum(), rtol=2e-6)
                        # Reconstruct metadata for a legacy replay without changing its x.
                        legacy = graph.clone()
                        for key in ['tube_position_scale','tube_reference_length','tube_segment_weight','edge_role']:
                            del legacy[key]
                        metadata = reconstruct_control_metadata(cfg, cfg, legacy)
                        recovered = attach_control_metadata(legacy, metadata)
                        torch.testing.assert_close(recovered.x, graph.x)
                        torch.testing.assert_close(tube_physics_features(recovered.x, recovered.edge_index, tube_components(recovered), recovered), values)
                finally:env.close()

    def test_three_arm_launcher_and_schema(self):
        from scripts.launch_tube_ablation import build_commands
        command, jobs = build_commands(Path('/tmp/tube-physics'), Path('/tmp/tube-cache'), [1,2,3],
                                       exp_name='tube-physics-v1', arms=('signed','membership','physics'))
        self.assertEqual(len(jobs), 9)
        self.assertIn('+tube_ablation=signed,membership,physics', command)
        self.assertIn('distillation.eval_freq=2000', command)
        self.assertEqual({j['steps'] for j in jobs}, {0})
        for job in jobs:
            features = job['graph_features']
            self.assertEqual(bool(features['tube_physics']), job['arm'] == 'physics')
            self.assertTrue(features['edge_direction'] and features['edge_distance'])
        self.assertNotEqual(graph_feature_schema(dict(graph_features=dict(tube_nodes=True))),
                            graph_feature_schema(dict(graph_features=dict(tube_nodes=True, tube_physics=True))))
        mapping = next(value for value in command if value.startswith('distillation.teachers='))
        self.assertIn('clean-replay-v1', mapping)
        self.assertIn('domain_randomization_params.observation_noise.enabled=false', command)
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
            cfg = compose(config_name='config', overrides=command[3:-2], return_hydra_config=True)
            self.assertEqual(len(cfg.distillation.teachers), 19)
            self.assertTrue(str(cfg.hydra.sweep.dir).startswith('/tmp/tube-physics/hydra/'))

    def test_clean_teacher_collection_and_cache_provenance(self):
        from env import make_env
        from gnn_sac import GNNSAC
        from scripts.prepare_tube_replay import prepare_replay
        from tests.test_gnn_mujoco_truss_gen_smoke import graph_test_cfg
        with tempfile.TemporaryDirectory() as tmp:
            cfg = graph_test_cfg(device='cpu', max_steps=2, nsubsteps=1,
                                 mpl_dims=[8], message_hidden_dims=[8], head_hidden_dims=[8],
                                 domain_randomization=True,
                                 domain_randomization_params=dict(observation_noise=dict(enabled=True, std=.02),
                                                                 length_scale=dict(enabled=False)))
            env = make_env(cfg)
            try:
                source, output = Path(tmp) / 'teacher.pt', Path(tmp) / 'clean.pt'
                agent = GNNSAC(cfg)
                torch.save(dict(config=vars(cfg), agent=dict(model=agent.model.state_dict(),
                           graph_feature_schema=graph_feature_schema(cfg))), source)
            finally:
                env.close()
            prepare_replay(source, output, 4, 17)
            state = torch.load(output, weights_only=False)
            raw = next(replay_observations(next(iter(state['buffer']['buffers'].values()))))
            student = dict(graph_features=dict(tube_nodes=True, tube_physics=True, edge_roles=True))
            with self.assertRaisesRegex(ValueError, 'clean'):
                reconstruct_control_metadata(cfg, student, raw)
            metadata = reconstruct_control_metadata(state['config'], student, raw)
            enriched = attach_control_metadata(raw, metadata)
            self.assertTrue(torch.isfinite(tube_physics_features(raw.x, raw.edge_index, tube_components(enriched), enriched)).all())
            timestamp = output.stat().st_mtime_ns
            prepare_replay(source, output, 4, 17)
            self.assertEqual(output.stat().st_mtime_ns, timestamp)
            with self.assertRaisesRegex(ValueError, 'provenance'):
                prepare_replay(source, output, 5, 17)
            second = Path(tmp) / 'clean_again.pt'
            prepare_replay(source, second, 4, 17)
            replay = torch.load(second, weights_only=False)['buffer']['buffers']
            other = next(replay_observations(next(iter(replay.values()))))
            torch.testing.assert_close(raw.x, other.x)

    def test_offline_diagnostic_snapshots_are_chronological(self):
        from scripts.evaluate_routing_ablation import select_checkpoints
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('distillation_10000.pt', 'distillation_2000.pt',
                         'distillation.pt', 'distillation_2000.agent.pt'):
                (Path(tmp) / name).touch()
            self.assertEqual([label for label, _ in select_checkpoints(Path(tmp))],
                             ['distillation_2000', 'distillation_10000', 'offline'])
