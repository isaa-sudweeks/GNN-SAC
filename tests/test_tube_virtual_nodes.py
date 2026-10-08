"""Tube grouping, equivariance, replay, and teacher compatibility regressions."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'sac'):
    sys.path.insert(0, str(path))
from common.graph_transforms import prepare_graph, graph_feature_flags, tube_components
from common.gnn_actor_critic import GNNActorCritic
from common.tensor_gnn_buffer import build_graph_static, dense_graph_batch, DenseGraphGroup, TensorGNNBuffer
from env.mujoco_gen.topology_envs import MujocoPresetGraphEnv
from gnn_sac import GNNSAC
from tests.test_gnn_mujoco_truss_gen_smoke import graph_test_cfg
from tests.test_virtual_node import cfg as model_cfg
from tests.test_tensor_gnn_buffer import config as replay_config, transition

FLAGS = dict(use_virtual_node=True, use_node_roles=True, use_edge_roles=True,
             use_edge_distance=True, use_edge_direction=True, use_tube_nodes=True)


def fixture():
    # Two open three-instance tubes joined at duplicated logical vertex instances.
    edges = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5, 2, 3],
                          [1, 0, 2, 1, 4, 3, 5, 4, 3, 2]])
    return Data(x=torch.randn(6, 3), edge_index=edges,
                edge_role=torch.tensor([0] * 8 + [1, 1]),
                edge_direction=torch.tensor([1., -1.] * 4 + [0., 0.]),
                action_mask=torch.tensor([1, 1, 0, 1, 1, 0], dtype=torch.bool),
                rigidity=torch.tensor([.7]))


class TubeVirtualNodeTest(unittest.TestCase):
    def test_membership_masks_features_and_global_isolation(self):
        raw = fixture()
        graph = prepare_graph(raw, **FLAGS)
        self.assertEqual(tube_components(raw), [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(graph.num_nodes, 9)
        self.assertEqual(graph.edge_index.size(1), raw.edge_index.size(1) + 12 + 12)
        self.assertEqual(graph.global_node_mask.nonzero().flatten().tolist(), [6])
        self.assertEqual(int(graph.physical_node_mask.sum()), 6)
        self.assertEqual(int(graph.action_mask.sum()), 4)
        self.assertFalse(graph.action_mask[6:].any())
        self.assertTrue(graph.edge_attr[-12:, 3].eq(1).all())
        self.assertTrue(graph.edge_attr[-12:, 4:].eq(0).all())
        self.assertTrue(graph.x[7:, -3].eq(1).all())
        self.assertTrue(graph.x[7:, -2:].eq(0).all())
        self.assertFalse(((graph.edge_index == 6).any(0) & (graph.edge_index >= 7).any(0)).any())
        self.assertNotIn('global_node_mask', raw)

    def test_relabeling_equivariance_and_mixed_batch_critic_readouts(self):
        torch.manual_seed(8)
        raw = fixture()
        order = torch.tensor([3, 5, 4, 1, 0, 2])
        inverse = order.argsort()
        permuted = raw.clone()
        permuted.x = raw.x[order]
        permuted.action_mask = raw.action_mask[order]
        permuted.edge_index = inverse[raw.edge_index]
        left, right = prepare_graph(raw, **FLAGS), prepare_graph(permuted, **FLAGS)
        for readout in ('physical_mean', 'virtual_node', 'physical_mean_virtual_node'):
            cfg = model_cfg()
            cfg.critic_readout = readout
            cfg.graph_features = dict(tube_nodes=True, node_roles=True, edge_roles=True,
                                      edge_distance=True, edge_direction=True)
            model = GNNActorCritic(cfg).eval()
            with torch.no_grad():
                # Compare full node embeddings after reversing physical relabeling.
                a = model._pi(left.x, left.edge_index, left.edge_attr)
                b = model._pi(right.x, right.edge_index, right.edge_attr)
                torch.testing.assert_close(a[:6], b[:6][inverse], rtol=1e-5, atol=1e-6)
                batch = Batch.from_data_list([left, right])
                q = model.Q(batch, torch.zeros(12, 1), return_type='all')
                self.assertEqual(tuple(q.shape), (2, 2))
                torch.testing.assert_close(q[:, 0], q[:, 1])

    def test_dense_rebuilding_keeps_rigidity_on_global_node(self):
        cfg = replay_config(multitask=False, task='graph', use_virtual_node=True,
                            graph_features=dict(tube_nodes=True, node_roles=True, edge_roles=True,
                                                edge_distance=True, edge_direction=True))
        graphs = [fixture(), fixture()]
        graphs[1].rigidity.fill_(.2)
        static = build_graph_static(cfg, graphs[0], torch.zeros(4, 1))
        group = DenseGraphGroup(static, torch.stack([g.x for g in graphs]),
                               torch.stack([g.rigidity for g in graphs]))
        dense = dense_graph_batch(cfg, [group], 'cpu')
        expected = Batch.from_data_list([prepare_graph(g, **FLAGS) for g in graphs])
        for field in ('x', 'edge_index', 'edge_attr', 'action_mask', 'physical_node_mask', 'global_node_mask'):
            torch.testing.assert_close(dense[field], expected[field])
        torch.testing.assert_close(dense.x[dense.global_node_mask, -1], torch.tensor([.7, .2]))

    def test_exact_source_membership_for_development_presets(self):
        from hydra import compose, initialize_config_dir
        from mujoco_truss_gen.mujoco_model import presets, builders
        with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
            cfg = compose(config_name='config', overrides=['cross_validation=random_5fold'])
        topologies = [t for group in cfg.cross_validation.groups.values() for t in group]
        for topology in topologies:
            _, structures = presets.get_preset_definition(topology)
            shapes = isinstance(next(iter(structures.values())), dict)
            metadata = (builders._shape_control_graph_metadata(structures, realistic=False)
                        if shapes else builders._triangle_control_graph_metadata(structures, realistic=False))
            names = metadata.control_node_names
            pairs = [(names.index(e.from_node), names.index(e.to_node)) for e in metadata.edges]
            raw = Data(x=torch.zeros(len(names), 3), edge_index=torch.tensor(pairs).t(),
                       edge_role=torch.tensor([int(e.type == 'connector') for e in metadata.edges]))
            actual = [Counter(metadata.control_node_to_logical_node[names[i]] for i in group)
                      for group in tube_components(raw)]
            expected = [Counter(shape['route'] if shapes else shape[:3])
                        for shape in structures.values()]
            self.assertCountEqual(actual, expected, topology)
        # Native construction also checks the compiled control-node observation order.
        for topology, sizes in [('tetrahedron', [4, 4]), ('octahedron', [3]*4),
                                ('usevitch_212365307', [3]*6), ('henneberg_n8_2tube_127', [10, 10])]:
            cfg = graph_test_cfg(domain_randomization=False, truss_topology=topology,
                                 graph_features=dict(tube_nodes=True, edge_roles=True, edge_direction=True))
            env = MujocoPresetGraphEnv(cfg)
            try:
                obs, _ = env.reset(seed=0)
                raw = Data(**{key: torch.as_tensor(value) for key, value in obs.items()})
                self.assertEqual(sorted(map(len, tube_components(raw))), sizes)
                graph = prepare_graph(raw, use_virtual_node=True, **graph_feature_flags(cfg))
                self.assertEqual(int(graph.global_node_mask.sum()), 1)
            finally:
                env.close()

    def test_missing_roles_or_global_node_fail(self):
        raw = fixture()
        del raw.edge_role
        with self.assertRaisesRegex(ValueError, 'edge_role'):
            prepare_graph(raw, **FLAGS)
        with self.assertRaisesRegex(ValueError, 'requires|require'):
            prepare_graph(fixture(), **dict(FLAGS, use_virtual_node=False))

    def _run_cpu_distillation_smoke(self, offline_only=False):
        from common.gnn_buffer import GNNBuffer
        from common.logger import Logger
        from env import make_env
        from trainer.base import Trainer
        from trainer.online_trainer import OnlineTrainer
        from tests.test_checkpointing import DummyLogger
        with tempfile.TemporaryDirectory() as tmp:
            settings = dict(work_dir=tmp, seed=1, steps=8, seed_steps=1, pretrain_steps=1,
                            batch_size=2, buffer_size=16, eval_freq=100, eval_episodes=1,
                            max_steps=2, nsubsteps=1, normalize_rewards=False,
                            save_video=False, enable_wandb=False, save_agent=False,
                            save_csv=False, checkpoint_freq=0, replay_ratio=1, episodic=True,
                            mpl_dims=[8], message_hidden_dims=[8], head_hidden_dims=[8],
                            log_std_min=-2., log_std_max=0., pcgrad=True)
            mapping = {}
            for topology in ("tetrahedron", "octahedron"):
                cfg = graph_test_cfg(**settings, truss_topology=topology)
                env = make_env(cfg)
                try:
                    agent, buffer = GNNSAC(cfg), GNNBuffer(cfg)
                    obs = env.reset()
                    episode = [dict(obs=obs, action=torch.zeros(obs.num_nodes, 1),
                                    reward=torch.tensor(0.), terminated=torch.tensor(False))]
                    for _ in range(2):
                        action = agent.act(obs)
                        obs, reward, _, info = env.step(action)
                        episode.append(dict(obs=obs, action=action, reward=reward, terminated=info['terminated']))
                    buffer.add(episode)
                    trainer = Trainer(cfg=cfg, env=env, agent=agent, buffer=buffer, logger=DummyLogger())
                    path = Path(tmp) / f'{topology}.pt'
                    torch.save(trainer.checkpoint_state_dict(), path)
                    mapping[topology] = str(path)
                finally:
                    env.close()
            cfg = graph_test_cfg(**settings, truss_topologies=['tetrahedron', 'octahedron'],
                                 graph_features=dict(tube_nodes=True, node_roles=True, edge_roles=True, edge_direction=True, edge_distance=True),
                                 distillation=dict(enabled=True, offline_only=offline_only, teachers=mapping, pretrain_updates=2,
                                                   batch_size=2, shard_size=2, checkpoint_freq=0, log_freq=1,
                                                   reconstruct_control_metadata=True))
            env = make_env(cfg)
            try:
                agent = GNNSAC(cfg)
                trainer = OnlineTrainer(cfg=cfg, env=env, agent=agent,
                                        buffer=TensorGNNBuffer(cfg), logger=Logger(cfg))
                trainer.train()
                self.assertEqual(trainer.distillation.completed_updates, 2)
                self.assertEqual(trainer._step, 0 if offline_only else cfg.steps)
                if offline_only:
                    self.assertEqual(trainer._optimizer_updates, 0)
                    self.assertTrue((Path(cfg.work_dir) / 'checkpoints' / 'distillation.pt').exists())
                else:
                    self.assertGreater(trainer._optimizer_updates, 0)
                self.assertTrue(all(torch.isfinite(x).all() for x in agent.model.parameters()))
                # Exercise diagnostic I/O using fixture checkpoints, not research data.
                from scripts.evaluate_routing_ablation import evaluate_checkpoint
                state = trainer.checkpoint_state_dict()
                state['config']['truss_topologies'] = ['octahedron']
                state['config']['eval_extra_topologies'] = ['tetrahedron']
                path = Path(tmp) / 'diagnostic_fixture.pt'
                torch.save(state, path)
                diagnostics = evaluate_checkpoint(path, samples=2, episodes=1)
                metrics = diagnostics['topologies']['tetrahedron']
                self.assertTrue(np.isfinite(metrics['teacher_kl']))
                self.assertTrue(np.isfinite(metrics['episode_distance']))
                self.assertEqual(len(metrics['rollouts']), 1)
            finally:
                env.close()


    def test_tube_student_two_topology_cpu_distillation_smoke(self):
        self._run_cpu_distillation_smoke()

    def test_offline_screen_saves_checkpoint_without_sac_transitions(self):
        self._run_cpu_distillation_smoke(offline_only=True)

    def test_launcher_preserves_protocol_and_excludes_final_tests(self):
        from scripts.launch_tube_ablation import build_commands
        for stage, steps in [('offline', 0), ('online', 2000010)]:
            command, jobs = build_commands(Path('/tmp/tube-dry'), Path('/tmp/tube-cache'), [1, 2, 3], stage,
                                           exp_name='custom-tube')
            self.assertEqual(len(jobs), 6)
            self.assertEqual({job['exp_name'] for job in jobs},
                             {'custom-tube-signed', 'custom-tube-membership'})
            self.assertEqual({job['steps'] for job in jobs}, {steps})
            self.assertEqual(len({tuple(job['training_topologies']) for job in jobs}), 1)
            self.assertEqual(len({tuple(job['heldout_topologies']) for job in jobs}), 1)
            for job in jobs:
                self.assertEqual(len(job['training_topologies']), 15)
                self.assertEqual(len(job['heldout_topologies']), 4)
                self.assertFalse(any('_n7_' in t or t == 'usevitch_1514879'
                                     for t in job['training_topologies'] + job['heldout_topologies']))
                self.assertTrue(job['graph_features']['edge_direction'])
                self.assertTrue(job['enable_wandb'])
                self.assertTrue(job['set_wandb_offline'])
                self.assertEqual(job['wandb_dir'], '/tmp/tube-dry')
            self.assertIn('distillation.decay_steps=75000000', command)
            self.assertIn('wandb_dir=/tmp/tube-dry', command)
            self.assertIn('enable_wandb=true', command)
            self.assertIn('set_wandb_offline=true', command)

    def test_tensor_replay_resume_preserves_tube_template(self):
        from common.distillation import replay_observations
        cfg = replay_config(multitask=False, task='graph', use_virtual_node=True,
                            graph_features=dict(tube_nodes=True, node_roles=True, edge_roles=True,
                                                edge_distance=True, edge_direction=True))
        raw = fixture()
        steps = transition(1, 6)
        for step in steps:
            step['obs'] = raw.clone()
        replay = TensorGNNBuffer(cfg)
        replay.add(steps)
        restored = TensorGNNBuffer(cfg)
        restored.load_state_dict(replay.state_dict())
        saved = next(iter(restored.state_dict()['buffers'].values()))
        for graph in replay_observations(saved):
            expected = prepare_graph(raw, **FLAGS)
            actual = prepare_graph(graph, **FLAGS)
            for field in ('x', 'edge_index', 'edge_attr', 'global_node_mask'):
                torch.testing.assert_close(actual[field], expected[field])

    def test_native_mjx_reset_feature_and_inference_parity(self):
        from env import make_env
        from tests.test_mjx_vector_env import mjx_cfg
        native_cfg = graph_test_cfg(domain_randomization=False,
                                    graph_features=dict(tube_nodes=True, node_roles=True, edge_roles=True,
                                                        edge_distance=True, edge_direction=True))
        vector_cfg = mjx_cfg(graph_features=native_cfg.graph_features)
        native, vector = make_env(native_cfg), make_env(vector_cfg)
        try:
            native_raw, vector_raw = native.reset(), vector.reset_many()[0]
            # Backends sample reset noise differently. Compare their structure and
            # prepare/infer the identical physical state to isolate adapter parity.
            for field in ('edge_index', 'edge_role', 'edge_direction', 'action_mask'):
                torch.testing.assert_close(native_raw[field], vector_raw[field])
            self.assertTrue(torch.isfinite(vector_raw.x).all())
            vector_raw.x = native_raw.x.clone()
            vector_raw.rigidity = native_raw.rigidity.clone()
            left = prepare_graph(native_raw, use_virtual_node=True, **graph_feature_flags(native_cfg))
            right = prepare_graph(vector_raw, use_virtual_node=True, **graph_feature_flags(vector_cfg))
            for field in ('x', 'edge_index', 'edge_attr', 'action_mask', 'physical_node_mask', 'global_node_mask'):
                torch.testing.assert_close(left[field], right[field], rtol=1e-4, atol=1e-5)
            model = GNNActorCritic(native_cfg).eval()
            with torch.no_grad():
                a, b = model.policy_distribution(left), model.policy_distribution(right)
                for first, second in zip(a, b):
                    torch.testing.assert_close(first, second, rtol=1e-4, atol=1e-5)
        finally:
            native.close()
            vector.close()
