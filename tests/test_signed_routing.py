"""Signed-routing semantics, replay preservation, and legacy teacher compatibility."""
from copy import copy, deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "sac"):
    sys.path.insert(0, str(path))

from common.distillation import (
    Distillation, attach_control_metadata, reconstruct_control_metadata, replay_observations,
)
from common.graph_transforms import graph_feature_flags, graph_feature_schema, prepare_graph
from common.tensor_gnn_buffer import DenseGraphGroup, build_graph_static, dense_graph_batch, TensorGNNBuffer
from env.mujoco_gen.topology_envs import MujocoPresetGraphEnv, control_edge_directions
from gnn_sac import GNNSAC
from tests.test_gnn_mujoco_truss_gen_smoke import graph_test_cfg
from tests.test_tensor_gnn_buffer import config as replay_config, transition


class SignedRoutingTest(unittest.TestCase):
    def test_real_controller_reversal_and_permutation(self):
        cfg = graph_test_cfg(domain_randomization=False, graph_features=dict(edge_direction=True, edge_roles=True))
        env = MujocoPresetGraphEnv(cfg)
        try:
            raw, _ = env.reset(seed=0)
            metadata = env.mj_model.control_graph
            signs = control_edge_directions(env.mj_model)
            index = raw['edge_index']
            B = env.node_velocity_controller.incidence_matrix
            for row, edge in enumerate(env.node_velocity_controller.edges):
                names = env.node_velocity_controller.node_names
                src, dst = names.index(edge.from_node), names.index(edge.to_node)
                columns = np.where((index[0] == src) & (index[1] == dst))[0]
                self.assertTrue(np.any(signs[columns] == B[row, dst]))
            reversed_meta = replace(metadata, actuator_edges=[
                replace(e, from_node=e.to_node, to_node=e.from_node) for e in metadata.actuator_edges])
            model = copy(env.mj_model)
            model.control_graph = reversed_meta
            np.testing.assert_array_equal(control_edge_directions(model), -signs)
            controller = copy(env.node_velocity_controller)
            controller.edges = [replace(e, from_node=e.to_node, to_node=e.from_node) for e in controller.edges]
            controller.incidence_matrix = controller._build_incidence_matrix()
            u = np.linspace(-.5, .5, len(controller.node_names))
            np.testing.assert_allclose(controller.transform(-u), env.node_velocity_controller.transform(u))
            graph = Data(**{k: torch.as_tensor(v) for k, v in raw.items()})
            prepared = prepare_graph(graph, use_virtual_node=True, use_edge_roles=True, use_edge_direction=True)
            torch.testing.assert_close(prepared.edge_attr[:len(signs), 3], torch.as_tensor(signs))
            self.assertTrue(prepared.edge_attr[len(signs):, 3].eq(0).all())
        finally:
            env.close()

    def test_dense_replay_and_resume_preserve_signed_features(self):
        cfg = replay_config(multitask=False, task='graph', use_virtual_node=True,
                            graph_features=dict(node_roles=True, edge_roles=True, edge_distance=True, edge_direction=True))
        steps = transition(1, 3, metadata=True)
        for step in steps:
            g = step['obs'];g.edge_direction = torch.tensor([1.,-1.,0.,0.,1.,-1.])
        g = steps[0]['obs'];static = build_graph_static(cfg,g,steps[1]['action'].squeeze(0))
        group = DenseGraphGroup(static,torch.stack([s['obs'].x for s in steps]),
                               torch.stack([s['obs'].rigidity for s in steps]))
        dense=dense_graph_batch(cfg,[group],'cpu')
        reference=Batch.from_data_list([prepare_graph(s['obs'],use_virtual_node=True,**graph_feature_flags(cfg)) for s in steps])
        torch.testing.assert_close(dense.x,reference.x);torch.testing.assert_close(dense.edge_attr,reference.edge_attr)
        buffer=TensorGNNBuffer(cfg);buffer.add(steps)
        state=buffer.state_dict();other=TensorGNNBuffer(cfg);other.load_state_dict(state)
        replay=next(iter(other.state_dict()['buffers'].values()))
        for raw in replay_observations(replay):torch.testing.assert_close(raw.edge_direction,g.edge_direction)
        changed=deepcopy(steps);changed[0]['obs'].edge_direction.neg_()
        with self.assertRaisesRegex(ValueError,'direction'):other.add(changed)

    def test_reconstruction_preserves_legacy_teacher_targets_and_rejects_misalignment(self):
        cfg=graph_test_cfg(domain_randomization=False, truss_topology='tetrahedron')
        env=MujocoPresetGraphEnv(cfg)
        try:
            raw,_=env.reset(seed=0);g=Data(**{k:torch.as_tensor(v) for k,v in raw.items()})
            student=deepcopy(cfg);student.graph_features['edge_roles']=True;student.graph_features['edge_direction']=True
            teacher_cfg=SimpleNamespace(**vars(cfg))
            metadata=reconstruct_control_metadata(teacher_cfg,student,g)
            enriched=attach_control_metadata(g,metadata)
            self.assertNotIn('edge_direction',g)
            teacher=GNNSAC(cfg)
            with torch.no_grad():
                a=teacher.model.policy_distribution(prepare_graph(g,use_virtual_node=True,**graph_feature_flags(cfg)))
                b=teacher.model.policy_distribution(prepare_graph(enriched,use_virtual_node=True,**graph_feature_flags(cfg)))
            for left,right in zip(a,b):torch.testing.assert_close(left,right,rtol=0,atol=0)
            corrupted=g.clone();corrupted.edge_index=corrupted.edge_index.flip(1)
            with self.assertRaisesRegex(ValueError,'ordering'):reconstruct_control_metadata(teacher_cfg,student,corrupted)
            self.assertNotIn('edge_direction',graph_feature_schema(cfg))
            self.assertIn('edge_direction',graph_feature_schema(student))
        finally:env.close()

    def test_signed_student_two_topology_cpu_distillation_smoke(self):
        from common.gnn_buffer import GNNBuffer
        from common.logger import Logger
        from env import make_env
        from trainer.base import Trainer
        from trainer.online_trainer import OnlineTrainer
        from tests.test_checkpointing import DummyLogger
        with tempfile.TemporaryDirectory() as tmp:
            settings = dict(work_dir=tmp, steps=8, seed_steps=1, pretrain_steps=1,
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
                                 graph_features=dict(node_roles=True, edge_roles=True, edge_direction=True, edge_distance=True),
                                 distillation=dict(enabled=True, teachers=mapping, pretrain_updates=2,
                                                   batch_size=2, shard_size=2, checkpoint_freq=0, log_freq=1,
                                                   reconstruct_control_metadata=True))
            env = make_env(cfg)
            try:
                agent = GNNSAC(cfg)
                trainer = OnlineTrainer(cfg=cfg, env=env, agent=agent,
                                        buffer=TensorGNNBuffer(cfg), logger=Logger(cfg))
                trainer.train()
                self.assertEqual(trainer.distillation.completed_updates, 2)
                self.assertEqual(trainer._step, cfg.steps)
                self.assertGreater(trainer._optimizer_updates, 0)
                self.assertTrue(all(torch.isfinite(x).all() for x in agent.model.parameters()))
            finally:
                env.close()

    def test_missing_or_invalid_signs_fail(self):
        graph=Data(x=torch.zeros(2,6),edge_index=torch.tensor([[0,1],[1,0]]))
        with self.assertRaisesRegex(ValueError,'metadata'):prepare_graph(graph,use_virtual_node=True,use_edge_direction=True)
        graph.edge_direction=torch.tensor([1.,2.])
        with self.assertRaisesRegex(ValueError,'directions'):prepare_graph(graph,use_virtual_node=True,use_edge_direction=True)


if __name__=='__main__':unittest.main()
