from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sac"))

from common.distillation import (
    Distillation, ObservationShards, gaussian_forward_kl, graph_mean_kl,
    graph_signature, replay_observations,
)
from common.gnn_buffer import GNNBuffer
from common.graph_transforms import prepare_graph
from common.logger import Logger
from gnn_sac import GNNSAC
from trainer.base import Trainer
from trainer.online_trainer import OnlineTrainer
from tests.test_gnn_batched_inference import agent_cfg, graph
from tests.test_checkpointing import DummyLogger


def config(directory, **overrides):
    values = vars(agent_cfg()).copy()
    values.update(sac_backend="gnn", task="truss-graph", truss_topology="a",
                  truss_topologies=["a", "b"], tasks=["truss-graph:a", "truss-graph:b"],
                  multitask=True, steps=100, buffer_size=16, batch_size=4, seed=3,
                  work_dir=str(directory), checkpoint_freq=0, use_virtual_node=True,
                  target_entropy=-1, log_std_min=-2.0, log_std_max=0.0,
                  distillation=dict(enabled=True, teachers={}, pretrain_updates=3,
                                    batch_size=4, initial_weight=1., decay_fraction=.5,
                                    shard_size=2, checkpoint_freq=0, log_freq=1))
    values.update(overrides)
    return SimpleNamespace(**values)


def raw_graph(nodes):
    obs = graph(nodes)
    obs.action_mask = torch.arange(nodes) != nodes - 1
    return obs


def teacher_checkpoint(directory, topology, nodes, **overrides):
    cfg = config(directory, truss_topology=topology, truss_topologies=None,
                 multitask=False, tasks=["truss-graph"], **overrides)
    agent = GNNSAC(cfg)
    observations = [raw_graph(nodes) for _ in range(5)]
    replay = dict(capacity=5, size=5, idx=2, obs=observations)
    path = Path(directory) / f"{topology}.pt"
    torch.save(dict(config=vars(cfg), agent=agent.training_state_dict(), buffer=replay), path)
    return str(path)


def setup(directory, **overrides):
    cfg = config(directory, **overrides)
    cfg.distillation["teachers"] = {
        "a": teacher_checkpoint(directory, "a", 3, embedding_dim=8),
        "b": teacher_checkpoint(directory, "b", 5, embedding_dim=12),
        "heldout": "/does/not/exist.pt",
    }
    return cfg, Distillation(cfg, cfg.tasks)


def replay_buffer(cfg):
    buffer = GNNBuffer(cfg)
    for task, nodes in zip(cfg.tasks, (3, 5)):
        obs = raw_graph(nodes)
        item = dict(obs=obs, action=torch.zeros(nodes, 1), reward=torch.tensor(0.), terminated=torch.tensor(False))
        buffer.add([item] * 5, task=task)
    return buffer


class GaussianKLTest(unittest.TestCase):
    def test_matches_pytorch_and_trains_mean_and_variance(self):
        tm = torch.tensor([[0.1, -.2]], dtype=torch.double)
        tl = torch.tensor([[-1., -.5]], dtype=torch.double)
        sm = torch.tensor([[.3, .1]], dtype=torch.double, requires_grad=True)
        sl = torch.tensor([[-.7, -.2]], dtype=torch.double, requires_grad=True)
        actual = gaussian_forward_kl(tm, tl, sm, sl)
        expected = torch.distributions.kl_divergence(
            torch.distributions.Normal(tm, tl.exp()), torch.distributions.Normal(sm, sl.exp()))
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertTrue(sm.grad.ne(0).all())
        self.assertTrue(sl.grad.ne(0).all())
        torch.testing.assert_close(gaussian_forward_kl(tm, tl, tm, tl), torch.zeros_like(tm))
        self.assertTrue((gaussian_forward_kl(tm, tl, tm, tl + .2) > 0).all())

    def test_invalid_distributions_and_node_normalization(self):
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            gaussian_forward_kl(torch.tensor([float("nan")]), torch.zeros(1), torch.zeros(1), torch.zeros(1))
        batch = Batch.from_data_list([prepare_graph(raw_graph(n), use_virtual_node=True) for n in (3, 5)])
        # 2 active nodes at KL=2, 4 at KL=6 => graph mean 4, not node mean 14/3.
        kl = torch.tensor([[2.]] * 2 + [[6.]] * 4)
        self.assertEqual(float(graph_mean_kl(kl, batch)), 4.)

    def test_distribution_interface_preserves_policy_and_rng(self):
        cfg = config("/tmp")
        agent = GNNSAC(cfg)
        obs = prepare_graph(raw_graph(3), use_virtual_node=True)
        rng = torch.random.get_rng_state()
        mean, log_std = agent.model.policy_distribution(obs)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        torch.testing.assert_close(mean.tanh(), agent.model.pi_mean(obs))
        _, info = agent.model.pi(obs)
        torch.testing.assert_close(info["mean"], mean.tanh())
        torch.testing.assert_close(info["log_std"], log_std)

    def test_distribution_interface_disables_dropout_and_restores_mode(self):
        agent = GNNSAC(config("/tmp", dropout=.5))
        agent.model.train()
        obs = prepare_graph(raw_graph(3), use_virtual_node=True)
        rng = torch.random.get_rng_state()
        first = agent.model.policy_distribution(obs)
        second = agent.model.policy_distribution(obs)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(agent.model._pi.training)
        self.assertTrue(agent.model._action_head.training)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b)


class TeacherReplayTest(unittest.TestCase):
    def test_hydra_preset_and_teacher_mapping(self):
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
            cfg = compose(config_name="archieved/gnn_config", overrides=[
                "distillation=kl", "+distillation.teachers={a:/tmp/a.pt,b:/tmp/b.pt}"])
        self.assertTrue(cfg.distillation.enabled)
        self.assertEqual(cfg.distillation.pretrain_updates, 10000)
        self.assertEqual(cfg.distillation.batch_size, 256)
        self.assertEqual(cfg.distillation.teachers.a, "/tmp/a.pt")

    def test_partial_and_wrapped_replay_and_shard_reuse(self):
        values = [raw_graph(3) for _ in range(5)]
        for i, value in enumerate(values):
            value.x.fill_(i)
        partial = dict(capacity=5, size=3, idx=3, obs=values[:3] + [None, None])
        self.assertEqual([int(g.x[0, 0]) for g in replay_observations(partial)], [0, 1, 2])
        full = dict(capacity=5, size=5, idx=2, obs=values)
        self.assertEqual([int(g.x[0, 0]) for g in replay_observations(full)], [2, 3, 4, 0, 1])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shards"
            dataset = ObservationShards.prepare(path, replay_observations(full), 2, graph_signature(values[0]))
            self.assertEqual(dataset.sizes.tolist(), [2, 2, 1])
            rng = torch.Generator().manual_seed(5)
            counts = torch.zeros(5)
            for _ in range(1000):
                for value in dataset.sample(4, rng):
                    counts[int(value.x[0, 0])] += 1
            self.assertTrue((counts > 600).all(), counts)
            self.assertTrue((counts < 1000).all(), counts)
            self.assertEqual(ObservationShards.prepare(path, iter(()), 2, {}).sizes.tolist(), [2, 2, 1])
        with self.assertRaises(ValueError):
            list(replay_observations(dict(full, size=0)))

    def test_frozen_teachers_routing_and_split_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, distill = setup(tmp)
            self.assertEqual(set(distill.teachers), set(cfg.tasks))
            self.assertTrue(all(not p.requires_grad for t in distill.teachers.values() for p in t.parameters()))
            agent = GNNSAC(cfg)
            obs = Batch.from_data_list([distill.prepare(raw_graph(3))])
            distill.loss(agent.model, cfg.tasks[0], obs).backward()
            self.assertTrue(all(p.grad is None for t in distill.teachers.values() for p in t.parameters()))
            with self.assertRaisesRegex(ValueError, "ordering"):
                distill.loss(agent.model, cfg.tasks[1], obs)
            with self.assertRaisesRegex(ValueError, "Missing"):
                Distillation(cfg, ["truss-graph:missing"])
            cfg.obs_dim = 4
            with self.assertRaisesRegex(ValueError, "convention"):
                Distillation(cfg, cfg.tasks)

    def test_bad_topology_schema_and_control_conventions(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, _ = setup(tmp)
            path = cfg.distillation["teachers"]["a"]
            original = torch.load(path, weights_only=False)
            for key, value in (("truss_topology", "wrong"), ("speed", 99),
                               ("graph_features", {"node_roles": True})):
                bad = deepcopy(original)
                bad["config"][key] = value
                torch.save(bad, path)
                with self.assertRaises(ValueError):
                    Distillation(cfg, cfg.tasks)
            torch.save(original, path)


class DistillationTrainingTest(unittest.TestCase):
    def test_online_only_skips_shards_and_optimizer_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, _ = setup(tmp)
            cfg.distillation["pretrain_updates"] = 0
            with patch.object(ObservationShards, "prepare", side_effect=AssertionError("Unexpected offline data")):
                distill = Distillation(cfg, cfg.tasks)
            self.assertEqual(distill.stage, "online")
            self.assertFalse(distill.datasets)
            agent = GNNSAC(cfg)
            agent.pi_optim.state["sentinel"] = {}
            distill.finish_pretraining(agent)
            self.assertIn("sentinel", agent.pi_optim.state)

    def test_wandb_offline_updates_do_not_reuse_environment_event_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(tmp, enable_wandb=False, save_agent=False, save_csv=False)
            logger = Logger(cfg)
            logger._wandb = Mock()
            logger.log(dict(step=0, offline_updates=1, kl=1.), "distillation")
            logger.log(dict(step=0, offline_updates=2, kl=.5), "distillation")
            logger.log(dict(step=0, episode_reward=1.), "eval")
            for call in logger._wandb.log.call_args_list:
                self.assertNotIn("step", call.kwargs)
            self.assertEqual(logger._wandb.log.call_count, 3)

    def test_offline_learning_only_changes_actor_and_transition_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, distill = setup(tmp)
            distill.pretrain_updates = 30
            agent = GNNSAC(cfg)
            critics = deepcopy(agent.model._Qs.state_dict())
            targets = deepcopy(agent.model._target_Qs.state_dict())
            alpha = agent.log_alpha.detach().clone()
            first = distill.offline_update(agent)["kl"]
            for _ in range(29):
                last = distill.offline_update(agent)["kl"]
            self.assertLess(last, first)
            for key, value in critics.items():
                torch.testing.assert_close(agent.model._Qs.state_dict()[key], value, rtol=0, atol=0)
            for key, value in targets.items():
                torch.testing.assert_close(agent.model._target_Qs.state_dict()[key], value, rtol=0, atol=0)
            torch.testing.assert_close(agent.log_alpha, alpha, rtol=0, atol=0)
            self.assertFalse(agent.q_optim.state)
            self.assertFalse(agent.alpha_optim.state)
            self.assertTrue(agent.pi_optim.state)
            distill.finish_pretraining(agent)
            self.assertFalse(agent.pi_optim.state)
            agent.pi_optim.state["sentinel"] = {}
            distill.finish_pretraining(agent)
            self.assertIn("sentinel", agent.pi_optim.state)

    def test_resume_matches_uninterrupted_offline_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, distill = setup(tmp)
            agent = GNNSAC(cfg)
            trainer = Trainer(cfg=cfg, env=None, agent=agent, buffer=GNNBuffer(cfg), logger=DummyLogger())
            trainer.distillation = distill
            distill.offline_update(agent)
            snapshot = trainer._checkpoint_state_snapshot()
            expected = distill.offline_update(agent)
            restored = GNNSAC(cfg)
            resumed = Trainer(cfg=cfg, env=None, agent=restored, buffer=GNNBuffer(cfg), logger=DummyLogger())
            resumed.distillation = Distillation(cfg, cfg.tasks)
            resumed.load_checkpoint_state_dict(snapshot)
            actual = resumed.distillation.offline_update(restored)
            self.assertEqual(actual, expected)
            for key, value in agent.model.state_dict().items():
                torch.testing.assert_close(value, restored.model.state_dict()[key], rtol=0, atol=0)
            changed = deepcopy(snapshot["distillation"])
            changed["sources"][cfg.tasks[0]]["sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "changed"):
                resumed.distillation.load_state_dict(changed)

    def test_online_ordinary_and_pcgrad_and_zero_weight_equivalence(self):
        for pcgrad in (False, True):
            with self.subTest(pcgrad=pcgrad), tempfile.TemporaryDirectory() as tmp:
                cfg, distill = setup(tmp, pcgrad=pcgrad)
                agent = GNNSAC(cfg)
                agent.distillation = distill
                buffer = replay_buffer(cfg)
                info = agent.update(buffer, compute_diagnostics=True)
                self.assertGreater(float(info["distillation/kl"]), 0)
                torch.testing.assert_close(info["pi_loss"].double(),
                    info["distillation/sac_actor_loss"].double() + info["distillation/weighted_kl"].double())
                distill.step = 50
                baseline = GNNSAC(cfg)
                baseline.load_training_state_dict(deepcopy(agent.training_state_dict()))
                rng = torch.random.get_rng_state()
                with patch.object(distill, "loss", side_effect=AssertionError("Teacher called at zero weight")):
                    agent.update(buffer)
                torch.random.set_rng_state(rng)
                baseline.update(buffer)
                for key, value in baseline.model.state_dict().items():
                    torch.testing.assert_close(value, agent.model.state_dict()[key], rtol=0, atol=0)

    def test_stage_orchestration_and_decay_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, distill = setup(tmp)
            agent = GNNSAC(cfg)
            trainer = Trainer(cfg=cfg, env=None, agent=agent, buffer=GNNBuffer(cfg), logger=DummyLogger())
            trainer.distillation = distill
            trainer._step = 0
            trainer.logger.log = Mock()
            trainer.eval = Mock()
            trainer.save_checkpoint = Mock()
            OnlineTrainer._run_distillation_pretraining(trainer)
            trainer.eval.assert_called_once()
            self.assertEqual(trainer.buffer.size, 0)
            self.assertEqual(trainer._step, 0)
            OnlineTrainer._run_distillation_pretraining(trainer)
            trainer.eval.assert_called_once()
            trainer._step = 25
            snapshot = trainer._checkpoint_state_snapshot()
            trainer.load_checkpoint_state_dict(snapshot)
            self.assertEqual(distill.weight, .5)
            self.assertEqual(distill.stage, "online")
            self.assertFalse(distill.datasets)

    def test_two_topology_cpu_environment_smoke(self):
        from common.logger import Logger
        from env import make_env
        from tests.test_gnn_mujoco_truss_gen_smoke import graph_test_cfg

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
                        episode.append(dict(obs=obs, action=action, reward=reward,
                                            terminated=info["terminated"]))
                    buffer.add(episode)
                    trainer = Trainer(cfg=cfg, env=env, agent=agent, buffer=buffer, logger=DummyLogger())
                    path = Path(tmp) / f"{topology}.pt"
                    torch.save(trainer.checkpoint_state_dict(), path)
                    mapping[topology] = str(path)
                finally:
                    env.close()
            cfg = graph_test_cfg(**settings, truss_topologies=["tetrahedron", "octahedron"],
                                 distillation=dict(enabled=True, teachers=mapping, pretrain_updates=2,
                                                   batch_size=2, shard_size=2, checkpoint_freq=0, log_freq=1))
            env = make_env(cfg)
            try:
                agent = GNNSAC(cfg)
                trainer = OnlineTrainer(cfg=cfg, env=env, agent=agent,
                                        buffer=GNNBuffer(cfg), logger=Logger(cfg))
                trainer.train()
                self.assertEqual(trainer.distillation.completed_updates, 2)
                self.assertEqual(trainer.distillation.stage, "online")
                self.assertEqual(trainer._step, 8)
                self.assertGreater(trainer._optimizer_updates, 0)
                self.assertEqual(trainer.distillation.weight, 0)
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
