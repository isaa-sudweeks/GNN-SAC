from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from omegaconf import OmegaConf
from torch_geometric.data import Data

from common.gnn_buffer import GNNBuffer
from common.parser import parse_cfg
from env import make_env
from gnn_sac import GNNSAC


def flat_test_cfg(**overrides):
    cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "config" / "algorithm.yaml"),
        OmegaConf.load(ROOT / "config" / "environment.yaml"),
        OmegaConf.create(
            {
                "save_video": False,
                "multitask": False,
                "device": "cpu",
                "steps": 20,
                "seed_steps": 1,
                "batch_size": 2,
                "enable_wandb": False,
                "save_csv": False,
                "save_agent": False,
                "work_dir": str(ROOT / "logs" / "test-smoke"),
            }
        ),
    )
    cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return parse_cfg(cfg)


def graph_test_cfg(**overrides):
    cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "config" / "algorithm.yaml"),
        OmegaConf.load(ROOT / "config" / "environment.yaml"),
        OmegaConf.load(ROOT / "config" / "gnn_config.yaml"),
        OmegaConf.create(
            {
                "save_video": False,
                "multitask": False,
                "device": "cpu",
                "steps": 20,
                "seed_steps": 1,
                "batch_size": 2,
                "enable_wandb": False,
                "save_csv": False,
                "save_agent": False,
                "work_dir": str(ROOT / "logs" / "test-smoke"),
            }
        ),
    )
    cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return parse_cfg(cfg)


class GNNMujocoTrussGenSmokeTest(unittest.TestCase):
    def test_repeated_truss_envs_reset_and_step(self):
        cfg = flat_test_cfg(
            num_envs=4,
            max_steps=2,
            episode_length=2,
        )
        env = make_env(cfg)
        try:
            self.assertEqual(len(env.env.envs), 4)
            self.assertEqual(cfg.action_dim, 8)
            observations = env.reset_many(env_indices=range(cfg.num_envs))
            actions = [env.rand_act() for _ in range(cfg.num_envs)]
            results = env.step_many(actions, env_indices=range(cfg.num_envs))
            self.assertEqual(len(observations), cfg.num_envs)
            self.assertEqual(len(results), cfg.num_envs)
            for env_idx, (next_obs, reward, done, info) in enumerate(results):
                self.assertEqual(observations[env_idx].shape, next_obs.shape)
                self.assertEqual(info["task"], cfg.task)
                self.assertEqual(info["env_idx"], env_idx)
                self.assertTrue(float(reward) == float(reward))
            for env_idx in range(cfg.num_envs):
                obs = env.reset(task_idx=env_idx)
                action = env.rand_act()
                next_obs, reward, done, info = env.step(action)
                self.assertEqual(obs.shape, next_obs.shape)
                self.assertEqual(action.shape, (cfg.action_dim,))
                self.assertEqual(info["task"], cfg.task)
                self.assertEqual(info["env_idx"], env_idx)
                self.assertNotIn("task_idx", info)
                self.assertTrue(float(reward) == float(reward))
        finally:
            env.close()

    def test_graph_env_reset_and_node_action_step(self):
        cfg = graph_test_cfg()
        env = make_env(cfg)
        try:
            obs = env.reset()
            self.assertIsInstance(obs, Data)
            self.assertEqual(obs.x.shape[0], cfg.num_nodes)
            self.assertEqual(obs.edge_index.shape[0], 2)
            self.assertEqual(cfg.node_action_dim, 1)
            self.assertEqual(cfg.action_dim, cfg.node_action_dim)
            self.assertEqual(cfg.num_policy_actions, cfg.num_nodes * cfg.node_action_dim)
            self.assertEqual(cfg.num_actuators, env.unwrapped.mj_model.model.nu)

            action = env.rand_act()
            next_obs, reward, done, info = env.step(action)
            self.assertIsInstance(next_obs, Data)
            self.assertEqual(action.shape, (cfg.num_nodes, 1))
            self.assertEqual(env.unwrapped._node_action_to_actuator_action(action.numpy()).shape[0], env.unwrapped.mj_model.model.nu)
            self.assertTrue(float(reward) == float(reward))
            self.assertIn("terminated", info)
            self.assertIn("truncated", info)
        finally:
            env.close()

    def test_gnn_sac_update_smoke(self):
        cfg = graph_test_cfg()
        env = make_env(cfg)
        try:
            agent = GNNSAC(cfg)
            buffer = GNNBuffer(cfg)
            self.assertEqual(agent.target_entropy, -float(cfg.num_policy_actions))

            obs0 = env.reset()
            action0 = agent.act(obs0)
            obs1, reward1, _, info1 = env.step(action0)
            action1 = env.rand_act()
            obs2, reward2, _, info2 = env.step(action1)

            episode = [
                {"obs": obs0, "action": action0, "reward": reward1, "terminated": info1["terminated"]},
                {"obs": obs1, "action": action1, "reward": reward2, "terminated": info2["terminated"]},
                {"obs": obs2, "action": action1, "reward": reward2, "terminated": info2["terminated"]},
            ]
            buffer.add(episode)
            obs, action, reward, terminated, next_obs = buffer.sample()

            self.assertEqual(action.shape[1], 1)
            self.assertEqual(reward.shape[0], cfg.batch_size)
            self.assertEqual(terminated.shape[0], cfg.batch_size)
            self.assertEqual(obs.x.shape[0], next_obs.x.shape[0])

            update_info = agent.update(buffer)
            self.assertIn("value_loss", update_info)
            self.assertIn("pi_loss", update_info)
            self.assertIn("alpha", update_info)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
