from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest

import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.logger import Logger
from common.parser import parse_cfg
from common.tensor_gnn_buffer import make_gnn_buffer
from env import make_env
from gnn_sac import GNNSAC
from trainer.online_trainer import OnlineTrainer
from trainer.vector_collection import VectorBucket


def training_cfg(work_dir, **overrides):
    """Compose the production Hydra config for a tiny CPU MJX run."""
    from hydra import compose, initialize_config_dir

    values = {
        "sac_backend": "gnn",
        "sim_backend": "mjx",
        "mjx_impl": "jax",
        "truss_topologies": "[octahedron,tetrahedron]",
        "num_envs": 4,
        "nsubsteps": 1,
        "max_steps": 5,
        "graph_features.node_roles": True,
        "device": "cpu",
        "steps": 16,
        "seed_steps": 8,
        "pretrain_steps": 2,
        "batch_size": 2,
        "buffer_size": 64,
        "eval_freq": 1_000_000,
        "eval_episodes": 1,
        "eval_at_end": False,
        "checkpoint_freq": 0,
        "save_agent": False,
        "save_csv": False,
        "save_video": False,
        "enable_wandb": False,
        "replay_storage": "cpu_pinned",
        "work_dir": str(work_dir),
        **overrides,
    }
    override_list = [
        f"{key}={str(value).lower() if isinstance(value, bool) else value}"
        for key, value in values.items()
    ]
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        cfg = compose(config_name="config", overrides=override_list)
    return parse_cfg(cfg)


class VectorizedCollectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from mujoco_truss_gen import MjxNodeVelocityEnv  # noqa: F401
        except (ImportError, AttributeError) as exc:
            raise unittest.SkipTest(f"updated mujoco-truss-gen is unavailable: {exc}")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_step_batch_matches_per_environment_step(self):
        cfg = training_cfg(self._tmp.name)
        per_env, batched = make_env(cfg), make_env(training_cfg(self._tmp.name))
        try:
            observations = per_env.reset_many()
            groups = batched.env.batched_groups()
            dense = [bucket.reset_batch() for bucket, _ in groups]
            actions = [
                torch.linspace(-1, 1, obs["x"].size(0) * obs["x"].size(1)).view(*obs["x"].shape[:2], 1)
                for obs in dense
            ]
            results = per_env.step_many([
                actions[index % 2][index // 2] for index in range(per_env.num_envs)
            ])
            for bucket_idx, ((bucket, global_indices), action) in enumerate(zip(groups, actions)):
                next_obs, reward, done, info = bucket.step_batch(action)
                for local_idx, global_idx in enumerate(global_indices.tolist()):
                    reset_obs = observations[global_idx]
                    torch.testing.assert_close(dense[bucket_idx]["x"][local_idx], reset_obs.x)
                    graph, expected_reward, expected_done, expected_info = results[global_idx]
                    torch.testing.assert_close(next_obs["x"][local_idx], graph.x)
                    torch.testing.assert_close(next_obs["rigidity"][local_idx], graph.rigidity)
                    self.assertTrue(torch.equal(next_obs["action_mask"][local_idx], graph.action_mask))
                    torch.testing.assert_close(reward[local_idx], expected_reward)
                    self.assertEqual(bool(done[local_idx]), expected_done)
                    torch.testing.assert_close(
                        info["terminated"][local_idx].float(), expected_info["terminated"]
                    )
        finally:
            per_env.close()
            batched.close()

    def test_act_dense_matches_act_batch(self):
        cfg = training_cfg(self._tmp.name)
        env = make_env(cfg)
        try:
            agent = GNNSAC(cfg)
            buckets = [VectorBucket(bucket, indices) for bucket, indices in env.env.batched_groups()]
            for bucket in buckets:
                bucket.reset_done(lambda x: x)
            dense_actions = agent.act_dense([bucket.policy_group(cfg) for bucket in buckets], eval_mode=True)
            for bucket, actions in zip(buckets, dense_actions):
                graphs = [
                    Data(
                        x=bucket.obs["x"][row],
                        edge_index=bucket.env.edge_index,
                        action_mask=bucket.obs["action_mask"][row],
                        rigidity=bucket.obs["rigidity"][row],
                    )
                    for row in range(bucket.size)
                ]
                expected = torch.stack(agent.act_batch(graphs, eval_mode=True))
                torch.testing.assert_close(actions, expected, rtol=1e-5, atol=1e-6)
                self.assertTrue(torch.all(actions[~bucket.obs["action_mask"]] == 0))
        finally:
            env.close()

    def _train(self, mode):
        work_dir = Path(self._tmp.name) / mode
        cfg = training_cfg(work_dir, vectorized_collection=mode, seed=3)
        env = make_env(cfg)
        agent = GNNSAC(cfg)
        buffer = make_gnn_buffer(cfg)
        trainer = OnlineTrainer(cfg=cfg, env=env, agent=agent, logger=Logger(cfg), buffer=buffer)
        try:
            trainer.train()
        finally:
            env.close()
        return trainer

    def test_vectorized_training_matches_per_environment_bookkeeping(self):
        vectorized = self._train("true")
        per_env = self._train("false")
        self.assertEqual(vectorized._step, per_env._step)
        self.assertEqual(vectorized._ep_idx, per_env._ep_idx)
        self.assertEqual(vectorized._optimizer_updates, per_env._optimizer_updates)
        self.assertEqual(vectorized.buffer.sizes_by_task, per_env.buffer.sizes_by_task)
        stats = vectorized.reward_normalizer.metrics()
        self.assertEqual(
            {task: values["count"] for task, values in stats.items()},
            {task: values["count"] for task, values in per_env.reward_normalizer.metrics().items()},
        )

    def test_forced_vectorized_collection_rejects_unsupported_setups(self):
        trainer = SimpleNamespace(
            cfg=SimpleNamespace(vectorized_collection=True),
            env=SimpleNamespace(env=None),
            agent=SimpleNamespace(),
            buffer=SimpleNamespace(),
        )
        with self.assertRaisesRegex(ValueError, "requires an MJX vector environment"):
            OnlineTrainer._vector_collection_buckets(trainer)
        trainer.cfg.vectorized_collection = "auto"
        self.assertIsNone(OnlineTrainer._vector_collection_buckets(trainer))


if __name__ == "__main__":
    unittest.main()
