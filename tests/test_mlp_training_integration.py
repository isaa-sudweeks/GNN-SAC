"""Bounded native integration checks for the paper MLP workflows."""

from pathlib import Path
import sys
import tempfile
import unittest

import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "sac"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.logger import Logger
from common.parser import parse_cfg
from common.tensor_gnn_buffer import DenseGraphGroup, build_graph_static, make_gnn_buffer
from env import make_env
from gnn_infer import _make_agent
from padded_mlp_sac import PaddedMLPSAC
from trainer.online_trainer import OnlineTrainer


def integration_config(directory, **overrides):
    """Resolve actual Hydra defaults with small, deterministic CPU budgets."""
    options = dict(
        sac_backend="padded_mlp", device="cpu", domain_randomization=False,
        work_dir=str(directory), padded_mlp_hidden_dims="[16,16]",
        truss_topology="tetrahedron", padded_mlp_max_nodes=12,
        steps=16, seed_steps=4, pretrain_steps=1, batch_size=2,
        buffer_size=32, max_steps=4, nsubsteps=1, replay_ratio=1,
        eval_freq=100000, eval_episodes=1, eval_at_end=False,
        save_agent=False, save_csv=False, save_video=False,
        enable_wandb=False, checkpoint_freq=0, checkpoint_async=False,
        replay_storage="cpu_pinned",
    )
    options.update(overrides)
    arguments = [
        f"{key}={str(value).lower() if isinstance(value, bool) else value}"
        for key, value in options.items()
    ]
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        return parse_cfg(compose(config_name="config", overrides=arguments))


def make_trainer(cfg):
    environment = make_env(cfg)
    try:
        return OnlineTrainer(
            cfg=cfg, env=environment, agent=PaddedMLPSAC(cfg),
            buffer=make_gnn_buffer(cfg), logger=Logger(cfg),
        )
    except BaseException:
        environment.close()
        raise


def close_trainer(trainer):
    trainer.env.close()
    if trainer.eval_env is not trainer.env:
        trainer.eval_env.close()
    trainer.logger.finish()


class PaddedTrainingIntegrationTest(unittest.TestCase):
    def test_mjx_vectorized_padded_training_updates_tensor_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = integration_config(
                directory, sim_backend="mjx", num_envs=4,
                truss_topologies="[tetrahedron,octahedron]",
                vectorized_collection=True, steps=4, seed_steps=0,
                batch_size=1, max_steps=2,
            )
            trainer = make_trainer(cfg)
            try:
                trainer.train()
                self.assertEqual(trainer._step, cfg.steps)
                self.assertGreater(trainer._optimizer_updates, 0)
                self.assertEqual(set(trainer.buffer.sizes_by_task),
                                 {"truss-graph:tetrahedron", "truss-graph:octahedron"})
                self.assertTrue(all(size > 0 for size in trainer.buffer.sizes_by_task.values()))
            finally:
                close_trainer(trainer)

    def test_dense_collection_actions_match_serial_with_passive_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = integration_config(directory)
            agent = PaddedMLPSAC(cfg)
            groups, graph_groups = [], []
            for topology in ("tetrahedron", "octahedron"):
                env_cfg = integration_config(directory, truss_topology=topology)
                environment = make_env(env_cfg)
                try:
                    first = environment.reset()
                    second = first.clone()
                    second.x = second.x + 0.1
                    # A disabled formerly-active row tests per-observation
                    # masks as well as the permanent passive-node mask.
                    second.action_mask = second.action_mask.clone()
                    second.action_mask[second.action_mask.nonzero()[0]] = False
                    observations = [first, second]
                    static = build_graph_static(
                        cfg, first, torch.zeros(first.num_nodes, 1),
                    )
                    groups.append(DenseGraphGroup(
                        static=static, x=torch.stack([obs.x for obs in observations]),
                        rigidity=torch.stack([obs.rigidity for obs in observations]),
                        action_mask=torch.stack([obs.action_mask for obs in observations]),
                    ))
                    graph_groups.append(observations)
                finally:
                    environment.close()
            for group, observations, dense in zip(
                groups, graph_groups, agent.act_dense(groups, eval_mode=True),
            ):
                expected = torch.stack(agent.act_batch(observations, eval_mode=True))
                torch.testing.assert_close(dense, expected, rtol=1e-5, atol=1e-6)
                self.assertTrue(torch.all(dense[~group.action_mask] == 0))

    def test_tensor_replay_single_and_heterogeneous_pcgrad_training(self):
        for topologies, pcgrad in ((None, False),
                                   ("[tetrahedron,octahedron]", False),
                                   ("[tetrahedron,octahedron]", True)):
            with self.subTest(topologies=topologies, pcgrad=pcgrad):
                with tempfile.TemporaryDirectory() as directory:
                    options = {"pcgrad": pcgrad}
                    if topologies is not None:
                        options["truss_topologies"] = topologies
                    cfg = integration_config(directory, **options)
                    trainer = make_trainer(cfg)
                    try:
                        trainer.train()
                        self.assertEqual(trainer._step, cfg.steps)
                        self.assertGreater(trainer._optimizer_updates, 0)
                        self.assertGreater(trainer.buffer.size, 0)
                        self.assertEqual(len(trainer.buffer.task_names),
                                         1 if topologies is None else 2)
                        for parameter in trainer.agent.model.parameters():
                            self.assertTrue(torch.isfinite(parameter).all())
                    finally:
                        close_trainer(trainer)

    def test_full_checkpoint_and_export_restore_then_continue_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = integration_config(directory)
            trainer = make_trainer(cfg)
            try:
                trainer.train()
                snapshot = trainer._checkpoint_state_snapshot()
                observation = trainer.env.reset()
                expected_action = trainer.agent.act(observation, eval_mode=True)
                export = Path(directory) / "student.pt"
                trainer.agent.save(export)
                restored_agent = _make_agent(cfg)
                restored_agent.load(export)
                torch.testing.assert_close(
                    restored_agent.act(observation, eval_mode=True), expected_action,
                )
            finally:
                close_trainer(trainer)

            restored = make_trainer(integration_config(directory))
            try:
                restored.load_checkpoint_state_dict(snapshot)
                self.assertEqual(restored._step, snapshot["trainer"]["step"])
                self.assertEqual(restored._optimizer_updates,
                                 snapshot["trainer"]["optimizer_updates"])
                self.assertEqual(restored.buffer.size, trainer.buffer.size)
                for name, expected in snapshot["agent"]["model"].items():
                    torch.testing.assert_close(
                        restored.agent.model.state_dict()[name], expected,
                    )
                torch.testing.assert_close(restored.agent.log_alpha,
                                           snapshot["agent"]["log_alpha"])
                for optimizer in (restored.agent.pi_optim, restored.agent.q_optim,
                                  restored.agent.alpha_optim):
                    self.assertTrue(optimizer.state)
                # Continue from restored replay rather than relying on an
                # unchanged already-finished interaction budget.
                for _ in range(2):
                    metrics = restored.agent.update(restored.buffer)
                    self.assertTrue(torch.isfinite(metrics["pi_loss"]))
                after = restored.agent.model.state_dict()
                self.assertTrue(any(
                    not torch.equal(after[name], before)
                    for name, before in snapshot["agent"]["model"].items()
                    if name.startswith("_pi.")
                ))
            finally:
                close_trainer(restored)

    def test_heldout_evaluation_does_not_add_replay_or_normalization_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = integration_config(
                directory, eval_extra_topologies="[octahedron]", eval_at_end=True,
            )
            trainer = make_trainer(cfg)
            try:
                trainer.train()
                self.assertEqual(trainer._eval_heldout_topologies, ["octahedron"])
                self.assertEqual(trainer.buffer.task_names, ["truss-graph"])
                replay_size = trainer.buffer.size
                normalization = trainer.reward_normalizer.state_dict()
                trainer._evaluate_and_log()
                self.assertEqual(trainer.buffer.size, replay_size)
                self.assertEqual(trainer.reward_normalizer.state_dict(), normalization)
            finally:
                close_trainer(trainer)


if __name__ == "__main__":
    unittest.main()
