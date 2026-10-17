from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "sac"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.sgs import SuccessGuidedSampler, beta_probabilities
from common.gnn_buffer import GNNBuffer
from common.tensor_gnn_buffer import TensorGNNBuffer
from gnn_sac import GNNSAC
from trainer.online_trainer import OnlineTrainer
from tests.test_tensor_gnn_buffer import config, populate
from tests.test_checkpointing import make_trainer


def sampler(mode="both", **kwargs):
    return SuccessGuidedSampler(dict(mode=mode, warmup_steps=0, min_episodes=1, **kwargs),
                                ["graph:a", "graph:b"], seed=37)


def prime(value):
    for _ in range(10):
        value.record_episode("graph:a", {"success": True})
    for index in range(10):
        value.record_episode("graph:b", {"success": index < 6})
    return value


class SGSTest(unittest.TestCase):
    def test_paper_formula_and_appendix_floor(self):
        p = np.array([0, .2, .66, 1])
        for temperature in (.2, 2, 4):
            w = (p + 1e-8) ** (5 * .66) * (1 - p + 1e-8) ** (5 * .34)
            logits = np.log(np.maximum(w, 1e-8) + 1e-8) / max(temperature, 1)
            expected = np.exp(logits - logits.max())
            expected /= expected.sum()
            np.testing.assert_allclose(beta_probabilities(p, temperature=temperature), expected)
        self.assertEqual(int(beta_probabilities(p).argmax()), 2)
        self.assertTrue((beta_probabilities(p) > 0).all())
        np.testing.assert_allclose(beta_probabilities(p, concentration=0), [.25] * 4)
        self.assertTrue(np.isfinite(beta_probabilities(p, concentration=1e6)).all())

    def test_invalid_parameters_fail(self):
        for kwargs in ({"target": 2}, {"concentration": -1}, {"temperature": 0},
                       {"epsilon": 0}, {"temperature": float("nan")}):
            with self.assertRaises(ValueError):
                beta_probabilities([.5], **kwargs)
        for p in ([], [float("nan")], [-.1], [1.1]):
            with self.assertRaises(ValueError):
                beta_probabilities(p)
        with self.assertRaises(ValueError):
            sampler("typo")

    def test_history_warmup_and_training_only(self):
        value = SuccessGuidedSampler(dict(mode="both", warmup_steps=3, min_episodes=2, history_size=2), ["a", "b"])
        for task in value.tasks:
            for outcome in (False, True):
                value.record_episode(task, {"success": outcome})
        self.assertFalse(value.warmed_up)
        value.record_collection("a", 3)
        self.assertTrue(value.warmed_up)
        value.record_episode("a", {"success": True})
        self.assertEqual(list(value.histories["a"]), [True, True])
        with self.assertRaises(ValueError):
            value.record_episode("heldout", {"success": True})
        with self.assertRaises(ValueError):
            value.record_episode("a", {})
        with self.assertRaises(ValueError):
            value.record_episode("a", {"success": .5})
        survival = sampler(require_survival=True)
        survival.record_episode("graph:a", {"success": True, "terminated": True})
        self.assertEqual(list(survival.histories["graph:a"]), [False])

    def test_distance_outcomes_and_survival(self):
        value = sampler(success_metric="distance", distance_threshold=.5, require_survival=True)
        for distance, terminated, expected in ((.49, False, False), (.5, False, True),
                                                (.8, True, False), (-.8, False, False)):
            value.record_episode("graph:a", {"sgs_distance": distance, "terminated": terminated})
            self.assertEqual(value.histories["graph:a"][-1], expected)
        for info in ({"terminated": False}, {"sgs_distance": float("nan"), "terminated": False}):
            with self.assertRaises(ValueError):
                value.record_episode("graph:a", info)

    def test_fifteen_meters_and_four_hundred_survived_steps(self):
        value = sampler(success_metric="distance", distance_threshold=15, min_survival_steps=400)
        for distance, length, terminated, expected in (
            (14.99, 400, False, False), (15, 399, False, False),
            (15, 400, False, True), (15, 400, True, False),
            (15, 401, True, True), (16, 1000, False, True),
        ):
            with self.subTest(distance=distance, length=length, terminated=terminated):
                value.record_episode("graph:a", {"sgs_distance": distance,
                    "sgs_episode_length": length, "terminated": terminated})
                self.assertEqual(value.histories["graph:a"][-1], expected)
        for length in (float("nan"), -1, 399.5):
            with self.assertRaises(ValueError):
                value.record_episode("graph:a", {"sgs_distance": 15,
                    "sgs_episode_length": length, "terminated": False})
        with self.assertRaises(ValueError):
            value.record_episode("graph:a", {"sgs_distance": 15, "terminated": False})
        with self.assertRaises(ValueError):
            sampler(min_survival_steps=-1)

    def test_collection_rng_is_independent_of_replay_sampling(self):
        first, second = prime(sampler(collection_block_steps=1)), prime(sampler(collection_block_steps=1))
        for _ in range(20):
            first.replay_counts(40, 20)
            self.assertEqual(first.collection_group(first.tasks, [2, 2]),
                             second.collection_group(second.tasks, [2, 2]))

    def test_modes_and_fixed_replay_budget(self):
        for mode in ("collection", "replay", "both"):
            value = prime(sampler(mode))
            totals = dict.fromkeys(value.tasks, 0)
            for _ in range(100):
                counts = value.replay_counts(40, 20)
                self.assertEqual(sum(counts.values()), 40)
                for task, count in counts.items():
                    totals[task] += count
            if mode == "collection":
                self.assertEqual(totals, {"graph:a": 2000, "graph:b": 2000})
            else:
                self.assertGreater(totals["graph:b"], 3900)
            selected = value.collection_group(value.tasks, [2, 2])
            self.assertEqual(selected is None, mode == "replay")

    def test_collection_probability_accounts_for_batch_size(self):
        value = prime(sampler(collection_block_steps=1, uniform_mix=1))
        transitions = np.zeros(2)
        for _ in range(10000):
            index = value.collection_group(value.tasks, [1, 4])
            transitions[index] += [1, 4][index]
        self.assertLess(abs(transitions[0] / transitions.sum() - .5), .025)

    def test_rng_history_and_block_resume(self):
        original = prime(sampler(collection_block_steps=3))
        original.collection_group(original.tasks, [2, 2])
        restored = sampler(collection_block_steps=3)
        restored.load_state_dict(original.state_dict())
        for _ in range(10):
            self.assertEqual(original.collection_group(original.tasks, [2, 2]), restored.collection_group(restored.tasks, [2, 2]))
            self.assertEqual(original.replay_counts(40, 20), restored.replay_counts(40, 20))
        state = original.state_dict()
        original.record_episode("graph:a", {"success": False})
        self.assertNotEqual(state["histories"], original.state_dict()["histories"])
        with self.assertRaises(ValueError):
            sampler("collection", collection_block_steps=3).load_state_dict(state)

    def test_replay_both_backends_variable_counts_and_replacement(self):
        for cls in (GNNBuffer, TensorGNNBuffer):
            buffer = cls(config(buffer_size=32, batch_size=8))
            populate(buffer)
            buffer.sgs = prime(sampler("replay"))
            # The favored task gets more samples than its stored population;
            # zero-count tasks must not produce empty graph batches.
            batch = buffer.sample_with_tasks()
            self.assertEqual(sum(b[2].numel() for b in batch.by_task.values()), 8)
            self.assertEqual(batch.combined[2].numel(), 8)
            self.assertGreater(batch.by_task["graph:b"][2].numel(), 4)
            self.assertEqual(buffer._buffers["graph:b"]._batch_size, 4)

    def test_pcgrad_retains_replay_weighting(self):
        dummy = SimpleNamespace(sgs=prime(sampler()))
        batches = {"a": (None, None, torch.zeros(1)), "b": (None, None, torch.zeros(3))}
        gradients = [(torch.tensor([2.]),), (torch.tensor([4.]),)]
        scaled = GNNSAC._sgs_scale_task_gradients(dummy, gradients, batches)
        result = GNNSAC.pcgrad_project(scaled)[0]
        torch.testing.assert_close(result, torch.tensor([3.5]))

    def test_pcgrad_weighting_with_conflicting_gradients(self):
        dummy = SimpleNamespace(sgs=prime(sampler()))
        batches = {"a": (None, None, torch.zeros(1)), "b": (None, None, torch.zeros(3))}
        gradients = [(torch.tensor([1., -1.]),), (torch.tensor([1., 2.]),)]
        scaled = GNNSAC._sgs_scale_task_gradients(dummy, gradients, batches)
        # Orthogonal projection against the other ORIGINAL task gradient,
        # followed by 1/4 and 3/4 weighted reduction.
        torch.testing.assert_close(GNNSAC.pcgrad_project(scaled)[0], torch.tensor([1.425, .975]))

    def test_checkpoint_snapshot_and_mode_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer(directory)
            trainer.sgs = prime(sampler())
            snapshot = trainer._checkpoint_state_snapshot()
            trainer.sgs.record_episode("graph:a", {"success": False})
            trainer.load_checkpoint_state_dict(snapshot)
            self.assertEqual(trainer.sgs.state_dict()["histories"], snapshot["sgs"]["histories"])
            del trainer.sgs
            with self.assertRaisesRegex(ValueError, "SGS mode"):
                trainer.load_checkpoint_state_dict(snapshot)

class SGSTrainingSmokeTest(unittest.TestCase):
    def test_cpu_training_modes_native_and_mjx(self):
        from common.logger import Logger
        from common.tensor_gnn_buffer import make_gnn_buffer
        from env import make_env
        from tests.test_vectorized_collection import training_cfg

        with tempfile.TemporaryDirectory() as directory:
            for backend, mode in (("mujoco", "disabled"), ("mujoco", "collection"), ("mujoco", "replay"),
                                  ("mujoco", "both"), ("mjx", "both")):
                with self.subTest(backend=backend, mode=mode):
                    cfg = training_cfg(Path(directory) / f"{backend}-{mode}", **{
                        "sim_backend": backend, "steps": 15, "max_steps": 2,
                        "num_envs": 1 if backend == "mujoco" else 4,
                        "seed_steps": 4, "pretrain_steps": 1, "replay_ratio": .5,
                        "sgs.mode": mode, "sgs.warmup_steps": 0, "sgs.min_episodes": 1,
                        "sgs.collection_block_steps": 1, "progress_freq": 4, "eval_freq": 8,
                        "mpl_dims": "[8]", "message_hidden_dims": "[8]", "head_hidden_dims": "[8]",
                    })
                    env = make_env(cfg)
                    trainer = OnlineTrainer(cfg=cfg, env=env, agent=GNNSAC(cfg),
                                            buffer=make_gnn_buffer(cfg), logger=Logger(cfg))
                    self.assertIsNot(trainer.eval_env, env)
                    if backend == "mujoco":
                        original_step_many = env.step_many
                        native_envs = env.env.envs

                        def checked_step_many(actions, env_indices=None):
                            paused = [i for i in range(len(native_envs)) if i not in env_indices]
                            before = {i: native_envs[i].unwrapped.mj_model.data.qpos.copy() for i in paused}
                            result = original_step_many(actions, env_indices=env_indices)
                            for i in paused:
                                np.testing.assert_array_equal(native_envs[i].unwrapped.mj_model.data.qpos, before[i])
                            return result

                        env.step_many = checked_step_many
                    try:
                        trainer.train()
                        self.assertEqual(trainer._step, cfg.steps)
                        if mode == "disabled":
                            self.assertIsNone(trainer.sgs)
                        else:
                            self.assertEqual(sum(trainer.sgs.collected.values()), cfg.steps)
                            self.assertEqual(sum(trainer.sgs.episodes.values()), trainer.buffer.num_eps)
                            self.assertEqual(sum(trainer.sgs.replayed.values()), trainer._optimizer_updates * cfg.batch_size)
                            self.assertTrue(trainer.sgs.warmed_up)
                        self.assertGreater(trainer._optimizer_updates, 0)
                        self.assertTrue(all(torch.isfinite(p).all() for p in trainer.agent.model.parameters()))
                        if mode == "replay":
                            self.assertEqual(len(set(trainer.sgs.collected.values())), 1)
                        if mode == "collection":
                            self.assertEqual(len(set(trainer.sgs.replayed.values())), 1)
                        if backend == "mujoco" and mode == "both":
                            checkpoint = trainer.save_checkpoint()
                            cfg.resume_from_checkpoint = str(checkpoint)
                            resumed_env = make_env(cfg)
                            resumed = OnlineTrainer(cfg=cfg, env=resumed_env, agent=GNNSAC(cfg),
                                                    buffer=make_gnn_buffer(cfg), logger=Logger(cfg))
                            try:
                                self.assertEqual(resumed._step, trainer._step)
                                self.assertEqual(resumed.sgs.collected, trainer.sgs.collected)
                                self.assertEqual(resumed.sgs.state_dict()["histories"], trainer.sgs.state_dict()["histories"])
                                self.assertEqual(resumed.sgs.replay_counts(cfg.batch_size, cfg.batch_size // 2),
                                                 trainer.sgs.replay_counts(cfg.batch_size, cfg.batch_size // 2))
                                self.assertEqual(resumed.sgs.collection_group(resumed.sgs.tasks, [1, 1]),
                                                 trainer.sgs.collection_group(trainer.sgs.tasks, [1, 1]))
                            finally:
                                resumed.eval_env.close()
                                resumed_env.close()
                    finally:
                        env.close()


if __name__ == "__main__":
    unittest.main()
