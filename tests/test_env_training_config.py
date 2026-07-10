from types import SimpleNamespace
import unittest

from env import _configure_episode_length


class EnvironmentTrainingConfigTest(unittest.TestCase):
    def test_explicit_seed_steps_survive_environment_construction(self):
        cfg = SimpleNamespace(seed_steps=4)

        _configure_episode_length(cfg, 10)

        self.assertEqual(cfg.episode_length, 10)
        self.assertEqual(cfg.seed_steps, 4)

    def test_missing_seed_steps_uses_horizon_based_default(self):
        cfg = SimpleNamespace(seed_steps=None)

        _configure_episode_length(cfg, 300)

        self.assertEqual(cfg.episode_length, 300)
        self.assertEqual(cfg.seed_steps, 1500)


if __name__ == "__main__":
    unittest.main()
