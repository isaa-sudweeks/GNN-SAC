from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gnn_sac import GNNSAC
from sac import SAC


def agent_stub(agent_cls, **overrides):
    values = dict(
        episode_length=1000,
        discount_denom=500,
        discount_min=0.95,
        discount_max=0.995,
    )
    values.update(overrides)
    stub = SimpleNamespace(cfg=SimpleNamespace(**values))
    stub._get_discount = lambda episode_length: agent_cls._get_discount(stub, episode_length)
    return stub


class DiscountTest(unittest.TestCase):
    def test_algorithm_config_defaults_to_0_995(self):
        algorithm_cfg = OmegaConf.load(ROOT / "config" / "algorithm.yaml")
        self.assertEqual(algorithm_cfg.discount, 0.995)

    def test_explicit_discount_overrides_heuristic(self):
        for agent_cls in (SAC, GNNSAC):
            with self.subTest(agent=agent_cls.__name__):
                stub = agent_stub(agent_cls, discount=0.995)
                self.assertEqual(agent_cls._resolve_discount(stub), 0.995)

    def test_null_discount_uses_episode_length_heuristic(self):
        for agent_cls in (SAC, GNNSAC):
            with self.subTest(agent=agent_cls.__name__):
                short = agent_stub(agent_cls, discount=None, episode_length=1000)
                self.assertAlmostEqual(agent_cls._resolve_discount(short), 0.95)
                long = agent_stub(agent_cls, discount=None, episode_length=50_000)
                self.assertAlmostEqual(agent_cls._resolve_discount(long), 0.99)

    def test_missing_discount_uses_episode_length_heuristic(self):
        for agent_cls in (SAC, GNNSAC):
            with self.subTest(agent=agent_cls.__name__):
                stub = agent_stub(agent_cls, episode_length=1000)
                self.assertAlmostEqual(agent_cls._resolve_discount(stub), 0.95)


if __name__ == "__main__":
    unittest.main()
