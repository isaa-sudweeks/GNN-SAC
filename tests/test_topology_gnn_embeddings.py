"""Verify the embedding-extraction plumbing without needing a real trained
checkpoint: save a freshly-initialized (untrained) agent and load it back
through the same resolve_checkpoint/load_agent_checkpoint path the real
script uses."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from env import make_env
from figures.topology_gnn_embeddings import _build_cfg, embed_topology
from gnn_infer import _make_agent


class TopologyGnnEmbeddingsTest(unittest.TestCase):
    def test_extracts_a_finite_fixed_size_pooled_embedding(self):
        topology = "tetrahedron"
        cfg = _build_cfg("placeholder", topology, "cpu", seed=0)
        env = make_env(cfg)
        try:
            agent = _make_agent(cfg)
        finally:
            env.close()

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "untrained.pt"
            agent.save(checkpoint_path)
            embedding = embed_topology(str(checkpoint_path), topology, device="cpu", seed=0)

        self.assertEqual(embedding.ndim, 1)
        self.assertGreater(embedding.shape[0], 0)
        self.assertTrue(np.all(np.isfinite(embedding)))

    def test_embedding_is_similar_across_reruns_for_a_fixed_seed(self):
        # Not bit-exact: MuJoCo's reset touches RNG state that set_seed() does
        # not fully pin down across separate make_env() calls, so this only
        # checks the two runs land close together, not identically.
        topology = "tetrahedron"
        cfg = _build_cfg("placeholder", topology, "cpu", seed=0)
        env = make_env(cfg)
        try:
            agent = _make_agent(cfg)
        finally:
            env.close()

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "untrained.pt"
            agent.save(checkpoint_path)
            first = embed_topology(str(checkpoint_path), topology, device="cpu", seed=0)
            second = embed_topology(str(checkpoint_path), topology, device="cpu", seed=0)

        np.testing.assert_allclose(first, second, atol=0.05)


if __name__ == "__main__":
    unittest.main()
