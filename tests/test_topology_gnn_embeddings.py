"""Tests for figures/topology_gnn_embeddings.py.

The smoke tests save a freshly-initialized (untrained) agent and load it back
through the same resolve_checkpoint/load_agent_checkpoint path the real
script uses, so no trained checkpoint is needed.
"""

import tempfile
import textwrap
import unittest
import warnings
from pathlib import Path

import numpy as np

from env import make_env
from figures.topology_gnn_embeddings import (
    CheckpointEntry,
    _build_cfg,
    available_entries,
    embed_topology,
    embed_topology_resets,
    load_fold_embeddings,
    load_manifest,
    reset_noise_stats,
    save_fold_embeddings,
    validate_training_config,
)
from gnn_infer import _make_agent


def _save_untrained_agent(topology: str, path: Path) -> None:
    cfg = _build_cfg("placeholder", topology, "cpu", seed=0)
    env = make_env(cfg)
    try:
        agent = _make_agent(cfg)
    finally:
        env.close()
    agent.save(path)


class EmbeddingSmokeTest(unittest.TestCase):
    def test_extracts_a_finite_fixed_size_pooled_embedding(self):
        topology = "tetrahedron"
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "untrained.pt"
            _save_untrained_agent(topology, checkpoint_path)
            embedding = embed_topology(str(checkpoint_path), topology, device="cpu", seed=0)
            resets = embed_topology_resets(
                str(checkpoint_path), topology, device="cpu", seed=0, n_resets=3
            )

        self.assertEqual(embedding.ndim, 1)
        self.assertGreater(embedding.shape[0], 0)
        self.assertTrue(np.all(np.isfinite(embedding)))
        self.assertEqual(resets.shape, (3, embedding.shape[0]))
        self.assertTrue(np.all(np.isfinite(resets)))

    def test_embedding_is_similar_across_reruns_for_a_fixed_seed(self):
        # Not bit-exact: MuJoCo's reset touches RNG state that set_seed() does
        # not fully pin down across separate make_env() calls, so this only
        # checks the two runs land close together, not identically.
        topology = "tetrahedron"
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "untrained.pt"
            _save_untrained_agent(topology, checkpoint_path)
            first = embed_topology(str(checkpoint_path), topology, device="cpu", seed=0)
            second = embed_topology(str(checkpoint_path), topology, device="cpu", seed=0)

        np.testing.assert_allclose(first, second, atol=0.05)


class ManifestTest(unittest.TestCase):
    def _write(self, tmp_dir: Path, text: str) -> Path:
        path = tmp_dir / "manifest.yaml"
        path.write_text(textwrap.dedent(text))
        return path

    def test_parses_entries_and_resolves_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            manifest = load_manifest(
                self._write(tmp_dir, """
                    folds:
                      node_4:
                        - seed: 2
                          wandb_run: i-suds/CV/abc
                          checkpoint: ckpts/a.pt
                          training_config: /abs/config.yaml
                      node_6:
                        - seed: 1
                          wandb_run: i-suds/CV/def
                          checkpoint: null
                      node_8: []
                """),
                base_dir=tmp_dir,
            )

        self.assertEqual(sorted(manifest), ["node_4", "node_6", "node_8"])
        entry = manifest["node_4"][0]
        self.assertEqual(entry.fold, "node_4")
        self.assertEqual(entry.seed, 2)
        self.assertEqual(entry.wandb_run, "i-suds/CV/abc")
        self.assertEqual(entry.checkpoint, tmp_dir / "ckpts" / "a.pt")
        self.assertEqual(entry.training_config, Path("/abs/config.yaml"))
        self.assertIsNone(manifest["node_6"][0].checkpoint)
        self.assertIsNone(manifest["node_6"][0].training_config)
        self.assertEqual(manifest["node_8"], [])

    def test_rejects_duplicate_seeds_and_missing_folds_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            with self.assertRaises(ValueError):
                load_manifest(self._write(tmp_dir, """
                    folds:
                      node_4:
                        - {seed: 1, checkpoint: a.pt}
                        - {seed: 1, checkpoint: b.pt}
                """), base_dir=tmp_dir)
            with self.assertRaises(ValueError):
                load_manifest(self._write(tmp_dir, "node_4: []\n"), base_dir=tmp_dir)

    def test_repo_manifest_parses_and_matches_catalog_folds(self):
        import _topology_catalog

        manifest = load_manifest(Path(__file__).resolve().parents[1] / "figures" / "cv_base_checkpoints.yaml")
        self.assertEqual(sorted(manifest), sorted(_topology_catalog.groups()))
        for fold, entries in manifest.items():
            for entry in entries:
                self.assertTrue(entry.wandb_run.startswith("i-suds/CV/"), entry)

    def test_missing_folds_are_skipped_not_borrowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            present = tmp_dir / "present.pt"
            present.write_bytes(b"")
            manifest = {
                "node_4": [CheckpointEntry("node_4", 2, "e/p/a", present)],
                "node_5": [
                    CheckpointEntry("node_5", 1, "e/p/b", present),
                    CheckpointEntry("node_5", 2, "e/p/c", tmp_dir / "absent.pt"),
                ],
                "node_6": [CheckpointEntry("node_6", 1, "e/p/d", None)],
                "node_8": [],
            }
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                usable = available_entries(manifest)

        self.assertEqual(sorted(usable), ["node_4", "node_5"])
        self.assertEqual([e.seed for e in usable["node_5"]], [1])
        # Every usable entry still belongs to its own fold.
        for fold, entries in usable.items():
            self.assertTrue(all(e.fold == fold for e in entries))
        messages = " ".join(str(w.message) for w in caught)
        self.assertIn("node_6", messages)
        self.assertIn("node_8", messages)
        self.assertIn("absent.pt", messages)


class ValidateTrainingConfigTest(unittest.TestCase):
    def _config(self, held_out: str, seed: int = 2) -> dict:
        import _topology_catalog

        return {
            "seed": seed,
            "cross_validation": {
                "held_out_group": held_out,
                "training_topologies": _topology_catalog.fold_membership()[held_out]["train"],
            },
        }

    def test_accepts_matching_run(self):
        entry = CheckpointEntry("node_4", 2, "e/p/r", Path("CV-BASE_seed2_node4heldout_latest.agent.pt"))
        validate_training_config(entry, self._config("node_4"))

    def test_rejects_run_from_another_fold(self):
        entry = CheckpointEntry("node_4", 2, "e/p/r", Path("x.pt"))
        with self.assertRaises(ValueError):
            validate_training_config(entry, self._config("node_8"))

    def test_rejects_seed_or_filename_mismatch(self):
        with self.assertRaises(ValueError):
            validate_training_config(CheckpointEntry("node_4", 1, "e/p/r", Path("x.pt")), self._config("node_4"))
        with self.assertRaises(ValueError):
            validate_training_config(
                CheckpointEntry("node_4", 2, "e/p/r", Path("CV-BASE_seed2_node9heldout_latest.agent.pt")),
                self._config("node_4"),
            )


class FoldEmbeddingsIOTest(unittest.TestCase):
    def test_round_trips_in_requested_topology_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            save_fold_embeddings(
                tmp_dir / "node_4.npz",
                {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0]), "seed2/a": np.zeros(2)},
                {"fold": "node_4"},
            )
            save_fold_embeddings(
                tmp_dir / "node_5.npz", {"a": np.zeros(2), "b": np.ones(2)}, {"fold": "node_5"}
            )
            loaded = load_fold_embeddings(tmp_dir, ["b", "a"])

        self.assertEqual(sorted(loaded), ["node_4", "node_5"])
        np.testing.assert_array_equal(loaded["node_4"], [[3.0, 4.0], [1.0, 2.0]])

    def test_raises_on_missing_topologies(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            save_fold_embeddings(tmp_dir / "node_4.npz", {"a": np.zeros(2)}, {})
            with self.assertRaises(KeyError):
                load_fold_embeddings(tmp_dir, ["a", "b"])

    def test_raises_on_missing_requested_fold(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            save_fold_embeddings(tmp_dir / "node_4.npz", {"a": np.zeros(2)}, {})
            with self.assertRaises(FileNotFoundError):
                load_fold_embeddings(tmp_dir, ["a"], folds=["node_4", "node_8"])


class ResetNoiseStatsTest(unittest.TestCase):
    def test_reports_spread_relative_to_between_topology_distance(self):
        per_reset = {
            "a": np.array([[0.0, 0.0], [0.0, 0.2]]),
            "b": np.array([[3.0, 4.0], [3.0, 4.0]]),
        }
        stats = reset_noise_stats(per_reset)
        self.assertAlmostEqual(stats["reset_spread_max"], 0.1)
        self.assertEqual(stats["reset_spread_max_topology"], "a")
        self.assertAlmostEqual(stats["between_topology_min"], np.hypot(3.0, 3.9))


if __name__ == "__main__":
    unittest.main()
