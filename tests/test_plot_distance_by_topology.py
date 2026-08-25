import unittest
from types import SimpleNamespace

from figures.plot_distance_by_topology import (
    TOPOLOGIES,
    aggregate_results,
    collect_seed_results,
    load_greedy_results,
    make_figure,
)


def _run(run_id, seed, distance, step):
    topology = "henneberg_n5_1tube_1"
    return SimpleNamespace(
        id=run_id,
        name=f"paper-v2-{topology}-{seed}",
        state="finished",
        url=f"https://wandb.ai/run/{run_id}",
        config={"truss_topology": topology, "seed": seed},
        summary={"eval/episode_distance": distance, "_step": step},
    )


class DistanceByTopologyTest(unittest.TestCase):
    def test_retry_with_largest_logged_step_wins(self):
        rows = collect_seed_results(
            [_run("old", 1, 1.0, 100), _run("new", 1, 2.0, 200)], "paper-v2-"
        )
        self.assertEqual(rows.iloc[0]["run_id"], "new")
        self.assertEqual(rows.iloc[0]["distance_m"], 2.0)

    def test_aggregate_uses_sample_standard_deviation(self):
        rows = collect_seed_results(
            [_run("one", 1, 1.0, 100), _run("two", 2, 3.0, 100)], "paper-v2-"
        )
        summary = aggregate_results(rows)
        self.assertEqual(summary.iloc[0]["distance_m"], 2.0)
        self.assertAlmostEqual(summary.iloc[0]["distance_std_m"], 2**0.5)
        self.assertEqual(summary.iloc[0]["n_seeds"], 2)

    def test_greedy_measurements_include_per_topology_error_bars(self):
        rows = collect_seed_results([_run("one", 1, 2.0, 100)], "paper-v2-")
        figure = make_figure(aggregate_results(rows))
        greedy = next(trace for trace in figure.data if trace.name == "Greedy")

        metrics = load_greedy_results()
        self.assertEqual(list(greedy.x), list(metrics["distance_m"]))
        self.assertEqual(list(greedy.error_x.array), list(metrics["distance_std_m"]))
        self.assertAlmostEqual(greedy.x[0], 7.4338)
        self.assertAlmostEqual(greedy.x[-1], 8.50808597868836)
        self.assertTrue(all(value == value for value in greedy.error_x.array))


if __name__ == "__main__":
    unittest.main()
