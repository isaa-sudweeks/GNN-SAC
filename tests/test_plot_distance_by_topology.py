import unittest
from types import SimpleNamespace

from figures.plot_distance_by_topology import (
    TOPOLOGIES,
    aggregate_results,
    collect_seed_results,
    load_greedy_results,
    make_figure,
)


def _run(run_id, seed, history):
    topology = "henneberg_n5_1tube_1"
    run = SimpleNamespace(
        id=run_id,
        name=f"paper-v2-{topology}-{seed}",
        state="finished",
        url=f"https://wandb.ai/run/{run_id}",
        config={"truss_topology": topology, "seed": seed},
        summary={"_step": max(step for step, _ in history)},
    )
    run.history = lambda keys, samples, pandas: [
        {"_step": step, "eval/episode_distance": distance}
        for step, distance in history
    ]
    return run


class DistanceByTopologyTest(unittest.TestCase):
    def test_retry_with_largest_logged_step_wins(self):
        rows = collect_seed_results(
            [_run("old", 1, [(100, 1.0)]), _run("new", 1, [(200, 2.0)])],
            "paper-v2-",
        )
        self.assertEqual(rows.iloc[0]["run_id"], "new")
        self.assertEqual(rows.iloc[0]["distance_m"], 2.0)

    def test_aggregate_uses_sample_standard_deviation(self):
        rows = collect_seed_results(
            [_run("one", 1, [(100, 1.0)]), _run("two", 2, [(100, 3.0)])],
            "paper-v2-",
        )
        summary = aggregate_results(rows)
        self.assertEqual(summary.iloc[0]["distance_m"], 2.0)
        self.assertAlmostEqual(summary.iloc[0]["distance_std_m"], 2**0.5)
        self.assertEqual(summary.iloc[0]["n_seeds"], 2)

    def test_selects_common_step_with_highest_across_seed_mean(self):
        rows = collect_seed_results(
            [
                _run("one", 1, [(100, 9.0), (200, 6.0)]),
                _run("two", 2, [(100, 1.0), (200, 6.0)]),
                _run("three", 3, [(100, 1.0), (200, 6.0)]),
            ],
            "paper-v2-",
        )

        self.assertEqual(set(rows["evaluation_step"]), {200})
        self.assertEqual(list(rows.sort_values("seed")["distance_m"]), [6.0, 6.0, 6.0])
        self.assertEqual(set(rows["selection_mean_distance_m"]), {6.0})

    def test_does_not_mix_steps_that_are_missing_for_a_seed(self):
        rows = collect_seed_results(
            [
                _run("one", 1, [(100, 2.0), (200, 100.0)]),
                _run("two", 2, [(100, 4.0)]),
            ],
            "paper-v2-",
        )

        self.assertEqual(set(rows["evaluation_step"]), {100})

    def test_greedy_measurements_include_per_topology_error_bars(self):
        rows = collect_seed_results([_run("one", 1, [(100, 2.0)])], "paper-v2-")
        figure = make_figure(aggregate_results(rows))
        greedy = next(trace for trace in figure.data if trace.name == "Greedy")

        metrics = load_greedy_results()
        self.assertEqual(figure.layout.title.text, "Peak Mean Evaluation Distance by Topology")
        self.assertEqual(list(greedy.x), list(metrics["distance_m"]))
        self.assertEqual(list(greedy.error_x.array), list(metrics["distance_std_m"]))
        self.assertAlmostEqual(greedy.x[0], 7.4338)
        self.assertAlmostEqual(greedy.x[-1], 8.50808597868836)
        self.assertTrue(all(value == value for value in greedy.error_x.array))


if __name__ == "__main__":
    unittest.main()
