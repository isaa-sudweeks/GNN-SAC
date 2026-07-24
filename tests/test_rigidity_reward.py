import unittest

import numpy as np

from env.mujoco_gen.rigidity_reward import danger_zone_rigidity_penalty


class DangerZoneRigidityPenaltyTest(unittest.TestCase):
    def test_is_zero_at_and_above_safe_threshold(self):
        values = danger_zone_rigidity_penalty(
            np.asarray([0.3, 0.4]),
            collapse_threshold=0.1,
            safe_threshold=0.3,
            weight=2.0,
            power=2.0,
            epsilon=1e-8,
            max_penalty=5.0,
        )
        np.testing.assert_array_equal(values, np.zeros(2))

    def test_grows_toward_collapse_and_is_bounded(self):
        values = danger_zone_rigidity_penalty(
            np.asarray([0.25, 0.15, 0.100001]),
            collapse_threshold=0.1,
            safe_threshold=0.3,
            weight=2.0,
            power=2.0,
            epsilon=1e-8,
            max_penalty=5.0,
        )
        self.assertGreater(values[0], values[1])
        self.assertEqual(values[1], -5.0)
        self.assertEqual(values[2], -5.0)

    def test_rejects_invalid_threshold_order(self):
        with self.assertRaisesRegex(ValueError, "must exceed"):
            danger_zone_rigidity_penalty(
                0.1,
                collapse_threshold=0.1,
                safe_threshold=0.1,
                weight=1.0,
                power=2.0,
                epsilon=1e-8,
                max_penalty=5.0,
            )


if __name__ == "__main__":
    unittest.main()
