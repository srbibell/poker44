from __future__ import annotations

import unittest
from types import SimpleNamespace

from poker44.validator.forward import _compute_windowed_rewards, _select_weight_targets


class ValidatorForwardTests(unittest.TestCase):
    def test_compute_windowed_rewards_requires_full_window(self) -> None:
        validator = SimpleNamespace(
            reward_window=3,
            prediction_buffer={5: [0.1, 0.9]},
            label_buffer={5: [0, 1]},
        )
        rewards, metrics = _compute_windowed_rewards(validator, [5])
        self.assertEqual(rewards.tolist(), [0.0])
        self.assertEqual(metrics[0]["reward"], 0.0)

    def test_compute_windowed_rewards_scores_full_window(self) -> None:
        validator = SimpleNamespace(
            reward_window=2,
            prediction_buffer={5: [0.0, 1.0]},
            label_buffer={5: [0, 1]},
        )
        rewards, metrics = _compute_windowed_rewards(validator, [5])
        self.assertGreater(float(rewards[0]), 0.0)
        self.assertGreater(metrics[0]["reward"], 0.0)

    def test_select_weight_targets_empty_reward_map_falls_back_to_uid_zero(self) -> None:
        uids, rewards = _select_weight_targets({})
        self.assertEqual(uids, [0])
        self.assertAlmostEqual(float(rewards[0]), 1.0)

    def test_select_weight_targets_prefers_higher_reward(self) -> None:
        uids, rewards = _select_weight_targets({7: 0.8, 8: 0.2})
        self.assertAlmostEqual(float(sum(rewards)), 1.0, places=6)
        self.assertIn(0, uids)
        self.assertIn(7, uids)
        self.assertIn(8, uids)
        reward_by_uid = dict(zip(uids, rewards.tolist()))
        self.assertGreater(reward_by_uid[7], reward_by_uid[8])


if __name__ == "__main__":
    unittest.main()

