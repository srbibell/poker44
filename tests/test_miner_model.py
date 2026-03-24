from __future__ import annotations

import unittest

import numpy as np

from poker44.miner.model import MinerRiskModel

from tests.test_miner_features import make_hand


class MinerModelTests(unittest.TestCase):
    def test_remap_scores_maps_threshold_to_half(self) -> None:
        probabilities = np.asarray([0.0, 0.2, 0.5, 1.0], dtype=np.float32)
        remapped = MinerRiskModel._remap_scores(probabilities, 0.2)

        self.assertAlmostEqual(float(remapped[0]), 0.0)
        self.assertAlmostEqual(float(remapped[1]), 0.5)
        self.assertAlmostEqual(float(remapped[-1]), 1.0)
        self.assertTrue(np.all(np.diff(remapped) >= 0.0))

    def test_fallback_heuristic_scores_bot_like_chunk_higher(self) -> None:
        human_chunk = [
            make_hand(
                player_count=6,
                street_count=1,
                action_types=["small_blind", "big_blind", "fold", "check", "fold"],
                amount_bb=0.5,
            )
            for _ in range(3)
        ]
        bot_chunk = [
            make_hand(
                player_count=3,
                street_count=3,
                action_types=["small_blind", "big_blind", "call", "raise", "bet"],
                amount_bb=5.0,
            )
            for _ in range(3)
        ]

        human_score = MinerRiskModel.fallback_score_chunk(human_chunk)
        bot_score = MinerRiskModel.fallback_score_chunk(bot_chunk)

        self.assertGreaterEqual(human_score, 0.0)
        self.assertLessEqual(human_score, 1.0)
        self.assertGreaterEqual(bot_score, 0.0)
        self.assertLessEqual(bot_score, 1.0)
        self.assertGreater(bot_score, human_score)


if __name__ == "__main__":
    unittest.main()
