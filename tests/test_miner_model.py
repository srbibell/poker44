from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from poker44.miner.model import (
    DEFAULT_HUMAN_JSON_PATH,
    MinerRiskModel,
    MinerTrainingConfig,
    _normalize_startup_mode,
)

from tests.test_miner_features import make_hand


class MinerModelTests(unittest.TestCase):
    def test_startup_mode_normalization(self) -> None:
        self.assertEqual(_normalize_startup_mode("background"), "background")
        self.assertEqual(_normalize_startup_mode("BACKGROUND"), "background")
        self.assertEqual(_normalize_startup_mode("unknown"), "blocking")

    def test_training_config_startup_mode_defaults_to_blocking(self) -> None:
        with patch.dict(os.environ, {"POKER44_MINER_STARTUP_MODE": ""}):
            cfg = MinerTrainingConfig.from_env(cache_dir=Path(".cache/test_model"))
        self.assertEqual(cfg.startup_mode, "blocking")

    def test_training_config_reads_background_startup_mode(self) -> None:
        with patch.dict(
            os.environ, {"POKER44_MINER_STARTUP_MODE": "background"}
        ):
            cfg = MinerTrainingConfig.from_env(cache_dir=Path(".cache/test_model"))
        self.assertEqual(cfg.startup_mode, "background")

    def test_cache_fingerprint_ignores_runtime_startup_flags(self) -> None:
        common_kwargs = dict(
            human_json_path=DEFAULT_HUMAN_JSON_PATH,
            cache_dir=Path(".cache/test_model"),
            train_window_count=6,
            validation_window_count=2,
            chunk_count=40,
            min_hands_per_chunk=60,
            max_hands_per_chunk=120,
            human_ratio=0.5,
            refresh_seconds=3600,
            seed=123,
            bot_candidate_attempts_per_chunk=4,
            max_bot_generation_rounds=2,
        )
        blocking_cfg = MinerTrainingConfig(
            **common_kwargs,
            force_retrain=False,
            startup_mode="blocking",
        )
        background_cfg = MinerTrainingConfig(
            **common_kwargs,
            force_retrain=True,
            startup_mode="background",
        )

        self.assertEqual(
            MinerRiskModel._cache_path_for_config(blocking_cfg),
            MinerRiskModel._cache_path_for_config(background_cfg),
        )

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
