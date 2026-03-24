#!/usr/bin/env python3
"""Local benchmark for the Poker44 miner model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hands_generator.mixed_dataset_provider import (
    MixedDatasetConfig,
    build_mixed_labeled_chunks,
)
from poker44.miner.model import MinerRiskModel
from poker44.score.scoring import reward
from poker44.validator.sanitization import sanitize_hand_for_miner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the local miner model against generated validator windows."
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=3,
        help="How many evaluation windows to score.",
    )
    parser.add_argument(
        "--start-window",
        type=int,
        default=0,
        help="First evaluation window. Defaults to the first unseen window after training+validation windows.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/miner_eval"),
        help="Directory for cached miner model artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = MinerRiskModel(cache_dir=args.cache_dir)

    default_start = (
        model.training_cfg.train_window_count
        + model.training_cfg.validation_window_count
        + 1
    )
    start_window = args.start_window or default_start

    cfg = MixedDatasetConfig(
        human_json_path=model.training_cfg.human_json_path,
        output_path=args.cache_dir / "evaluation_dataset.json",
        chunk_count=model.training_cfg.chunk_count,
        min_hands_per_chunk=model.training_cfg.min_hands_per_chunk,
        max_hands_per_chunk=model.training_cfg.max_hands_per_chunk,
        human_ratio=model.training_cfg.human_ratio,
        refresh_seconds=model.training_cfg.refresh_seconds,
        seed=model.training_cfg.seed,
        bot_candidate_attempts_per_chunk=(
            model.training_cfg.bot_candidate_attempts_per_chunk
        ),
        max_bot_generation_rounds=model.training_cfg.max_bot_generation_rounds,
    )

    rows: list[dict[str, float | int]] = []
    for window_id in range(start_window, start_window + args.windows):
        chunks, _, _ = build_mixed_labeled_chunks(cfg, window_id=window_id)
        sanitized_chunks = [
            [sanitize_hand_for_miner(hand) for hand in chunk.get("hands") or []]
            for chunk in chunks
        ]
        labels = np.asarray(
            [1 if chunk.get("is_bot", False) else 0 for chunk in chunks],
            dtype=bool,
        )
        scores = np.asarray(model.score_chunks(sanitized_chunks), dtype=float)
        reward_value, metrics = reward(scores, labels)
        rows.append(
            {
                "window_id": window_id,
                "reward": round(float(reward_value), 6),
                "fpr": round(float(metrics["fpr"]), 6),
                "bot_recall": round(float(metrics["bot_recall"]), 6),
                "ap_score": round(float(metrics["ap_score"]), 6),
            }
        )

    average_reward = (
        sum(float(row["reward"]) for row in rows) / len(rows) if rows else 0.0
    )
    payload = {
        "model_threshold": round(model.threshold, 6),
        "model_metrics": model.metrics,
        "evaluation_windows": rows,
        "average_reward": round(average_reward, 6),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
