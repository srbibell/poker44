"""Trainable miner-side scoring model with local caching."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import bittensor as bt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from hands_generator.mixed_dataset_provider import (
    MixedDatasetConfig,
    build_mixed_labeled_chunks,
)
from poker44 import __version__
from poker44.miner.features import FEATURE_VERSION, extract_chunk_features
from poker44.score.scoring import reward
from poker44.validator.sanitization import sanitize_hand_for_miner

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HUMAN_JSON_PATH = (
    REPO_ROOT / "hands_generator" / "human_hands" / "poker_hands_combined.json.gz"
)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class MinerTrainingConfig:
    human_json_path: Path
    cache_dir: Path
    train_window_count: int = 6
    validation_window_count: int = 2
    chunk_count: int = 40
    min_hands_per_chunk: int = 60
    max_hands_per_chunk: int = 120
    human_ratio: float = 0.5
    refresh_seconds: int = 60 * 60
    seed: int = 123
    bot_candidate_attempts_per_chunk: int = 4
    max_bot_generation_rounds: int = 2
    force_retrain: bool = False

    @classmethod
    def from_env(cls, *, cache_dir: Path | None = None) -> "MinerTrainingConfig":
        human_json_path = Path(
            os.getenv("POKER44_MINER_HUMAN_JSON_PATH", str(DEFAULT_HUMAN_JSON_PATH))
        ).expanduser()
        resolved_cache_dir = (
            cache_dir
            if cache_dir is not None
            else Path(
                os.getenv(
                    "POKER44_MINER_MODEL_CACHE_DIR",
                    str(REPO_ROOT / ".cache" / "miner_model"),
                )
            ).expanduser()
        )
        return cls(
            human_json_path=human_json_path,
            cache_dir=resolved_cache_dir,
            train_window_count=max(2, _env_int("POKER44_MINER_TRAIN_WINDOWS", 6)),
            validation_window_count=max(
                1, _env_int("POKER44_MINER_VALIDATION_WINDOWS", 2)
            ),
            chunk_count=max(8, _env_int("POKER44_MINER_CHUNK_COUNT", 40)),
            min_hands_per_chunk=max(
                10, _env_int("POKER44_MINER_MIN_HANDS_PER_CHUNK", 60)
            ),
            max_hands_per_chunk=max(
                _env_int("POKER44_MINER_MIN_HANDS_PER_CHUNK", 60),
                _env_int("POKER44_MINER_MAX_HANDS_PER_CHUNK", 120),
            ),
            human_ratio=min(
                0.9, max(0.1, _env_float("POKER44_MINER_HUMAN_RATIO", 0.5))
            ),
            refresh_seconds=max(
                60, _env_int("POKER44_MINER_REFRESH_SECONDS", 60 * 60)
            ),
            seed=_env_int("POKER44_MINER_TRAINING_SEED", 123),
            bot_candidate_attempts_per_chunk=max(
                1,
                _env_int("POKER44_MINER_BOT_CANDIDATE_ATTEMPTS", 4),
            ),
            max_bot_generation_rounds=max(
                1,
                _env_int("POKER44_MINER_MAX_BOT_GENERATION_ROUNDS", 2),
            ),
            force_retrain=_env_bool("POKER44_MINER_FORCE_RETRAIN", False),
        )


class MinerRiskModel:
    """Cached local model that scores validator-sanitized hand chunks."""

    def __init__(self, *, cache_dir: Path | None = None):
        self.training_cfg = MinerTrainingConfig.from_env(cache_dir=cache_dir)
        self.model: HistGradientBoostingClassifier | None = None
        self.threshold: float = 0.5
        self.metrics: dict[str, Any] = {}
        self.cache_path = self._cache_path_for_config(self.training_cfg)
        self._load_or_train()

    def score_chunks(self, chunks: Sequence[Sequence[dict]]) -> list[float]:
        if not chunks:
            return []

        if self.model is None:
            return [self.fallback_score_chunk(chunk) for chunk in chunks]

        try:
            feature_matrix = np.vstack(
                [extract_chunk_features(chunk) for chunk in chunks]
            )
            probabilities = self.model.predict_proba(feature_matrix)[:, 1]
            adjusted = self._remap_scores(probabilities, self.threshold)
            return [round(float(score), 6) for score in adjusted]
        except Exception as err:
            bt.logging.warning(
                f"Miner model inference failed, using heuristic fallback: {err}"
            )
            return [self.fallback_score_chunk(chunk) for chunk in chunks]

    def score_chunk(self, chunk: Sequence[dict]) -> float:
        scores = self.score_chunks([chunk])
        return scores[0] if scores else 0.5

    def _load_or_train(self) -> None:
        if not self.training_cfg.force_retrain and self.cache_path.exists():
            try:
                payload = pickle.loads(self.cache_path.read_bytes())
                self.model = payload["model"]
                self.threshold = float(payload["threshold"])
                self.metrics = dict(payload.get("metrics") or {})
                bt.logging.info(
                    f"Loaded cached miner model from {self.cache_path} | threshold={self.threshold:.3f}"
                )
                return
            except Exception as err:
                bt.logging.warning(
                    f"Failed to load cached miner model, retraining: {err}"
                )

        try:
            self._train_and_cache()
        except Exception as err:
            bt.logging.warning(
                f"Miner model training failed, keeping heuristic fallback only: {err}"
            )
            self.model = None
            self.threshold = 0.5
            self.metrics = {"status": "fallback_only", "error": str(err)}

    def _train_and_cache(self) -> None:
        if not self.training_cfg.human_json_path.exists():
            raise FileNotFoundError(
                f"Missing miner human dataset: {self.training_cfg.human_json_path}"
            )

        start = time.time()
        train_window_ids = list(range(1, self.training_cfg.train_window_count + 1))
        validation_start = train_window_ids[-1] + 1
        validation_window_ids = list(
            range(
                validation_start,
                validation_start + self.training_cfg.validation_window_count,
            )
        )

        bt.logging.info(
            "Training local miner model "
            f"| train_windows={train_window_ids} validation_windows={validation_window_ids}"
        )
        train_x, train_y = self._build_dataset(train_window_ids)
        validation_x, validation_y = self._build_dataset(validation_window_ids)

        model = HistGradientBoostingClassifier(
            max_depth=5,
            learning_rate=0.05,
            max_iter=350,
            min_samples_leaf=8,
            l2_regularization=0.05,
            random_state=self.training_cfg.seed,
        )
        model.fit(train_x, train_y)

        validation_probs = model.predict_proba(validation_x)[:, 1]
        threshold, _ = self._select_threshold(validation_probs, validation_y)

        train_scores = self._remap_scores(model.predict_proba(train_x)[:, 1], threshold)
        train_reward, train_metrics = reward(train_scores, train_y)
        validation_reward, validation_summary = reward(
            self._remap_scores(validation_probs, threshold),
            validation_y,
        )

        self.training_cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "bundle_version": 1,
            "model": model,
            "threshold": threshold,
            "metrics": {
                "feature_version": FEATURE_VERSION,
                "train_reward": train_reward,
                "train_metrics": train_metrics,
                "validation_reward": validation_reward,
                "validation_metrics": validation_summary,
                "selected_threshold": threshold,
                "train_examples": int(train_y.size),
                "validation_examples": int(validation_y.size),
                "train_window_ids": train_window_ids,
                "validation_window_ids": validation_window_ids,
                "training_seconds": round(time.time() - start, 2),
                "package_version": __version__,
            },
        }
        self.cache_path.write_bytes(pickle.dumps(payload))

        self.model = model
        self.threshold = threshold
        self.metrics = dict(payload["metrics"])
        bt.logging.info(
            "Finished miner model training "
            f"| reward={validation_reward:.4f} threshold={threshold:.3f} "
            f"cache={self.cache_path}"
        )

    def _build_dataset(
        self, window_ids: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = MixedDatasetConfig(
            human_json_path=self.training_cfg.human_json_path,
            output_path=self.training_cfg.cache_dir / "training_dataset.json",
            chunk_count=self.training_cfg.chunk_count,
            min_hands_per_chunk=self.training_cfg.min_hands_per_chunk,
            max_hands_per_chunk=self.training_cfg.max_hands_per_chunk,
            human_ratio=self.training_cfg.human_ratio,
            refresh_seconds=self.training_cfg.refresh_seconds,
            seed=self.training_cfg.seed,
            bot_candidate_attempts_per_chunk=(
                self.training_cfg.bot_candidate_attempts_per_chunk
            ),
            max_bot_generation_rounds=self.training_cfg.max_bot_generation_rounds,
        )

        rows: list[np.ndarray] = []
        labels: list[bool] = []
        for window_id in window_ids:
            bt.logging.info(f"Building miner training window {window_id}")
            chunks, _, _ = build_mixed_labeled_chunks(cfg, window_id=window_id)
            for chunk in chunks:
                sanitized_chunk = [
                    sanitize_hand_for_miner(hand)
                    for hand in (chunk.get("hands") or [])
                    if isinstance(hand, dict)
                ]
                rows.append(extract_chunk_features(sanitized_chunk))
                labels.append(bool(chunk.get("is_bot", False)))

        if not rows:
            raise RuntimeError("Generated no training chunks for miner model")

        return np.vstack(rows), np.asarray(labels, dtype=bool)

    def _select_threshold(
        self, probabilities: np.ndarray, labels: np.ndarray
    ) -> tuple[float, dict[str, Any]]:
        best_threshold = 0.5
        best_reward = -1.0
        best_metrics: dict[str, Any] = {}

        for threshold in np.linspace(0.20, 0.80, 61):
            adjusted = self._remap_scores(probabilities, threshold)
            reward_value, metrics = reward(adjusted, labels)
            if reward_value > best_reward:
                best_reward = reward_value
                best_threshold = float(threshold)
                best_metrics = metrics

        return best_threshold, best_metrics

    @staticmethod
    def _remap_scores(probabilities: np.ndarray, threshold: float) -> np.ndarray:
        threshold = min(max(float(threshold), 1e-6), 1.0 - 1e-6)
        probabilities = np.asarray(probabilities, dtype=np.float32)

        lower_mask = probabilities <= threshold
        adjusted = np.empty_like(probabilities)
        adjusted[lower_mask] = 0.5 * (probabilities[lower_mask] / threshold)
        adjusted[~lower_mask] = 0.5 + 0.5 * (
            (probabilities[~lower_mask] - threshold) / (1.0 - threshold)
        )
        return np.clip(adjusted, 0.0, 1.0)

    @staticmethod
    def _cache_path_for_config(cfg: MinerTrainingConfig) -> Path:
        human_stats = {
            "path": str(cfg.human_json_path.resolve()),
            "size": cfg.human_json_path.stat().st_size
            if cfg.human_json_path.exists()
            else 0,
            "mtime_ns": cfg.human_json_path.stat().st_mtime_ns
            if cfg.human_json_path.exists()
            else 0,
        }
        fingerprint_payload = {
            **asdict(cfg),
            "human_json_path": str(cfg.human_json_path),
            "cache_dir": str(cfg.cache_dir),
            "feature_version": FEATURE_VERSION,
            "package_version": __version__,
            "human_stats": human_stats,
        }
        digest = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        return cfg.cache_dir / f"miner_model_{digest}.pkl"

    @staticmethod
    def fallback_score_chunk(chunk: Sequence[dict]) -> float:
        if not chunk:
            return 0.5

        per_hand_scores: list[float] = []
        for hand in chunk:
            actions = hand.get("actions") or []
            players = hand.get("players") or []
            streets = hand.get("streets") or []

            action_types = [
                str((action or {}).get("action_type") or "").strip().lower()
                for action in actions
            ]
            action_total = max(1, len(action_types))
            call_ratio = action_types.count("call") / action_total
            check_ratio = action_types.count("check") / action_total
            bet_ratio = action_types.count("bet") / action_total
            raise_ratio = action_types.count("raise") / action_total
            fold_ratio = action_types.count("fold") / action_total
            street_depth = min(len(streets), 3) / 3.0
            player_count_signal = max(0.0, (6 - min(len(players), 6)) / 4.0)

            score = 0.40
            score += 0.20 * street_depth
            score += 0.12 * min(1.0, call_ratio / 0.40)
            score += 0.08 * min(1.0, check_ratio / 0.35)
            score += 0.10 * min(1.0, player_count_signal)
            score += 0.08 * min(1.0, bet_ratio / 0.20)
            score += 0.08 * min(1.0, raise_ratio / 0.16)
            score -= 0.18 * min(1.0, fold_ratio / 0.45)

            per_hand_scores.append(max(0.0, min(1.0, score)))

        return round(float(np.mean(per_hand_scores)), 6)
