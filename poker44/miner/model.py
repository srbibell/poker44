"""Trainable miner-side scoring model with local caching."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import bittensor as bt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

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


def _normalize_startup_mode(value: str) -> Literal["blocking", "background"]:
    normalized = value.strip().lower()
    if normalized == "background":
        return "background"
    return "blocking"


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
    startup_mode: Literal["blocking", "background"] = "blocking"

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
            startup_mode=_normalize_startup_mode(
                os.getenv("POKER44_MINER_STARTUP_MODE", "blocking")
            ),
        )


@dataclass
class WeightedEnsembleModel:
    """Simple weighted-probability ensemble for sklearn-style classifiers."""

    models: tuple[Any, ...]
    weights: tuple[float, ...]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self.models:
            raise ValueError("Ensemble must contain at least one model")
        if len(self.models) != len(self.weights):
            raise ValueError("Ensemble models/weights length mismatch")

        weights = np.asarray(self.weights, dtype=np.float64)
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            weights = np.ones(len(self.models), dtype=np.float64)
            weight_sum = float(weights.sum())
        weights = weights / weight_sum

        positive = np.zeros(x.shape[0], dtype=np.float64)
        for model, weight in zip(self.models, weights):
            proba = np.asarray(model.predict_proba(x), dtype=np.float64)
            if proba.ndim != 2 or proba.shape[1] < 2:
                raise ValueError("Model predict_proba() returned invalid shape")
            positive += float(weight) * proba[:, 1]

        positive = np.clip(positive, 1e-6, 1.0 - 1e-6)
        return np.column_stack((1.0 - positive, positive))


class MinerRiskModel:
    """Cached local model that scores validator-sanitized hand chunks."""

    def __init__(self, *, cache_dir: Path | None = None):
        self.training_cfg = MinerTrainingConfig.from_env(cache_dir=cache_dir)
        self._lock = threading.RLock()
        self._training_thread: threading.Thread | None = None
        self._is_training = False
        self.model: Any | None = None
        self.threshold: float = 0.5
        self.metrics: dict[str, Any] = {}
        self.cache_path = self._cache_path_for_config(self.training_cfg)
        self._initialize_model()

    @property
    def model_ready(self) -> bool:
        with self._lock:
            return self.model is not None

    @property
    def training_in_progress(self) -> bool:
        with self._lock:
            return self._is_training

    def status_snapshot(self) -> dict[str, Any]:
        with self._lock:
            metrics_status_default = "trained" if self.model is not None else "unknown"
            return {
                "startup_mode": self.training_cfg.startup_mode,
                "model_ready": self.model is not None,
                "training_in_progress": self._is_training,
                "threshold": round(float(self.threshold), 6),
                "cache_path": str(self.cache_path),
                "metrics_status": str(
                    self.metrics.get("status", metrics_status_default)
                ),
            }

    def score_chunks(self, chunks: Sequence[Sequence[dict]]) -> list[float]:
        if not chunks:
            return []

        with self._lock:
            model = self.model
            threshold = self.threshold

        if model is None:
            return [self.fallback_score_chunk(chunk) for chunk in chunks]

        try:
            feature_matrix = np.vstack(
                [extract_chunk_features(chunk) for chunk in chunks]
            )
            probabilities = model.predict_proba(feature_matrix)[:, 1]
            adjusted = self._remap_scores(probabilities, threshold)
            return [round(float(score), 6) for score in adjusted]
        except Exception as err:
            bt.logging.warning(
                f"Miner model inference failed, using heuristic fallback: {err}"
            )
            return [self.fallback_score_chunk(chunk) for chunk in chunks]

    def score_chunk(self, chunk: Sequence[dict]) -> float:
        scores = self.score_chunks([chunk])
        return scores[0] if scores else 0.5

    def _initialize_model(self) -> None:
        if self.training_cfg.startup_mode == "background":
            loaded = self._load_cached_model(ignore_force_retrain=True)
            if self.training_cfg.force_retrain or not loaded:
                self._start_background_training()
            if not loaded:
                with self._lock:
                    self.metrics = {
                        "status": "warming_up",
                        "note": "serving fallback until training finishes",
                    }
            return

        loaded = self._load_cached_model(ignore_force_retrain=False)
        if loaded:
            return
        try:
            self._train_and_cache()
        except Exception as err:
            bt.logging.warning(
                "Miner model training failed, "
                f"keeping heuristic fallback only: {err}"
            )
            with self._lock:
                self.model = None
                self.threshold = 0.5
                self.metrics = {"status": "fallback_only", "error": str(err)}

    def _load_cached_model(self, *, ignore_force_retrain: bool) -> bool:
        if self.training_cfg.force_retrain and not ignore_force_retrain:
            return False
        if not self.cache_path.exists():
            return False

        try:
            payload = pickle.loads(self.cache_path.read_bytes())
            loaded_model = payload["model"]
            loaded_threshold = float(payload["threshold"])
            loaded_metrics = dict(payload.get("metrics") or {})
            loaded_metrics.setdefault("status", "trained")
            with self._lock:
                self.model = loaded_model
                self.threshold = loaded_threshold
                self.metrics = loaded_metrics
            bt.logging.info(
                "Loaded cached miner model from "
                f"{self.cache_path} | threshold={loaded_threshold:.3f}"
            )
            return True
        except Exception as err:
            bt.logging.warning(
                f"Failed to load cached miner model, retraining: {err}"
            )
            return False

    def _start_background_training(self) -> None:
        with self._lock:
            if self._is_training:
                return
            self._is_training = True

        self._training_thread = threading.Thread(
            target=self._background_training_worker,
            name="poker44-miner-trainer",
            daemon=True,
        )
        self._training_thread.start()
        bt.logging.info(
            "Started background miner model training "
            f"| cache={self.cache_path}"
        )

    def _background_training_worker(self) -> None:
        try:
            self._train_and_cache()
        except Exception as err:
            bt.logging.warning(
                "Background miner model training failed, "
                f"continuing with current scorer: {err}"
            )
            with self._lock:
                if self.model is None:
                    self.threshold = 0.5
                    self.metrics = {"status": "fallback_only", "error": str(err)}
                else:
                    self.metrics = {
                        **self.metrics,
                        "status": "trained_with_refresh_error",
                        "background_error": str(err),
                    }
        finally:
            with self._lock:
                self._is_training = False

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

        hist_model = HistGradientBoostingClassifier(
            max_depth=5,
            learning_rate=0.05,
            max_iter=350,
            min_samples_leaf=8,
            l2_regularization=0.05,
            random_state=self.training_cfg.seed,
        )
        hist_model.fit(train_x, train_y)

        rf_model = RandomForestClassifier(
            n_estimators=220,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=self.training_cfg.seed,
            n_jobs=-1,
        )
        rf_model.fit(train_x, train_y)

        ensemble_model = WeightedEnsembleModel(
            models=(hist_model, rf_model),
            weights=(0.6, 0.4),
        )

        candidate_models: dict[str, Any] = {
            "hist_gradient_boosting": hist_model,
            "random_forest": rf_model,
            "weighted_ensemble": ensemble_model,
        }

        candidate_results: list[dict[str, float | str]] = []
        best_model_name = "hist_gradient_boosting"
        best_model: Any = hist_model
        best_threshold = 0.5
        best_validation_reward = -1.0
        best_validation_metrics: dict[str, Any] = {}

        for model_name, candidate_model in candidate_models.items():
            validation_probs = candidate_model.predict_proba(validation_x)[:, 1]
            threshold, _ = self._select_threshold(validation_probs, validation_y)
            remapped_validation_scores = self._remap_scores(validation_probs, threshold)
            validation_reward, validation_summary = reward(
                remapped_validation_scores, validation_y
            )
            validation_brier = self._brier_score(validation_probs, validation_y)
            candidate_results.append(
                {
                    "model": model_name,
                    "selected_threshold": round(float(threshold), 6),
                    "validation_reward": round(float(validation_reward), 6),
                    "validation_brier": round(float(validation_brier), 6),
                }
            )
            if validation_reward > best_validation_reward:
                best_validation_reward = float(validation_reward)
                best_model_name = model_name
                best_model = candidate_model
                best_threshold = float(threshold)
                best_validation_metrics = validation_summary

        candidate_results.sort(
            key=lambda item: float(item["validation_reward"]), reverse=True
        )

        train_probs = best_model.predict_proba(train_x)[:, 1]
        train_scores = self._remap_scores(train_probs, best_threshold)
        train_reward, train_metrics = reward(train_scores, train_y)
        validation_probs = best_model.predict_proba(validation_x)[:, 1]
        validation_reward, validation_summary = reward(
            self._remap_scores(validation_probs, best_threshold),
            validation_y,
        )
        validation_brier = self._brier_score(validation_probs, validation_y)

        self.training_cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "bundle_version": 1,
            "model": best_model,
            "threshold": best_threshold,
            "metrics": {
                "status": "trained",
                "feature_version": FEATURE_VERSION,
                "selected_model": best_model_name,
                "candidate_results": candidate_results,
                "train_reward": train_reward,
                "train_metrics": train_metrics,
                "validation_reward": validation_reward,
                "validation_metrics": validation_summary,
                "validation_brier": validation_brier,
                "selected_threshold": best_threshold,
                "train_examples": int(train_y.size),
                "validation_examples": int(validation_y.size),
                "train_window_ids": train_window_ids,
                "validation_window_ids": validation_window_ids,
                "training_seconds": round(time.time() - start, 2),
                "package_version": __version__,
            },
        }
        self.cache_path.write_bytes(pickle.dumps(payload))

        with self._lock:
            self.model = best_model
            self.threshold = best_threshold
            self.metrics = dict(payload["metrics"])
        bt.logging.info(
            "Finished miner model training "
            f"| model={best_model_name} reward={validation_reward:.4f} "
            f"threshold={best_threshold:.3f} "
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
    def _brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
        probs = np.asarray(probabilities, dtype=np.float32)
        y_true = np.asarray(labels, dtype=np.float32)
        return float(np.mean((probs - y_true) ** 2))

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
        cfg_fingerprint = asdict(cfg)
        # Runtime toggles should not fragment the persisted model cache.
        cfg_fingerprint.pop("force_retrain", None)
        cfg_fingerprint.pop("startup_mode", None)
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
            **cfg_fingerprint,
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
