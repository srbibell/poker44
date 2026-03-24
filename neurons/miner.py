"""Poker44 miner with cached local training and heuristic fallback."""

import os
import time
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.miner.model import MinerRiskModel
from poker44.validator.synapse import DetectionSynapse


class Miner(BaseMinerNeuron):
    """
    Miner that scores sanitized chunks with a cached local classifier.

    On first launch it trains a local model from the bundled public human corpus
    plus generated bot windows, caches the result, and then serves fast
    inference. If training fails it falls back to a deterministic heuristic.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        cache_dir_env = os.getenv("POKER44_MINER_MODEL_CACHE_DIR", "").strip()
        cache_dir = (
            Path(cache_dir_env).expanduser()
            if cache_dir_env
            else Path(self.config.neuron.full_path) / "miner_model"
        )
        self.risk_model = MinerRiskModel(cache_dir=cache_dir)
        status = self.risk_model.status_snapshot()
        bt.logging.info(
            "Poker44 miner ready "
            f"| startup_mode={status['startup_mode']} "
            f"| model_ready={status['model_ready']} "
            f"| threshold={status['threshold']:.3f} "
            f"| cache={status['cache_path']}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one calibrated bot-risk score per chunk."""
        chunks = synapse.chunks or []
        scores = self.risk_model.score_chunks(chunks)
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        score_source = "model" if self.risk_model.model_ready else "fallback"
        average_score = sum(scores) / len(scores) if scores else 0.0
        bt.logging.info(
            f"Scored {len(chunks)} chunks | source={score_source} "
            f"| avg_score={average_score:.4f}"
        )
        return synapse

    @classmethod
    def score_chunk(cls, chunk: list[dict]) -> float:
        """Cheap heuristic used for fallback-only paths and local sanity checks."""
        return MinerRiskModel.fallback_score_chunk(chunk)

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Determine whether to blacklist incoming requests."""
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Assign priority based on caller's stake."""
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 miner running...")
        while True:
            status = miner.risk_model.status_snapshot()
            bt.logging.info(
                f"Miner UID: {miner.uid} "
                f"| Incentive: {miner.metagraph.I[miner.uid]} "
                f"| model_ready={status['model_ready']} "
                f"| training={status['training_in_progress']} "
                f"| threshold={status['threshold']:.3f}"
            )
            time.sleep(5 * 60)
