"""Poker44 miner with cached local training and heuristic fallback."""

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
        cache_dir = Path(self.config.neuron.full_path) / "miner_model"
        self.risk_model = MinerRiskModel(cache_dir=cache_dir)
        bt.logging.info(
            "Poker44 miner ready "
            f"| threshold={self.risk_model.threshold:.3f} "
            f"| cache={self.risk_model.cache_path}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one calibrated bot-risk score per chunk."""
        chunks = synapse.chunks or []
        scores = self.risk_model.score_chunks(chunks)
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        bt.logging.info(
            f"Scored {len(chunks)} chunks | predictions={synapse.predictions}"
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
            bt.logging.info(
                f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}"
            )
            time.sleep(5 * 60)
