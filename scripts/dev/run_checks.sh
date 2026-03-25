#!/usr/bin/env bash
set -euo pipefail

RUN_MINER_BENCH=0

if [[ "${1:-}" == "--with-miner-benchmark" ]]; then
  RUN_MINER_BENCH=1
fi

echo "[checks] Running unit tests..."
python -m unittest discover -s tests -p 'test_*.py'

echo "[checks] Running syntax checks..."
python -m py_compile $(git ls-files '*.py')

if [[ "$RUN_MINER_BENCH" -eq 1 ]]; then
  echo "[checks] Running fast miner benchmark..."
  POKER44_MINER_TRAIN_WINDOWS=2 \
  POKER44_MINER_VALIDATION_WINDOWS=1 \
  POKER44_MINER_CHUNK_COUNT=20 \
  POKER44_MINER_BOT_CANDIDATE_ATTEMPTS=2 \
  POKER44_MINER_MAX_BOT_GENERATION_ROUNDS=1 \
  python scripts/miner/evaluate_miner.py --windows 1
fi

echo "[checks] All checks completed."

