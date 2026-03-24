# Debugging Guide

This document covers the most common local issues contributors hit while
working on Poker44.

## Dependency Problems

If imports fail, recreate the environment and reinstall:

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e .
```

If editable install fails while trying to download build dependencies, use
`--no-build-isolation`.

## Miner Starts Slowly

The upgraded miner trains a cached local model on first start. This can take
noticeable time depending on the training settings.

Helpful environment variables:

- `POKER44_MINER_TRAIN_WINDOWS`
- `POKER44_MINER_VALIDATION_WINDOWS`
- `POKER44_MINER_CHUNK_COUNT`
- `POKER44_MINER_FORCE_RETRAIN`
- `POKER44_MINER_MODEL_CACHE_DIR`

If you want a faster local smoke test, lower the training windows and chunk
count temporarily.

## Validator Dataset Issues

The validator requires:

- `POKER44_HUMAN_JSON_PATH` to point to a valid local JSON dataset;
- the file to exist on disk before startup.

If validator startup fails, check:

```bash
ls -lh "$POKER44_HUMAN_JSON_PATH"
python hands_generator/consistency_checker.py
```

## Benchmark and Data Generation Are Slow

`build_mixed_labeled_chunks()` performs real bot-hand generation, so miner
benchmarking is intentionally heavier than a unit test.

To make local checks faster:

- reduce `POKER44_MINER_TRAIN_WINDOWS`;
- reduce `POKER44_MINER_VALIDATION_WINDOWS`;
- reduce `POKER44_MINER_CHUNK_COUNT`;
- reduce `POKER44_MINER_BOT_CANDIDATE_ATTEMPTS`;
- reduce `POKER44_MINER_MAX_BOT_GENERATION_ROUNDS`.

## PM2 Runtime Debugging

Common commands:

```bash
pm2 logs poker44_miner
pm2 restart poker44_miner
pm2 logs poker44_validator
pm2 restart poker44_validator
```

## Reporting a Good Bug

When opening an issue, include:

- the exact command you ran;
- the relevant environment variables;
- the commit hash you are on;
- the full traceback or log excerpt;
- whether the problem is reproducible.
