# Testing Guide

This repository does not yet have a large automated test suite, so contributors
should run a small set of focused checks before opening a pull request.

## Environment Setup

Create or activate a local environment, then install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e .
```

## Required Checks

Run these checks for most code changes:

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m py_compile $(git ls-files '*.py')
```

Or run the project helper:

```bash
./scripts/dev/run_checks.sh
```

If your change touches hand parsing or dataset generation, also run:

```bash
python hands_generator/consistency_checker.py
```

## Miner-Focused Checks

If your change touches the miner, also run a local benchmark:

```bash
python scripts/miner/evaluate_miner.py --windows 1
```

Useful overrides for faster local validation:

```bash
POKER44_MINER_TRAIN_WINDOWS=2 \
POKER44_MINER_VALIDATION_WINDOWS=1 \
POKER44_MINER_CHUNK_COUNT=20 \
POKER44_MINER_BOT_CANDIDATE_ATTEMPTS=2 \
POKER44_MINER_MAX_BOT_GENERATION_ROUNDS=1 \
python scripts/miner/evaluate_miner.py --windows 1
```

To run unit/syntax checks plus this fast miner benchmark in one command:

```bash
./scripts/dev/run_checks.sh --with-miner-benchmark
```

## Validator Notes

Validator runtime requires a private human dataset path via
`POKER44_HUMAN_JSON_PATH`, so validator changes should be verified with:

- unit tests where possible;
- the JSON consistency checker for dataset-related code;
- local runtime checks using a private dataset when available.

## Formatting

The style guide recommends `black`. If you have it installed locally, run:

```bash
black .
```
