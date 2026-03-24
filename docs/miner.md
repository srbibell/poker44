# 🛠️ Poker44 Miner Guide

This guide covers the production-facing miner flow for Poker44 subnet `126`.

---

## Install

```bash
git clone https://github.com/Poker44/Poker44-subnet
cd Poker44-subnet
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e .
python -m pip install bittensor-cli
```

Or use the helper script:

```bash
./scripts/miner/setup.sh
```

---

## Wallet and Registration

`btcli` is provided by the separate `bittensor-cli` package.

```bash
btcli wallet new_coldkey --wallet.name my_cold
btcli wallet new_hotkey --wallet.name my_cold --wallet.hotkey my_poker44_hotkey

btcli subnet register \
  --wallet.name my_cold \
  --wallet.hotkey my_poker44_hotkey \
  --netuid 126 \
  --subtensor.network finney

btcli wallet overview --wallet.name my_cold --subtensor.network finney
```

---

## Run Miner

Script path: `scripts/miner/run/run_miner.sh`

```bash
chmod +x ./scripts/miner/run/run_miner.sh
./scripts/miner/run/run_miner.sh
```

Before using the script, set at least:

- `WALLET_NAME`
- `HOTKEY`
- `AXON_PORT`
- `ALLOWED_VALIDATOR_HOTKEYS` for the recommended Swarm-like allowlist mode

If `ALLOWED_VALIDATOR_HOTKEYS` is left empty, the script falls back to `--blacklist.force_validator_permit`.

The script is environment-driven. Example:

```bash
WALLET_NAME=my_cold \
HOTKEY=my_poker44_hotkey \
AXON_PORT=8091 \
ALLOWED_VALIDATOR_HOTKEYS="validator_hotkey_1 validator_hotkey_2" \
./scripts/miner/run/run_miner.sh
```

Script defaults:

- `POKER44_MINER_STARTUP_MODE=background` so the miner serves immediately
  while model training warms up.
- Set `POKER44_MINER_STARTUP_MODE=blocking` if you want to wait for full
  local training before serving.

PM2:

```bash
pm2 logs poker44_miner
pm2 restart poker44_miner
pm2 stop poker44_miner
pm2 delete poker44_miner
```

Direct CLI example with explicit validator allowlist:

```bash
python neurons/miner.py \
  --netuid 126 \
  --wallet.name my_cold \
  --wallet.hotkey my_poker44_hotkey \
  --subtensor.network finney \
  --axon.port 8091 \
  --blacklist.allowed_validator_hotkeys <validator_hotkey_1> <validator_hotkey_2>
```

Operational note:

- on first start, the miner trains a local cached classifier from the bundled
  public human corpus plus generated bot windows;
- training evaluates multiple model candidates (gradient boosting, random
  forest, and weighted ensemble) and keeps the best validation-reward scorer;
- later starts reuse the cached model unless `POKER44_MINER_FORCE_RETRAIN=1`.
- set `POKER44_MINER_STARTUP_MODE=background` to serve fallback scores
  immediately while local training runs in the background.

Useful miner tuning env vars:

- `POKER44_MINER_TRAIN_WINDOWS` (default `6`)
- `POKER44_MINER_VALIDATION_WINDOWS` (default `2`)
- `POKER44_MINER_STARTUP_MODE` (`blocking` or `background`; model default is `blocking`, while `run_miner.sh` defaults to `background` unless overridden)
- `POKER44_MINER_FORCE_RETRAIN` (default `0`)
- `POKER44_MINER_MODEL_CACHE_DIR` (optional cache location override)

---

## Request/Response Contract

Miners receive `DetectionSynapse(chunks=...)`, where:

- `chunks` is a list of chunks.
- each chunk is a list of hands.
- return exactly one `risk_score` per chunk.

Expected output fields:

- `risk_scores: List[float]` in `[0, 1]`
- `predictions: List[bool]` (optional but recommended)

Important: validator payloads are sanitized to remove label/identity leakage before querying miners.

---

## Production Access Policy

Poker44 miners support two production access modes.

Recommended mode, similar to Swarm:

- `--blacklist.allowed_validator_hotkeys <validator_hotkey...>`

Meaning:

- only the listed validator hotkeys may query your miner;
- requests must be signed and pass Bittensor's default request verification;
- this does not depend on `validator_permit=True` being visible on the metagraph.

Fallback mode:

- `--blacklist.force_validator_permit`

Meaning:

- requests from non-permitted peers are rejected;
- access depends on the caller having `validator_permit=True` on the metagraph.

Operational note:

- if `--blacklist.allowed_validator_hotkeys` is set, the miner uses the allowlist policy;
- if no allowlist is set, the miner falls back to the `validator_permit` policy;
- in both cases, your miner must stay reachable and correctly served on-chain.

---

## Training Data (Miner Side)

Public human corpus:

`hands_generator/human_hands/poker_hands_combined.json.gz`

Bot generation:

```bash
python3 hands_generator/bot_hands/generate_poker_data.py
```

Output:

`hands_generator/bot_hands/bot_hands.json`

Validators evaluate with private human data (`POKER44_HUMAN_JSON_PATH`), not with the public training corpus.

Local benchmark:

```bash
python scripts/miner/evaluate_miner.py --windows 3
```

---

## Health Checklist

- Miner hotkey registered on netuid `126`.
- Axon served and visible on-chain.
- Validator queries are accepted.
- Miner returns non-empty `risk_scores` with correct chunk count.
