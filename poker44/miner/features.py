"""Chunk-level feature extraction for miner-visible poker payloads."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, Sequence

import numpy as np

FEATURE_VERSION = 2
SANITIZED_BB = 0.02
ACTION_TYPES: tuple[str, ...] = (
    "small_blind",
    "big_blind",
    "ante",
    "check",
    "call",
    "bet",
    "raise",
    "fold",
    "all_in",
    "other",
)
STREETS: tuple[str, ...] = ("preflop", "flop", "turn", "river")


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _ratio(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if denominator <= 0.0:
        return 0.0
    return float(numerator) / denominator


def _mean_std_min_max(values: Sequence[float]) -> list[float]:
    if not values:
        return [0.0, 0.0, 0.0, 0.0]
    arr = np.asarray(values, dtype=np.float32)
    return [
        float(arr.mean()),
        float(arr.std()),
        float(arr.min()),
        float(arr.max()),
    ]


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float32)
    return float(np.quantile(arr, q))


def _entropy_from_counts(counts: Sequence[int]) -> float:
    total = float(sum(counts))
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = float(count) / total
        entropy -= p * math.log(max(p, 1e-12))
    return float(entropy)


def _normalize_action_type(action_type: Any) -> str:
    value = str(action_type or "").strip().lower()
    return value if value in ACTION_TYPES else "other"


def _normalize_street(street: Any) -> str:
    value = str(street or "").strip().lower()
    return value if value in STREETS else "preflop"


def _per_hand_summary(hand: Dict[str, Any]) -> Dict[str, Any]:
    players = hand.get("players") or []
    actions = hand.get("actions") or []
    streets = hand.get("streets") or []

    player_count = float(len(players))
    street_count = float(len(streets))
    stacks = [_safe_float((player or {}).get("starting_stack")) for player in players]

    action_types: list[str] = []
    action_streets: list[str] = []
    action_amounts_bb: list[float] = []
    pot_afters_bb: list[float] = []
    amount_to_pot_ratios: list[float] = []
    actor_seats: list[int] = []
    same_action_repeats = 0
    same_actor_repeats = 0
    street_switches = 0
    action_run = 0
    max_action_run = 0
    actor_run = 0
    max_actor_run = 0

    prev_action_type: str | None = None
    prev_actor_seat: int | None = None
    prev_street: str | None = None

    for action in actions:
        action_type = _normalize_action_type(action.get("action_type"))
        street = _normalize_street(action.get("street"))
        amount_bb = _safe_float(action.get("normalized_amount_bb"))
        pot_after_bb = _safe_float(action.get("pot_after")) / SANITIZED_BB
        actor_seat = _safe_int(action.get("actor_seat"))

        action_types.append(action_type)
        action_streets.append(street)
        action_amounts_bb.append(amount_bb)
        pot_afters_bb.append(pot_after_bb)
        amount_to_pot_ratios.append(_ratio(amount_bb, max(pot_after_bb, 1e-6)))
        actor_seats.append(actor_seat)

        if prev_action_type is not None and action_type == prev_action_type:
            same_action_repeats += 1
            action_run += 1
        else:
            action_run = 1
        max_action_run = max(max_action_run, action_run)

        if prev_actor_seat is not None and actor_seat == prev_actor_seat:
            same_actor_repeats += 1
            actor_run += 1
        else:
            actor_run = 1
        max_actor_run = max(max_actor_run, actor_run)

        if prev_street is not None and street != prev_street:
            street_switches += 1

        prev_action_type = action_type
        prev_actor_seat = actor_seat
        prev_street = street

    total_actions = max(1, len(action_types))
    aggressive_actions = sum(
        1 for action_type in action_types if action_type in {"bet", "raise", "all_in"}
    )
    passive_actions = sum(
        1 for action_type in action_types if action_type in {"check", "call"}
    )
    blind_actions = sum(
        1
        for action_type in action_types
        if action_type in {"small_blind", "big_blind", "ante"}
    )
    unique_actors = len({seat for seat in actor_seats if seat > 0})
    preflop_actions = sum(1 for street in action_streets if street == "preflop")
    all_in_actions = sum(1 for action_type in action_types if action_type == "all_in")
    action_type_counts = Counter(action_types)
    actor_counts = Counter(seat for seat in actor_seats if seat > 0)
    street_counts = Counter(action_streets)

    summary = {
        "player_count": player_count,
        "street_count": street_count,
        "stack_mean": float(np.mean(stacks)) if stacks else 0.0,
        "stack_std": float(np.std(stacks)) if stacks else 0.0,
        "unique_actor_ratio": _ratio(unique_actors, max(len(players), 1)),
        "same_actor_repeat_rate": _ratio(same_actor_repeats, total_actions - 1),
        "same_action_repeat_rate": _ratio(same_action_repeats, total_actions - 1),
        "street_switch_rate": _ratio(street_switches, total_actions - 1),
        "preflop_action_ratio": _ratio(preflop_actions, total_actions),
        "postflop_action_ratio": _ratio(total_actions - preflop_actions, total_actions),
        "aggressive_ratio": _ratio(aggressive_actions, total_actions),
        "passive_ratio": _ratio(passive_actions, total_actions),
        "blind_ratio": _ratio(blind_actions, total_actions),
        "amount_mean_bb": float(np.mean(action_amounts_bb)) if action_amounts_bb else 0.0,
        "amount_std_bb": float(np.std(action_amounts_bb)) if action_amounts_bb else 0.0,
        "amount_max_bb": float(np.max(action_amounts_bb)) if action_amounts_bb else 0.0,
        "amount_median_bb": _quantile(action_amounts_bb, 0.50),
        "amount_p90_bb": _quantile(action_amounts_bb, 0.90),
        "pot_mean_bb": float(np.mean(pot_afters_bb)) if pot_afters_bb else 0.0,
        "pot_max_bb": float(np.max(pot_afters_bb)) if pot_afters_bb else 0.0,
        "amount_to_pot_mean": (
            float(np.mean(amount_to_pot_ratios)) if amount_to_pot_ratios else 0.0
        ),
        "amount_to_pot_std": (
            float(np.std(amount_to_pot_ratios)) if amount_to_pot_ratios else 0.0
        ),
        "all_in_ratio": _ratio(all_in_actions, total_actions),
        "fold_to_aggression_ratio": _ratio(
            action_types.count("fold"),
            aggressive_actions + 1,
        ),
        "actions_per_player": _ratio(total_actions, max(len(players), 1)),
        "actions_per_street": _ratio(total_actions, max(len(streets), 1)),
        "action_entropy": _entropy_from_counts(list(action_type_counts.values())),
        "actor_entropy": _entropy_from_counts(list(actor_counts.values())),
        "street_entropy": _entropy_from_counts(list(street_counts.values())),
        "max_action_run_rate": _ratio(max_action_run, total_actions),
        "max_actor_run_rate": _ratio(max_actor_run, total_actions),
    }

    for action_type in ACTION_TYPES:
        summary[f"{action_type}_ratio"] = _ratio(
            action_type_counts[action_type], total_actions
        )

    if action_types:
        summary["first_action_type"] = action_types[0]
        summary["last_action_type"] = action_types[-1]
    else:
        summary["first_action_type"] = "other"
        summary["last_action_type"] = "other"

    return summary


def _aggregate_bucket_features(
    features: list[float], values: Iterable[float], buckets: Sequence[int]
) -> None:
    counts = Counter(int(value) for value in values)
    total = max(1, sum(counts.values()))
    for bucket in buckets:
        features.append(_ratio(counts[bucket], total))


def extract_chunk_features(chunk: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Convert a miner-visible chunk into a dense numeric feature vector."""
    hand_summaries = [_per_hand_summary(hand) for hand in chunk]

    features: list[float] = [float(len(chunk))]
    player_counts = [summary["player_count"] for summary in hand_summaries]
    street_counts = [summary["street_count"] for summary in hand_summaries]

    aggregate_keys = (
        "stack_mean",
        "stack_std",
        "unique_actor_ratio",
        "same_actor_repeat_rate",
        "same_action_repeat_rate",
        "street_switch_rate",
        "preflop_action_ratio",
        "postflop_action_ratio",
        "aggressive_ratio",
        "passive_ratio",
        "blind_ratio",
        "amount_mean_bb",
        "amount_std_bb",
        "amount_max_bb",
        "amount_median_bb",
        "amount_p90_bb",
        "pot_mean_bb",
        "pot_max_bb",
        "amount_to_pot_mean",
        "amount_to_pot_std",
        "all_in_ratio",
        "fold_to_aggression_ratio",
        "actions_per_player",
        "actions_per_street",
        "action_entropy",
        "actor_entropy",
        "street_entropy",
        "max_action_run_rate",
        "max_actor_run_rate",
    )

    features.extend(_mean_std_min_max(player_counts))
    features.extend(_mean_std_min_max(street_counts))
    _aggregate_bucket_features(features, player_counts, buckets=(2, 3, 4, 5, 6))
    _aggregate_bucket_features(features, street_counts, buckets=(0, 1, 2, 3))

    for key in aggregate_keys:
        values = [summary[key] for summary in hand_summaries]
        features.extend(_mean_std_min_max(values))

    for action_type in ACTION_TYPES:
        values = [summary[f"{action_type}_ratio"] for summary in hand_summaries]
        features.extend(_mean_std_min_max(values))

    overall_action_counts: Counter[str] = Counter()
    first_action_counts: Counter[str] = Counter()
    last_action_counts: Counter[str] = Counter()
    street_action_counts: Counter[tuple[str, str]] = Counter()
    street_totals: Counter[str] = Counter()

    for hand, summary in zip(chunk, hand_summaries):
        first_action_counts[str(summary["first_action_type"])] += 1
        last_action_counts[str(summary["last_action_type"])] += 1

        for action in hand.get("actions") or []:
            action_type = _normalize_action_type(action.get("action_type"))
            street = _normalize_street(action.get("street"))
            overall_action_counts[action_type] += 1
            street_action_counts[(street, action_type)] += 1
            street_totals[street] += 1

    total_actions = max(1, sum(overall_action_counts.values()))
    total_hands = max(1, len(chunk))
    for action_type in ACTION_TYPES:
        features.append(_ratio(overall_action_counts[action_type], total_actions))
        features.append(_ratio(first_action_counts[action_type], total_hands))
        features.append(_ratio(last_action_counts[action_type], total_hands))

    for street in STREETS:
        features.append(_ratio(street_totals[street], total_actions))
        for action_type in ACTION_TYPES:
            features.append(
                _ratio(street_action_counts[(street, action_type)], max(street_totals[street], 1))
            )

    return np.asarray(features, dtype=np.float32)
