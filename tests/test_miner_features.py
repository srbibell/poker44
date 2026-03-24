from __future__ import annotations

import unittest

import numpy as np

from poker44.miner.features import extract_chunk_features


def make_hand(
    *,
    player_count: int,
    street_count: int,
    action_types: list[str],
    amount_bb: float,
) -> dict:
    players = [
        {
            "player_uid": f"seat_{seat}",
            "seat": seat,
            "starting_stack": 2.0 + (seat * 0.1),
            "hole_cards": None,
            "showed_hand": False,
        }
        for seat in range(1, player_count + 1)
    ]
    streets = [
        {"street": street_name, "board_cards": []}
        for street_name in ("flop", "turn", "river")[:street_count]
    ]
    actions = []
    for index, action_type in enumerate(action_types, start=1):
        actions.append(
            {
                "action_id": str(index),
                "street": "preflop" if index <= len(action_types) // 2 else "flop",
                "actor_seat": ((index - 1) % max(player_count, 1)) + 1,
                "action_type": action_type,
                "amount": round(amount_bb * 0.02, 4),
                "raise_to": None,
                "call_to": None,
                "normalized_amount_bb": amount_bb,
                "pot_before": round(index * 0.04, 4),
                "pot_after": round((index + 1) * 0.04, 4),
            }
        )
    return {
        "metadata": {
            "game_type": "Hold'em",
            "limit_type": "No Limit",
            "max_seats": 6,
            "hero_seat": 0,
            "hand_ended_on_street": "",
            "button_seat": 0,
            "sb": 0.01,
            "bb": 0.02,
            "ante": 0.0,
            "rng_seed_commitment": None,
        },
        "players": players,
        "streets": streets,
        "actions": actions,
        "outcome": {
            "winners": [],
            "payouts": {},
            "total_pot": 0.0,
            "rake": 0.0,
            "result_reason": "",
            "showdown": False,
        },
    }


class MinerFeatureTests(unittest.TestCase):
    def test_feature_vector_has_stable_shape(self) -> None:
        empty_features = extract_chunk_features([])
        sample_chunk = [
            make_hand(
                player_count=6,
                street_count=1,
                action_types=["small_blind", "big_blind", "call", "check", "fold"],
                amount_bb=1.0,
            ),
            make_hand(
                player_count=3,
                street_count=3,
                action_types=["small_blind", "big_blind", "call", "raise", "bet"],
                amount_bb=4.0,
            ),
        ]
        populated_features = extract_chunk_features(sample_chunk)

        self.assertEqual(empty_features.shape, populated_features.shape)
        self.assertGreater(populated_features.size, 100)

    def test_feature_vector_changes_with_chunk_behavior(self) -> None:
        human_chunk = [
            make_hand(
                player_count=6,
                street_count=1,
                action_types=["small_blind", "big_blind", "fold", "check", "fold"],
                amount_bb=0.5,
            )
        ]
        bot_chunk = [
            make_hand(
                player_count=3,
                street_count=3,
                action_types=["small_blind", "big_blind", "call", "raise", "bet"],
                amount_bb=5.0,
            )
        ]

        human_features = extract_chunk_features(human_chunk)
        bot_features = extract_chunk_features(bot_chunk)

        self.assertFalse(np.allclose(human_features, bot_features))


if __name__ == "__main__":
    unittest.main()
