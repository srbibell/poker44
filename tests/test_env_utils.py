from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from poker44.utils.env import env_float, env_int, env_optional_int


class EnvUtilsTests(unittest.TestCase):
    def test_env_int_uses_default_for_missing_or_invalid(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(env_int("POKER44_TEST_INT", 7), 7)

        with patch.dict(os.environ, {"POKER44_TEST_INT": "not-a-number"}, clear=True):
            self.assertEqual(env_int("POKER44_TEST_INT", 7), 7)

    def test_env_int_clamps_to_bounds(self) -> None:
        with patch.dict(os.environ, {"POKER44_TEST_INT": "1"}, clear=True):
            self.assertEqual(
                env_int("POKER44_TEST_INT", 7, minimum=3, maximum=9),
                3,
            )

        with patch.dict(os.environ, {"POKER44_TEST_INT": "99"}, clear=True):
            self.assertEqual(
                env_int("POKER44_TEST_INT", 7, minimum=3, maximum=9),
                9,
            )

    def test_env_float_clamps_to_bounds(self) -> None:
        with patch.dict(os.environ, {"POKER44_TEST_FLOAT": "-5"}, clear=True):
            self.assertAlmostEqual(
                env_float("POKER44_TEST_FLOAT", 0.5, minimum=0.1, maximum=0.9),
                0.1,
            )

        with patch.dict(os.environ, {"POKER44_TEST_FLOAT": "2.5"}, clear=True):
            self.assertAlmostEqual(
                env_float("POKER44_TEST_FLOAT", 0.5, minimum=0.1, maximum=0.9),
                0.9,
            )

    def test_env_optional_int_parsing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(env_optional_int("POKER44_TEST_OPTIONAL_INT"))

        with patch.dict(
            os.environ, {"POKER44_TEST_OPTIONAL_INT": "invalid"}, clear=True
        ):
            self.assertIsNone(env_optional_int("POKER44_TEST_OPTIONAL_INT"))

        with patch.dict(
            os.environ, {"POKER44_TEST_OPTIONAL_INT": "42"}, clear=True
        ):
            self.assertEqual(env_optional_int("POKER44_TEST_OPTIONAL_INT"), 42)


if __name__ == "__main__":
    unittest.main()

