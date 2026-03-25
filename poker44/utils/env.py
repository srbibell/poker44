"""Environment variable parsing helpers with safe defaults."""

from __future__ import annotations

import os
from typing import Callable

LoggerFn = Callable[[str], None] | None


def _warn(logger: LoggerFn, message: str) -> None:
    if logger is not None:
        logger(message)


def env_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    logger: LoggerFn = None,
) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        value = int(default)
    else:
        try:
            value = int(raw)
        except ValueError:
            _warn(
                logger,
                f"Invalid {name}={raw!r}; using default {default}.",
            )
            value = int(default)

    if minimum is not None and value < minimum:
        _warn(logger, f"{name}={value} is below minimum {minimum}; clamping.")
        value = minimum
    if maximum is not None and value > maximum:
        _warn(logger, f"{name}={value} exceeds maximum {maximum}; clamping.")
        value = maximum
    return value


def env_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    logger: LoggerFn = None,
) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        value = float(default)
    else:
        try:
            value = float(raw)
        except ValueError:
            _warn(
                logger,
                f"Invalid {name}={raw!r}; using default {default}.",
            )
            value = float(default)

    if minimum is not None and value < minimum:
        _warn(logger, f"{name}={value} is below minimum {minimum}; clamping.")
        value = float(minimum)
    if maximum is not None and value > maximum:
        _warn(logger, f"{name}={value} exceeds maximum {maximum}; clamping.")
        value = float(maximum)
    return value


def env_optional_int(
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    logger: LoggerFn = None,
) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None

    try:
        value = int(raw)
    except ValueError:
        _warn(logger, f"Invalid {name}={raw!r}; ignoring value.")
        return None

    if minimum is not None and value < minimum:
        _warn(logger, f"{name}={value} is below minimum {minimum}; clamping.")
        value = minimum
    if maximum is not None and value > maximum:
        _warn(logger, f"{name}={value} exceeds maximum {maximum}; clamping.")
        value = maximum
    return value

