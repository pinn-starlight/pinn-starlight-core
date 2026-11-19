"""Shared constants and small output helpers for the release CLI."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from experiments.configs.demo_methods import METHOD_NAMES

TOOL_NAME = "pinn-starlight"
SCHEMA_VERSION = 1

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_ENV = 3
EXIT_INPUT = 4
EXIT_METHOD = 5

METHOD_CHOICES = METHOD_NAMES


def tool_version() -> str:
    """Installed distribution version, falling back to the source checkout."""
    try:
        from importlib.metadata import version

        return version("pinn-starlight-core")
    except Exception:  # noqa: BLE001 - source checkout without an installed dist
        from pinn_starlight_core import __version__

        return __version__


def json_safe(value):
    """Convert numpy / Path / non-finite floats into strict-JSON values."""
    import numpy as np

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def base_payload(command: str, **fields) -> dict:
    """Header shared by every ``--json`` payload of every subcommand."""
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": TOOL_NAME, "version": tool_version()},
        "command": command,
        **fields,
    }


def apply_data_root(args) -> None:
    """Make ``--data-root`` win over every other source (must run before paths import)."""
    import os

    if getattr(args, "data_root", None):
        os.environ["PINN_STARLIGHT_DATA_ROOT"] = str(Path(args.data_root).expanduser())


DATA_ROOT_HELP = (
    "directory that contains collections/ (default: $PINN_STARLIGHT_DATA_ROOT, then the "
    "nearest parent of the current directory with collections/, then the source checkout)"
)


def print_json(payload: dict) -> None:
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False))


def fail(message: str, code: int) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code
