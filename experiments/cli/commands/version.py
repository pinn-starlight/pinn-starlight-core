"""``pinn-starlight version`` - print tool and core versions."""

from __future__ import annotations

import platform

from experiments.cli.context import (
    EXIT_OK,
    SCHEMA_VERSION,
    TOOL_NAME,
    print_json,
    tool_version,
)


def cmd_version(args) -> int:
    from pinn_starlight_core import __version__

    payload = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "version": tool_version(),
        "core_version": __version__,
        "python": platform.python_version(),
    }
    if args.json:
        print_json(payload)
    else:
        print(f"{TOOL_NAME} {payload['version']} (core {payload['core_version']})")
    return EXIT_OK
