"""``pinn-starlight version`` - print tool and core versions."""

from __future__ import annotations

import platform

from experiments.cli.context import (
    EXIT_OK,
    TOOL_NAME,
    base_payload,
    print_json,
    tool_version,
)


def cmd_version(args) -> int:
    from pinn_starlight_core import __version__

    payload = base_payload(
        "version",
        version=tool_version(),
        core_version=__version__,
        python=platform.python_version(),
    )
    if args.json:
        print_json(payload)
    else:
        print(f"{TOOL_NAME} {payload['version']} (core {payload['core_version']})")
    return EXIT_OK
