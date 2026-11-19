"""``pinn-starlight`` release CLI.

Layering:

* :mod:`experiments.cli.context` - constants, exit codes, JSON helpers (stdlib);
* :mod:`experiments.cli.parser` - argparse definition;
* :mod:`experiments.cli.commands` - one module per subcommand;
* :mod:`experiments.cli.render` - optional PNG preview (lazy matplotlib).

The package is intentionally importable without numpy or torch so ``version``,
``doctor`` and ``samples`` work on a machine that has not installed the
``estimator`` extras yet.  Only ``estimate`` pulls the scientific stack in.
"""

from __future__ import annotations

import sys

from experiments.cli.parser import build_parser

__all__ = ["build_parser", "main"]


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
