"""``python -m experiments.cli`` entry point.

The ``__name__`` guard keeps this importable: the smoke test imports every
module under ``experiments/``, and an unguarded ``main()`` call would terminate
the run with an argparse usage error.
"""

from experiments.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
