"""argparse definition for ``pinn-starlight``.

Importing this module must stay cheap (no numpy / torch): ``version``, ``doctor``
and ``samples`` should answer in milliseconds on a fresh machine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.cli.commands.doctor import cmd_doctor
from experiments.cli.commands.estimate import cmd_estimate
from experiments.cli.commands.samples import cmd_samples
from experiments.cli.commands.version import cmd_version
from experiments.cli.context import TOOL_NAME, tool_version
from experiments.configs.demo_methods import MODE_TO_METHOD
from experiments.configs.formal_experiment import FORMAL_SEEDS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=TOOL_NAME,
        description="PINN-Starlight demo CLI (not the formal experiment pipeline).",
    )
    parser.add_argument("--version", action="version", version=f"{TOOL_NAME} {tool_version()}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    version = subparsers.add_parser("version", help="print the tool and core versions")
    version.add_argument("--json", action="store_true")
    version.set_defaults(func=cmd_version)

    doctor = subparsers.add_parser("doctor", help="report environment and data readiness")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--data-root", type=Path, default=None)
    doctor.set_defaults(func=cmd_doctor)

    samples = subparsers.add_parser("samples", help="list validation samples on disk")
    samples.add_argument("--stage", choices=("a", "b", "all"), default="all")
    samples.add_argument("--json", action="store_true")
    samples.add_argument("--data-root", type=Path, default=None)
    samples.set_defaults(func=cmd_samples)

    estimate = subparsers.add_parser("estimate", help="estimate the background of one image")
    source = estimate.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="any supported image file")
    source.add_argument("--sample", help="validation sample_id from the extracted dataset")
    estimate.add_argument("--dataset", choices=("stage_a", "stage_b"), default="stage_a")
    estimate.add_argument(
        "--methods",
        default="all",
        help="all or comma separated fft_gaussian,fourier_ls,pinn,final_pinn,balanced",
    )
    estimate.add_argument(
        "--mode",
        choices=tuple(MODE_TO_METHOD),
        default=None,
        help="shortcut: star=final_pinn, nebula=pinn, balanced=anchored",
    )
    estimate.add_argument("--max-side", type=int, default=384)
    estimate.add_argument("--steps", type=int, default=500)
    estimate.add_argument("--seed", type=int, default=FORMAL_SEEDS[0])
    estimate.add_argument("--device", default="auto")
    estimate.add_argument("--out", type=Path, default=None)
    estimate.add_argument("--json", action="store_true")
    estimate.add_argument("--no-figure", action="store_true")
    estimate.add_argument("--quiet", action="store_true")
    estimate.add_argument("--data-root", type=Path, default=None)
    estimate.set_defaults(func=cmd_estimate)
    return parser
